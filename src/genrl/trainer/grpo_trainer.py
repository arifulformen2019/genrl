import contextlib
import gc
import os
from collections import defaultdict
from typing import Any, List

import torch
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl.data_utils import apply_chat_template
from trl.models import create_reference_model
from trl.trainer.grpo_config import GRPOConfig

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule

# PERFORMANCE NOTE: For better CPU performance, set this environment variable
# before running your script to match your CPU's physical core count.
# For example: export OMP_NUM_THREADS=8


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module.

        Args:
            models: List containing the model to be trained.
            **kwargs: Additional arguments for configuration.
        """
        # Extract model and reward functions
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        self.model = models[0]

        # Configuration parameters
        config = kwargs.get("config", None)
        self.args = (
            config
            if isinstance(config, GRPOConfig)
            else GRPOConfig(config) if config else GRPOConfig()
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        # Tokenizers
        self.processing_class = kwargs.get("processing_class", None)

        # Additional parameters
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0
        self.num_generations = kwargs.get("num_generations", 2)
        assert (
            self.num_generations ≥ 1
        ), f"For GRPO training, number of generations must be > 1, got {self.num_generations}"
        self.epsilon = kwargs.get("epsilon", 0.2)
        self.epsilon_high = kwargs.get("epsilon_high", 0.28)
        self.beta = kwargs.get("beta", 0.0)
        
        # OPTIMIZATION: Conditionally disable gradient checkpointing on CPU.
        self.enable_gradient_checkpointing = kwargs.get(
            "enable_gradient_checkpointing", True
        )
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.autocast = torch.amp.autocast(
                device_type=self.device.type, enabled=self.args.fp16
            )
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.autocast = contextlib.nullcontext()
        else:
            self.device = torch.device("cpu")
            self.autocast = contextlib.nullcontext()
            # It's counter-productive on CPU as it trades compute for memory.
            if self.enable_gradient_checkpointing:
                print("INFO: CPU detected. Disabling gradient checkpointing for better performance.")
                self.enable_gradient_checkpointing = False

        # Initialize core components
        self._initialize_model(self.enable_gradient_checkpointing)
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    def _initialize_model(self, enable_gradient_checkpointing):
        """Initialize the model and reference model."""
        self.model = self.model.to(self.device)
        if enable_gradient_checkpointing:
            print("INFO: Enabling gradient checkpointing.")
            self.model.gradient_checkpointing_enable()

        # Reference model setup
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.model).to(self.model.device)

    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model and reward models."""
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, padding_side="left"
            )

    def _initialize_metrics(self):
        """Initialize metrics tracking for training and evaluation."""
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
        """Set generation config."""
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            # OPTIMIZATION: Set num_return_sequences for efficient batched generation.
            num_return_sequences=self.num_generations,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        if with_template:
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.num_generations):
                        templated_prompts.append(
                            apply_chat_template(item, self.processing_class)["prompt"]
                        )
            else:
                templated_prompts = [
                    apply_chat_template(item, self.processing_class)["prompt"]
                    for item in inputs
                ]
        else:
            if for_training:
                templated_prompts = [output for generations in inputs for output in generations]
            else:
                templated_prompts = [item[0] for item in inputs]

        input_tokens = self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens

    def generate(
        self, inputs: Any, return_completion_ids: bool = False, stage=0
    ) -> Any:
        """
        OPTIMIZATION: Rewritten to use a single, efficient batch generation call.
        
        Generate outputs from the model for the given inputs.

        Args:
            inputs: Input data for generation
            return_completion_ids: Whether to return completion IDs along with text
            stage: Current stage (0, 1, or 2) for proper output formatting

        Returns:
            Generated outputs in the format expected by the next stage
        """
        input_tokens = self._process_inputs(inputs)
        prompt_length = input_tokens.input_ids.size(1)

        with torch.no_grad():
            # Single, batched generation call
            outputs = self.model.generate(
                input_tokens.input_ids.to(self.model.device),
                attention_mask=input_tokens.attention_mask.to(self.model.device),
                generation_config=self.generation_config,
            )

        # Extract completions (i.e., removes prompt part)
        completion_ids = outputs[:, prompt_length:]
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        
        # Reshape the flat list of completions into [batch_size, num_generations]
        num_prompts = input_tokens.input_ids.size(0)
        rollout = [
            completions_text[i : i + self.num_generations]
            for i in range(0, len(completions_text), self.num_generations)
        ]

        if return_completion_ids:
            # Reshape completion_ids tensor and convert to list of tensors
            completion_ids_reshaped = completion_ids.view(
                num_prompts, self.num_generations, -1
            ).tolist()
            rollout_ids = [
                [torch.tensor(cid, device=completion_ids.device) for cid in batch] 
                for batch in completion_ids_reshaped
            ]
            return rollout, rollout_ids
        else:
            return rollout

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Get the per-token log probabilities for the input tokens."""
        model = model.to(input_ids.device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]

        loss_mask = attention_mask[:, -logits_to_keep:].to(dtype=logits.dtype).contiguous()
        labels = input_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:].contiguous()
        logits = logits / self.args.temperature
        
        logits_shape = logits.shape
        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.view(-1, logits_shape[-1]),
            labels.view(-1),
            reduction="none",
        ).view(logits_shape[0], logits_shape[1])
        
        token_log_probs = (
            token_log_probs * loss_mask
            + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        )
        return token_log_probs

    def compute_loss(
        self, model, inputs, num_items_in_batch=1, mode="train", return_metrics=False
    ):
        """Compute the GRPO loss."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(self.model.device)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        if self.beta != 0.0:
            ref_per_token_logps = (
                self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
                if self.ref_model is not None
                else per_token_logps.clone()
            )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.args.num_iterations > 1
            else per_token_logps.detach()
        )

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.epsilon,
            1 + self.epsilon_high if self.epsilon_high is not None else self.epsilon,
        )
        
        per_token_loss = -torch.min(coef_1 * advantages.unsqueeze(dim=-1), coef_2 * advantages.unsqueeze(dim=-1))

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Metrics calculation
        mean_kl = None
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.item())

        is_clipped = (coef_1 > coef_2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        if return_metrics:
            metrics = {
                "loss": loss.item(),
                "kl": mean_kl.item() if mean_kl is not None else None,
                "clip_ratio": clip_ratio.item(),
            }
            return loss, metrics
        else:
            return loss

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """Train the model using the given game state and reward manager."""
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(
                stage, state, data_manager, reward_manager, global_step
            )
        self.global_step = global_step
        self.model.eval()

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        """Perform a single training step."""
        global_step += 1

        stage_inputs = state.get_stage_state(stage)
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
        assert stage_inputs is not None, f"No inputs found for stage {stage}"
        
        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [
            stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx in range(len(index_mapping))
        ]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"

        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"] = processed_inputs.input_ids.to(self.model.device)
        model_inputs["prompt_mask"] = processed_inputs.attention_mask.to(self.model.device)
        
        processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
        model_inputs["completion_ids"] = processed_outputs.input_ids.to(self.model.device)
        model_inputs["completion_mask"] = processed_outputs.attention_mask.to(self.model.device)

        rewards_raw = reward_manager[stage]
        rewards = [
            rewards_raw[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx in range(len(index_mapping))
        ]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards, device=self.model.device)

        with torch.no_grad():
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            if rewards.shape[1] > 1:
                advantages /= rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = torch.flatten(advantages)

        model_inputs["advantages"] = advantages
        model_inputs["old_per_token_logps"] = None

        with self.autocast:
            loss = self.compute_loss(self.model, model_inputs)

        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()

        metrics = {"train/loss": loss.cpu().mean().item(), "train/rewards": rewards.cpu().mean().item()}
        self.log(metrics, global_step)

        self.cleanup_step()
        return global_step

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        pass
    
    def save(self, save_dir: str) -> None:
        """Save the model and trainer state to the given directory."""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)

        torch.save(
            {
                "metrics": self._metrics,
                "total_train_tokens": self._total_train_tokens,
                "generation_config": self.generation_config,
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(save_dir, "trainer_state.pt"),
        )

    @classmethod
    def load(cls, load_dir: str, **kwargs) -> "GRPOLanguageTrainerModule":
        """Load a trainer module from the given directory."""
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        trainer = cls([model], **kwargs)

        trainer_state_path = os.path.join(load_dir, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            trainer_state = torch.load(trainer_state_path, map_location=trainer.device)
            trainer._metrics = trainer_state.get("metrics", trainer._metrics)
            trainer._total_train_tokens = trainer_state.get("total_train_tokens", 0)
            trainer.generation_config = trainer_state.get("generation_config", trainer.generation_config)
            trainer.optimizer.load_state_dict(trainer_state.get("optimizer"))
        return trainer

    def cleanup_step(self):
        """Clean up resources after a training step."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            # Note: torch.mps.empty_cache() is available in newer PyTorch versions
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        
        # OPTIMIZATION: Frequent garbage collection can severely harm performance by
        # pausing execution. It's better to remove it and let Python's automatic
        # GC handle memory management.
        # gc.collect()

    def cleanup(self):
        """Clean up resources at the end of training."""
        self.cleanup_trackers()
