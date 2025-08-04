import contextlib
import gc
import os
from collections import defaultdict
from typing import Any, List, Union

import torch
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from trl.data_utils import apply_chat_template
from trl.models import create_reference_model
from trl.trainer.grpo_config import GRPOConfig

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    """

    def __init__(self, models: List[Union[str, Any]], **kwargs):
        """
        Initialize the GRPO trainer module.

        Args:
            models: List containing the model name (string) or model object to be trained.
            **kwargs: Additional arguments for configuration.
        """
        # Extract model and reward functions
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        model_input = models[0]
        
        # ✅ SMART LOADING: Handle both strings and model objects
        if isinstance(model_input, str):
            print(f"Loading model from string: {model_input}")
            self.model = self._smart_load_model(model_input)
            self._model_name = model_input
        else:
            # Assume it's already a model object
            self.model = model_input
            self._model_name = getattr(self.model.config, '_name_or_path', 'unknown')

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
            self.num_generations > 1
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

    def _smart_load_model(self, model_name: str):
        """
        Smart model loading with automatic 4-bit quantization detection
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Loaded model with optimal configuration
        """
        print(f"🚀 Smart loading model: {model_name}")
        
        # Get available GPU memory
        available_memory_gb = self._get_available_gpu_memory()
        
        # Estimate model size from name
        model_size_b = self._estimate_model_size(model_name)
        
        print(f"📊 Available GPU memory: {available_memory_gb:.1f}GB")
        print(f"📊 Estimated model size: {model_size_b:.1f}B parameters")
        
        # Determine best loading strategy
        should_use_4bit = self._should_use_4bit(available_memory_gb, model_size_b)
        
        if should_use_4bit:
            return self._load_4bit_model(model_name)
        else:
            return self._load_standard_model(model_name)
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            torch.cuda.empty_cache()
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            
            available_memory = total_memory - allocated_memory
            available_gb = available_memory / (1024**3)
            
            return max(available_gb, 1.0)  # At least 1GB
            
        except Exception as e:
            print(f"⚠️ Error checking GPU memory: {e}")
            return 4.0  # Safe default
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in billions of parameters from name"""
        model_name_lower = model_name.lower()
        
        if "0.5b" in model_name_lower or "500m" in model_name_lower:
            return 0.5
        elif "0.6b" in model_name_lower or "600m" in model_name_lower:
            return 0.6
        elif "1.5b" in model_name_lower or "1500m" in model_name_lower:
            return 1.5
        elif "1.7b" in model_name_lower or "1700m" in model_name_lower:
            return 1.7
        elif "3b" in model_name_lower:
            return 3.0
        elif "7b" in model_name_lower:
            return 7.0
        else:
            return 1.5  # Default assumption
    
    def _should_use_4bit(self, available_memory_gb: float, model_size_b: float) -> bool:
        """Determine if 4-bit quantization should be used"""
        # Rough memory estimates with training overhead (3x multiplier):
        # - FP32: 4 bytes per param * 3 (training overhead)
        # - 4-bit: 0.5 bytes per param * 3 (training overhead)
        
        estimated_4bit_gb = (model_size_b * 0.5 * 3) / 1000  # Convert MB to GB
        estimated_fp32_gb = (model_size_b * 4 * 3) / 1000
        
        print(f"📊 Estimated memory needed:")
        print(f"   - 4-bit: {estimated_4bit_gb:.1f}GB")
        print(f"   - FP32:  {estimated_fp32_gb:.1f}GB")
        
        # Use 4-bit if:
        # 1. Available memory is less than FP32 requirement + 2GB buffer
        # 2. OR model is large (>1B params) and memory is limited
        
        if available_memory_gb < (estimated_fp32_gb + 2.0):
            print("✅ Using 4-bit: Memory constrained")
            return True
        elif model_size_b > 1.0 and available_memory_gb < 8.0:
            print("✅ Using 4-bit: Large model + limited memory")
            return True
        else:
            print("✅ Using FP32: Sufficient memory available")
            return False
    
    def _load_4bit_model(self, model_name: str):
        """Load model with 4-bit quantization"""
        print("🔥 Loading with 4-bit quantization...")
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Check actual memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                print(f"✅ 4-bit model loaded successfully! GPU memory: {memory_used:.1f}GB")
            
            return model
            
        except Exception as e:
            print(f"⚠️ 4-bit loading failed: {e}")
            print("🔄 Falling back to standard loading...")
            return self._load_standard_model(model_name)
    
    def _load_standard_model(self, model_name: str):
        """Load model with standard precision"""
        print("⚡ Loading with standard precision...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Check actual memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"✅ Standard model loaded successfully! GPU memory: {memory_used:.1f}GB")
        
        return model

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
            # Use the stored model name for tokenizer
            model_name = getattr(self, '_model_name', None)
            if model_name is None:
                model_name = getattr(self.model.config, '_name_or_path', None)
            
            if model_name is None:
                raise ValueError("Cannot determine model name for tokenizer initialization")
            
            self.processing_class = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
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

    # Rest of the methods remain unchanged...
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
        """Generate outputs from the model for the given inputs."""
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

    def cleanup(self):
        """Clean up resources at the end of training."""
        self.cleanup_trackers()
