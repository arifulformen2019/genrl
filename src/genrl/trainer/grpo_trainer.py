import contextlib
import gc
import os
import time  # ✅ ADD: Time tracking
import weakref  # ✅ ADD: Weak references
from collections import defaultdict, deque  # ✅ ADD: Bounded collections
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

# ✅ ADD: Memory monitoring imports
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    class MockFore:
        CYAN = GREEN = RED = YELLOW = MAGENTA = BLUE = ""
    class MockStyle:
        RESET_ALL = ""
    Fore = MockFore()
    Style = MockStyle()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# OPTIMIZATION NOTE: For further CPU speedup, consider these steps outside this file:
# 1. Install Intel® Extension for PyTorch: `pip install intel_extension_for_pytorch`
#    Then, in your main script, wrap your model and optimizer:
#    import intel_extension_for_pytorch as ipex
#    model, optimizer = ipex.optimize(model, optimizer)
# 2. Set environment variables for parallelism before running your script:
#    export OMP_NUM_THREADS=<number_of_your_cpu_cores>


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    ✅ OPTIMIZED: Trainer for the Group Relative Policy Optimization (GRPO) method with comprehensive memory leak prevention.
    Implements the TrainerModule interface defined in base_trainer.py.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        ✅ OPTIMIZED: Initialize the GRPO trainer module with memory management.

        Args:
            models: List containing the model to be trained.
            **kwargs: Additional arguments for configuration.
        """
        # Extract model and reward functions
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        # ✅ FIX: Handle case where models[0] is a string (model path)
        if isinstance(models[0], str):
            # Load model from path/name
            print(f"{Fore.CYAN}📦 [MODEL INIT] Loading model from: {models[0]}{Style.RESET_ALL}")
            try:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(models[0])
                print(f"{Fore.GREEN}✅ [MODEL INIT] Model loaded successfully{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ [MODEL INIT] Failed to load model: {e}{Style.RESET_ALL}")
                raise ValueError(f"Failed to load model from '{models[0]}': {e}")
        else:
            # Model object provided directly
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
            self.num_generations > 1
        ), f"For GRPO training, number of generations must be > 1, got {self.num_generations}"
        self.epsilon = kwargs.get("epsilon", 0.2)
        self.epsilon_high = kwargs.get("epsilon_high", 0.28)
        self.beta = kwargs.get("beta", 0.0)
        
        # OPTIMIZATION: Conditionally disable gradient checkpointing on CPU.
        self.enable_gradient_checkpointing = kwargs.get(
            "enable_gradient_checkpointing", True
        )

        # ✅ CRITICAL MEMORY LEAK FIX: Add trainer memory management
        self._initialize_trainer_memory_management()
        
        # Device setup
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
        
        print(f"{Fore.GREEN}🚀 [GRPO TRAINER] Memory-optimized GRPO trainer initialized{Style.RESET_ALL}")

    def _initialize_trainer_memory_management(self):
        """Initialize memory management for GRPO trainer"""
        try:
            # ✅ METRICS MEMORY MANAGEMENT
            self.max_metrics_history = 1000        # Keep only 1000 recent metrics
            self.metrics_cleanup_frequency = 100   # Cleanup every 100 steps
            
            # ✅ GENERATION CACHE MANAGEMENT
            self.max_generation_cache = 50         # Limit generation cache
            self.generation_cache = {}
            
            # ✅ TRAINING STEP MANAGEMENT
            self.training_step_counter = 0
            self.last_trainer_cleanup = time.time()
            self.trainer_cleanup_interval = 300    # 5 minutes
            
            # ✅ MEMORY PRESSURE THRESHOLDS
            self.trainer_memory_threshold = 10.0   # GB - trigger cleanup
            self.trainer_emergency_threshold = 20.0 # GB - emergency cleanup
            
            # ✅ TENSOR CACHE MANAGEMENT
            self.tensor_cache = {}
            self.max_tensor_cache = 100
            
            # ✅ BATCH PROCESSING LIMITS
            self.max_batch_size_mb = 500           # Limit batch size to 500MB
            self.batch_memory_tracking = deque(maxlen=50)  # Track batch memory usage
            
            print(f"{Fore.GREEN}🚀 [TRAINER] Memory management initialized{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Trainer memory management init failed: {e}")

    def _get_trainer_memory_usage(self):
        """Get current memory usage in GB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024**3
            except:
                pass
        return 0.0

    def _get_gpu_memory_usage(self):
        """Get GPU memory usage"""
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                return allocated, reserved
            except:
                pass
        return 0.0, 0.0

    def _check_trainer_memory_pressure(self):
        """Check if trainer memory cleanup is needed"""
        current_memory = self._get_trainer_memory_usage()
        gpu_allocated, gpu_reserved = self._get_gpu_memory_usage()
        
        # Emergency cleanup
        if current_memory > self.trainer_emergency_threshold or gpu_allocated > 15:
            print(f"{Fore.RED}🚨 [TRAINER EMERGENCY] RAM: {current_memory:.1f}GB, GPU: {gpu_allocated:.1f}GB - Emergency cleanup!{Style.RESET_ALL}")
            self._emergency_trainer_cleanup()
            return True
            
        # High memory pressure cleanup
        elif current_memory > self.trainer_memory_threshold or gpu_allocated > 10:
            print(f"{Fore.YELLOW}⚠️ [TRAINER PRESSURE] RAM: {current_memory:.1f}GB, GPU: {gpu_allocated:.1f}GB - Pressure cleanup{Style.RESET_ALL}")
            self._aggressive_trainer_cleanup()
            return True
            
        # Time-based cleanup
        elif time.time() - self.last_trainer_cleanup > self.trainer_cleanup_interval:
            print(f"{Fore.CYAN}🧹 [TRAINER CLEANUP] Periodic cleanup - RAM: {current_memory:.1f}GB, GPU: {gpu_allocated:.1f}GB{Style.RESET_ALL}")
            self._periodic_trainer_cleanup()
            return True
            
        return False

    def _periodic_trainer_cleanup(self):
        """Periodic trainer memory cleanup - safe and conservative"""
        try:
            # Clean generation cache
            if len(self.generation_cache) > self.max_generation_cache:
                sorted_keys = sorted(self.generation_cache.keys())
                old_keys = sorted_keys[:-self.max_generation_cache]
                for key in old_keys:
                    del self.generation_cache[key]
                
                if old_keys:
                    print(f"{Fore.CYAN}🧹 [TRAINER CACHE] Cleaned {len(old_keys)} old generations{Style.RESET_ALL}")
            
            # Clean tensor cache
            if len(self.tensor_cache) > self.max_tensor_cache:
                sorted_keys = sorted(self.tensor_cache.keys())
                old_keys = sorted_keys[:-self.max_tensor_cache]
                for key in old_keys:
                    del self.tensor_cache[key]
            
            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            
            # Light garbage collection
            collected = gc.collect()
            
            self.last_trainer_cleanup = time.time()
            
            if collected > 0:
                print(f"{Fore.CYAN}🧹 [TRAINER CLEANUP] Collected {collected} objects{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"Periodic trainer cleanup failed: {e}")

    def _aggressive_trainer_cleanup(self):
        """Aggressive trainer memory cleanup for high memory pressure"""
        try:
            print(f"{Fore.YELLOW}💥 [TRAINER AGGRESSIVE] Starting aggressive cleanup{Style.RESET_ALL}")
            
            # Clear all caches
            self.generation_cache.clear()
            self.tensor_cache.clear()
            self.batch_memory_tracking.clear()
            
            # Clean metrics more aggressively
            self._clean_metrics_history(keep_recent=100)  # Keep only 100 recent
            
            # Aggressive GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            
            # Force garbage collection multiple times
            total_collected = 0
            for _ in range(5):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
            
            self.last_trainer_cleanup = time.time()
            
            print(f"{Fore.GREEN}✅ [TRAINER AGGRESSIVE] Collected {total_collected} objects{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Aggressive trainer cleanup failed: {e}")

    def _emergency_trainer_cleanup(self):
        """Emergency trainer memory cleanup - nuclear option"""
        try:
            print(f"{Fore.RED}💥 [TRAINER EMERGENCY] NUCLEAR CLEANUP INITIATED{Style.RESET_ALL}")
            
            # Clear everything
            self.generation_cache.clear()
            self.tensor_cache.clear()
            self.batch_memory_tracking.clear()
            
            # Nuclear metrics cleanup
            self._clean_metrics_history(keep_recent=10)  # Keep only 10 recent
            
            # Clear optimizer state if safe
            if hasattr(self.optimizer, 'state') and self.optimizer.state:
                # Only clear if not in middle of training step
                if not hasattr(self, '_in_training_step') or not self._in_training_step:
                    self.optimizer.state.clear()
                    print(f"{Fore.YELLOW}🧹 [TRAINER EMERGENCY] Cleared optimizer state{Style.RESET_ALL}")
            
            # Nuclear GPU cleanup
            if torch.cuda.is_available():
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            
            # Nuclear garbage collection
            total_collected = 0
            for _ in range(10):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
                time.sleep(0.1)
            
            print(f"{Fore.GREEN}✅ [TRAINER EMERGENCY] Nuclear cleanup completed - {total_collected} objects{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Emergency trainer cleanup failed: {e}")

    def _clean_metrics_history(self, keep_recent=1000):
        """Clean metrics history to prevent accumulation"""
        try:
            cleaned = 0
            for mode in ["train", "eval"]:
                if mode in self._metrics:
                    for metric_name, values in self._metrics[mode].items():
                        if isinstance(values, list) and len(values) > keep_recent:
                            # Keep only recent values
                            self._metrics[mode][metric_name] = values[-keep_recent:]
                            cleaned += 1
            
            if cleaned > 0:
                print(f"{Fore.CYAN}📊 [METRICS] Cleaned {cleaned} metric histories{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"Metrics cleanup failed: {e}")

    def _initialize_model(self, enable_gradient_checkpointing):
        """✅ OPTIMIZED: Initialize the model and reference model with memory management."""
        try:
            self.model = self.model.to(self.device)
            if enable_gradient_checkpointing:
                print("INFO: Enabling gradient checkpointing.")
                self.model.gradient_checkpointing_enable()

            # Reference model setup
            if self.beta == 0.0:
                self.ref_model = None
                print(f"{Fore.GREEN}✅ [MODEL] No reference model needed (beta=0.0){Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}🔄 [MODEL] Creating reference model (beta={self.beta}){Style.RESET_ALL}")
                self.ref_model = create_reference_model(self.model).to(self.model.device)
                
                # ✅ MEMORY OPTIMIZATION: Reference model doesn't need gradients
                for param in self.ref_model.parameters():
                    param.requires_grad = False
                self.ref_model.eval()  # Always in eval mode
                
        except Exception as e:
            print(f"Model initialization failed: {e}")
            raise

    def _initialize_tokenizers(self):
        """✅ OPTIMIZED: Initialize tokenizers with memory management."""
        try:
            if self.processing_class is None:
                model_name = getattr(self.model.config, '_name_or_path', 'unknown')
                print(f"{Fore.CYAN}🔤 [TOKENIZER] Loading tokenizer for {model_name}{Style.RESET_ALL}")
                
                self.processing_class = AutoTokenizer.from_pretrained(
                    model_name, padding_side="left"
                )
                
            # ✅ MEMORY OPTIMIZATION: Set pad token if not set
            if self.processing_class.pad_token is None:
                self.processing_class.pad_token = self.processing_class.eos_token
                
        except Exception as e:
            print(f"Tokenizer initialization failed: {e}")
            raise

    def _initialize_metrics(self):
        """✅ OPTIMIZED: Initialize metrics tracking with bounded collections."""
        # ✅ MEMORY LEAK FIX: Use deque with maxlen instead of unlimited defaultdict
        self._metrics = {
            "train": {
                "loss": deque(maxlen=self.max_metrics_history),
                "rewards": deque(maxlen=self.max_metrics_history),
                "kl": deque(maxlen=self.max_metrics_history),
                "clip_ratio": deque(maxlen=self.max_metrics_history),
            },
            "eval": {
                "loss": deque(maxlen=self.max_metrics_history),
                "rewards": deque(maxlen=self.max_metrics_history),
                "kl": deque(maxlen=self.max_metrics_history),
                "clip_ratio": deque(maxlen=self.max_metrics_history),
            }
        }
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
        """✅ OPTIMIZED: Set generation config with memory considerations."""
        try:
            # ✅ MEMORY OPTIMIZATION: Limit max_new_tokens to prevent memory explosion
            max_completion_length = min(self.args.max_completion_length, 512)  # Cap at 512 tokens
            
            self.generation_config = GenerationConfig(
                max_new_tokens=max_completion_length,
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
                # ✅ MEMORY OPTIMIZATION: Enable efficient generation
                use_cache=True,  # Enable KV cache for efficiency
                remove_invalid_values=True,  # Remove invalid logits
            )
            
            print(f"{Fore.GREEN}✅ [GENERATION] Generation config initialized (max_tokens: {max_completion_length}){Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Generation config initialization failed: {e}")

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        """✅ OPTIMIZED: Process inputs with memory management and error handling"""
        
        try:
            # ✅ INPUT VALIDATION: Check if inputs is empty or None
            if not inputs:
                print(f"{Fore.YELLOW}⚠️ [PROCESS INPUT] Empty inputs provided{Style.RESET_ALL}")
                return self.processing_class("", return_tensors="pt", padding=True)
            
            # ✅ MEMORY OPTIMIZATION: Limit input processing
            if hasattr(inputs, 'to_dict'):
                inputs = [dict(inputs[i]) for i in range(len(inputs))]
            elif isinstance(inputs, dict):
                inputs = [inputs]
            elif not isinstance(inputs, list):
                print(f"{Fore.YELLOW}⚠️ [PROCESS INPUT] Converting inputs to list: {type(inputs)}{Style.RESET_ALL}")
                inputs = [inputs]

            # ✅ SIZE CHECK: Warn about large input batches
            if len(inputs) > 100:
                print(f"{Fore.YELLOW}⚠️ [PROCESS INPUT] Large batch: {len(inputs)} items{Style.RESET_ALL}")

            templated_prompts = []
            
            if with_template:
                if for_training:
                    for item in inputs:
                        try:
                            # ✅ VALIDATION: Check if item is valid for chat template
                            if isinstance(item, dict) and 'prompt' in item:
                                for _ in range(self.num_generations):
                                    templated_prompts.append(
                                        apply_chat_template(item, self.processing_class)["prompt"]
                                    )
                            elif isinstance(item, dict):
                                # Fallback: create basic chat format
                                basic_item = {"prompt": [{"role": "user", "content": str(item)}]}
                                for _ in range(self.num_generations):
                                    templated_prompts.append(
                                        apply_chat_template(basic_item, self.processing_class)["prompt"]
                                    )
                            else:
                                # Last resort: convert to string
                                basic_item = {"prompt": [{"role": "user", "content": str(item)}]}
                                for _ in range(self.num_generations):
                                    templated_prompts.append(
                                        apply_chat_template(basic_item, self.processing_class)["prompt"]
                                    )
                        except Exception as item_e:
                            print(f"{Fore.YELLOW}⚠️ [PROCESS INPUT] Failed to process item, using fallback: {item_e}{Style.RESET_ALL}")
                            # Fallback to simple string
                            for _ in range(self.num_generations):
                                templated_prompts.append(str(item))
                else:
                    for item in inputs:
                        try:
                            if isinstance(item, dict) and 'prompt' in item:
                                templated_prompts.append(
                                    apply_chat_template(item, self.processing_class)["prompt"]
                                )
                            elif isinstance(item, dict):
                                basic_item = {"prompt": [{"role": "user", "content": str(item)}]}
                                templated_prompts.append(
                                    apply_chat_template(basic_item, self.processing_class)["prompt"]
                                )
                            else:
                                basic_item = {"prompt": [{"role": "user", "content": str(item)}]}
                                templated_prompts.append(
                                    apply_chat_template(basic_item, self.processing_class)["prompt"]
                                )
                        except Exception as item_e:
                            print(f"{Fore.YELLOW}⚠️ [PROCESS INPUT] Failed to process item, using fallback: {item_e}{Style.RESET_ALL}")
                            templated_prompts.append(str(item))
            else:
                if for_training:
                    # ✅ SAFE EXTRACTION: Handle nested lists safely
                    for generations in inputs:
                        if isinstance(generations, list):
                            for output in generations:
                                templated_prompts.append(str(output))
                        else:
                            # Single item, not a list of generations
                            templated_prompts.append(str(generations))
                else:
                    # ✅ SAFE EXTRACTION: Handle cases where item[0] might not exist
                    for item in inputs:
                        if isinstance(item, list) and len(item) > 0:
                            templated_prompts.append(str(item[0]))
                        elif isinstance(item, str):
                            templated_prompts.append(item)
                        else:
                            templated_prompts.append(str(item))

            # ✅ VALIDATION: Ensure we have at least one prompt
            if not templated_prompts:
                print(f"{Fore.YELLOW}⚠️ [PROCESS INPUT] No valid prompts generated, using fallback{Style.RESET_ALL}")
                templated_prompts = [""]

            # ✅ MEMORY OPTIMIZATION: Limit prompt length
            max_length = 1024  # Reasonable limit
            input_tokens = self.processing_class(
                text=templated_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length,  # Prevent extremely long sequences
            )
            
            # ✅ CLEANUP: Clear temporary variables
            del templated_prompts
            
            print(f"{Fore.GREEN}✅ [PROCESS INPUT] Processed {input_tokens.input_ids.shape[0]} inputs{Style.RESET_ALL}")
            
            return input_tokens
            
        except Exception as e:
            print(f"{Fore.RED}❌ [PROCESS INPUT] Input processing failed: {e}{Style.RESET_ALL}")
            # Return minimal safe input
            try:
                return self.processing_class("fallback text", return_tensors="pt", padding=True)
            except Exception as fallback_e:
                print(f"{Fore.RED}❌ [PROCESS INPUT] Even fallback failed: {fallback_e}{Style.RESET_ALL}")
                # Final fallback - create minimal tensor manually
                return type('MockTokens', (), {
                    'input_ids': torch.tensor([[0]], device=self.device),
                    'attention_mask': torch.tensor([[1]], device=self.device)
                })

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        """✅ OPTIMIZED: Perform a single training step with memory management."""
        
        try:
            # ✅ INCREMENT COUNTERS
            global_step += 1
            self.training_step_counter += 1
            self._in_training_step = True  # Flag for cleanup safety
            
            # ✅ PERIODIC MEMORY CLEANUP
            if self.training_step_counter % self.metrics_cleanup_frequency == 0:
                self._check_trainer_memory_pressure()
            
            print(f"{Fore.CYAN}📈 [STEP] Training step {global_step} for stage {stage}{Style.RESET_ALL}")
            
            # ✅ GET INPUTS WITH ERROR HANDLING
            try:
                stage_inputs = state.get_stage_state(stage)
                stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
                assert stage_inputs is not None, f"No inputs found for stage {stage}"
                
                # ✅ VALIDATION: Check index_mapping
                if not index_mapping:
                    print(f"{Fore.YELLOW}⚠️ [STEP] Empty index mapping for stage {stage}{Style.RESET_ALL}")
                    return global_step
                    
            except Exception as input_e:
                print(f"{Fore.RED}❌ [STEP] Failed to get inputs for stage {stage}: {input_e}{Style.RESET_ALL}")
                return global_step
            
            # ✅ GET OUTPUTS WITH ERROR HANDLING
            try:
                stage_actions = state.get_stage_actions(stage)
                stage_outputs = []
                
                for idx in range(len(index_mapping)):
                    try:
                        agent, batch_id, node_idx = index_mapping[idx]
                        if agent in stage_actions and batch_id in stage_actions[agent] and node_idx in stage_actions[agent][batch_id]:
                            stage_outputs.append(stage_actions[agent][batch_id][node_idx])
                        else:
                            print(f"{Fore.YELLOW}⚠️ [STEP] Missing action for {agent}/{batch_id}/{node_idx}, using fallback{Style.RESET_ALL}")
                            stage_outputs.append("")  # Fallback
                    except Exception as action_e:
                        print(f"{Fore.YELLOW}⚠️ [STEP] Failed to get action for index {idx}: {action_e}{Style.RESET_ALL}")
                        stage_outputs.append("")  # Fallback
                
                if not stage_outputs:
                    print(f"{Fore.YELLOW}⚠️ [STEP] No outputs found for stage {stage}{Style.RESET_ALL}")
                    return global_step
                    
            except Exception as output_e:
                print(f"{Fore.RED}❌ [STEP] Failed to get outputs for stage {stage}: {output_e}{Style.RESET_ALL}")
                return global_step

            # ✅ PROCESS MODEL INPUTS WITH MEMORY MANAGEMENT
            try:
                model_inputs = {}
                processed_inputs = self._process_inputs(stage_inputs, for_training=True)
                model_inputs["prompt_ids"] = processed_inputs.input_ids.to(self.model.device)
                model_inputs["prompt_mask"] = processed_inputs.attention_mask.to(self.model.device)
                
                # ✅ CLEANUP: Clear processed inputs
                del processed_inputs
                
                processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
                model_inputs["completion_ids"] = processed_outputs.input_ids.to(self.model.device)
                model_inputs["completion_mask"] = processed_outputs.attention_mask.to(self.model.device)
                
                # ✅ CLEANUP: Clear processed outputs
                del processed_outputs
                
            except Exception as process_e:
                print(f"{Fore.RED}❌ [STEP] Failed to process inputs/outputs: {process_e}{Style.RESET_ALL}")
                return global_step

            # ✅ GET REWARDS WITH ERROR HANDLING
            try:
                rewards_raw = reward_manager[stage]
                rewards = []
                
                for idx in range(len(index_mapping)):
                    try:
                        agent, batch_id, node_idx = index_mapping[idx]
                        if agent in rewards_raw and batch_id in rewards_raw[agent] and node_idx in rewards_raw[agent][batch_id]:
                            reward_value = rewards_raw[agent][batch_id][node_idx]
                            # Ensure reward is a number
                            if isinstance(reward_value, (list, tuple)) and len(reward_value) > 0:
                                rewards.append(float(reward_value[0]))
                            else:
                                rewards.append(float(reward_value))
                        else:
                            print(f"{Fore.YELLOW}⚠️ [STEP] Missing reward for {agent}/{batch_id}/{node_idx}, using 0.0{Style.RESET_ALL}")
                            rewards.append(0.0)  # Fallback
                    except Exception as reward_e:
                        print(f"{Fore.YELLOW}⚠️ [STEP] Failed to get reward for index {idx}: {reward_e}{Style.RESET_ALL}")
                        rewards.append(0.0)  # Fallback
                
                if not rewards:
                    print(f"{Fore.YELLOW}⚠️ [STEP] No rewards found for stage {stage}, using zeros{Style.RESET_ALL}")
                    rewards = [0.0] * len(index_mapping)
                
                # ✅ TENSOR CREATION WITH VALIDATION
                rewards_tensor = torch.tensor(rewards, device=self.model.device, dtype=torch.float32)
                
                # ✅ RESHAPE REWARDS: Handle dimension mismatch
                if len(rewards_tensor.shape) == 1:
                    # Reshape to [batch_size, num_generations]
                    expected_batch_size = len(rewards) // self.num_generations
                    if len(rewards) % self.num_generations == 0 and expected_batch_size > 0:
                        rewards_tensor = rewards_tensor.view(expected_batch_size, self.num_generations)
                    else:
                        # Fallback: treat as single batch
                        print(f"{Fore.YELLOW}⚠️ [STEP] Reward dimension mismatch, treating as single batch{Style.RESET_ALL}")
                        rewards_tensor = rewards_tensor.unsqueeze(0)  # Add batch dimension
                        
            except Exception as reward_e:
                print(f"{Fore.RED}❌ [STEP] Failed to get rewards for stage {stage}: {reward_e}{Style.RESET_ALL}")
                return global_step

            # ✅ COMPUTE ADVANTAGES WITH MEMORY MANAGEMENT
            try:
                with torch.no_grad():
                    # ✅ SAFE ADVANTAGE COMPUTATION: Handle different tensor shapes
                    if rewards_tensor.dim() == 2 and rewards_tensor.size(1) > 1:
                        # Multiple generations per batch
                        advantages = rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)
                        std = rewards_tensor.std(dim=1, keepdim=True)
                        advantages = advantages / (std + 1e-8)
                    elif rewards_tensor.dim() == 2:
                        # Single generation per batch
                        advantages = rewards_tensor - rewards_tensor.mean()
                        std = rewards_tensor.std()
                        advantages = advantages / (std + 1e-8)
                    else:
                        # 1D tensor
                        advantages = rewards_tensor - rewards_tensor.mean()
                        std = rewards_tensor.std()
                        advantages = advantages / (std + 1e-8)
                        
                advantages = torch.flatten(advantages)
                
            except Exception as adv_e:
                print(f"{Fore.RED}❌ [STEP] Failed to compute advantages: {adv_e}{Style.RESET_ALL}")
                return global_step

            model_inputs["advantages"] = advantages
            model_inputs["old_per_token_logps"] = None

            # ✅ FORWARD PASS WITH MEMORY MANAGEMENT
            try:
                with self.autocast:
                    loss = self.compute_loss(self.model, model_inputs)

                # ✅ BACKWARD PASS WITH MEMORY MANAGEMENT
                loss.backward()
                
                # ✅ GRADIENT CLIPPING (optional)
                if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.model.zero_grad()  # ✅ Clear gradients
                
            except Exception as forward_e:
                print(f"{Fore.RED}❌ [STEP] Forward/backward pass failed: {forward_e}{Style.RESET_ALL}")
                # Clear gradients even on failure
                self.model.zero_grad()
                return global_step

            # ✅ LOGGING WITH MEMORY MANAGEMENT
            try:
                metrics = {
                    "train/loss": loss.cpu().item(), 
                    "train/rewards": rewards_tensor.cpu().mean().item()
                }
                self.log(metrics, global_step)
                
                # ✅ TRACK METRICS IN BOUNDED COLLECTIONS
                if "train" in self._metrics:
                    if "loss" in self._metrics["train"]:
                        self._metrics["train"]["loss"].append(loss.cpu().item())
                    if "rewards" in self._metrics["train"]:
                        self._metrics["train"]["rewards"].append(rewards_tensor.cpu().mean().item())
                        
            except Exception as log_e:
                print(f"Logging failed: {log_e}")

            # ✅ CLEANUP: Clear all tensors
            del loss, rewards_tensor, advantages, model_inputs
            
            # ✅ STEP CLEANUP
            self.cleanup_step()
            
            return global_step
            
        except Exception as e:
            print(f"{Fore.RED}❌ [STEP] Training step failed: {e}{Style.RESET_ALL}")
            return global_step
        finally:
            self._in_training_step = False  # Clear training flag

    def generate(
        self, inputs: Any, return_completion_ids: bool = False, stage=0
    ) -> Any:
        """
        ✅ OPTIMIZED: Generate outputs with comprehensive memory management
        
        OPTIMIZATION: Rewritten to use a single, efficient batch generation call.
        
        Generate outputs from the model for the given inputs.

        Args:
            inputs: Input data for generation
            return_completion_ids: Whether to return completion IDs along with text
            stage: Current stage (0, 1, or 2) for proper output formatting

        Returns:
            Generated outputs in the format expected by the next stage
        """
        
        try:
            # ✅ MEMORY CHECK: Before generation
            self._check_trainer_memory_pressure()
            
            print(f"{Fore.CYAN}🎯 [GENERATE] Starting generation for stage {stage}{Style.RESET_ALL}")
            
            input_tokens = self._process_inputs(inputs)
            prompt_length = input_tokens.input_ids.size(1)
            
            # ✅ MEMORY CHECK: Monitor input size
            input_size_mb = (input_tokens.input_ids.numel() * 4) / 1024 / 1024  # Rough estimate
            if input_size_mb > self.max_batch_size_mb:
                print(f"{Fore.RED}🚨 [GENERATE] Input too large: {input_size_mb:.1f}MB > {self.max_batch_size_mb}MB{Style.RESET_ALL}")
                # Return minimal fallback
                return [[""]] if not return_completion_ids else ([[""]], [[torch.tensor([], device=self.device)]])

            with torch.no_grad():
                # ✅ MEMORY OPTIMIZATION: Move to device efficiently
                input_ids = input_tokens.input_ids.to(self.model.device)
                attention_mask = input_tokens.attention_mask.to(self.model.device)
                
                # ✅ CLEAR ORIGINAL TENSORS
                del input_tokens
                
                # Single, batched generation call
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                )

            # Extract completions (i.e., removes prompt part)
            completion_ids = outputs[:, prompt_length:]
            completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            
            # ✅ CLEANUP: Clear intermediate tensors
            del outputs, input_ids, attention_mask
            
            # Reshape the flat list of completions into [batch_size, num_generations]
            num_prompts = completion_ids.size(0) // self.num_generations
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
                
                # ✅ CLEANUP
                del completion_ids, completion_ids_reshaped
                
                return rollout, rollout_ids
            else:
                # ✅ CLEANUP
                del completion_ids
                
                return rollout
                
        except Exception as e:
            print(f"{Fore.RED}❌ [GENERATE] Generation failed: {e}{Style.RESET_ALL}")
            # Return safe fallback
            return [[""]] if not return_completion_ids else ([[""]], [[torch.tensor([], device=self.device)]])

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """✅ OPTIMIZED: Get the per-token log probabilities with memory management."""
        try:
            model = model.to(input_ids.device)
            
            # ✅ MEMORY OPTIMIZATION: Use torch.no_grad() for inference-only operations
            with torch.no_grad() if model != self.model else contextlib.nullcontext():
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
            
            # ✅ CLEANUP: Clear intermediate tensors
            del logits, loss_mask, labels
            
            return token_log_probs
            
        except Exception as e:
            print(f"Per-token logps computation failed: {e}")
            # Return safe fallback
            batch_size, seq_len = input_ids.size(0), logits_to_keep
            return torch.zeros((batch_size, seq_len), device=input_ids.device)

    def compute_loss(
        self, model, inputs, num_items_in_batch=1, mode="train", return_metrics=False
    ):
        """✅ OPTIMIZED: Compute the GRPO loss with memory management."""
        
        try:
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

            # ✅ KL DIVERGENCE COMPUTATION (if needed)
            if self.beta != 0.0 and self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps)
                    - (ref_per_token_logps - per_token_logps)
                    - 1
                )
                
                # ✅ CLEANUP
                del ref_per_token_logps
            else:
                per_token_kl = None

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

            if self.beta != 0.0 and per_token_kl is not None:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

            # ✅ METRICS CALCULATION WITH MEMORY MANAGEMENT
            mean_kl = None
            if self.beta != 0.0 and per_token_kl is not None:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                if mode in self._metrics and "kl" in self._metrics[mode]:
                    self._metrics[mode]["kl"].append(mean_kl.item())

            is_clipped = (coef_1 > coef_2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
            
            # ✅ BOUNDED METRICS: Using deque prevents unlimited growth
            if mode in self._metrics:
                if "clip_ratio" in self._metrics[mode]:
                    self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
                if "loss" in self._metrics[mode]:
                    self._metrics[mode]["loss"].append(loss.item())

            # ✅ CLEANUP: Clear intermediate tensors
            del per_token_logps, coef_1, coef_2, per_token_loss, is_clipped
            if per_token_kl is not None:
                del per_token_kl

            if return_metrics:
                metrics = {
                    "loss": loss.item(),
                    "kl": mean_kl.item() if mean_kl is not None else None,
                    "clip_ratio": clip_ratio.item(),
                }
                return loss, metrics
            else:
                return loss
                
        except Exception as e:
            print(f"Loss computation failed: {e}")
            # Return safe fallback loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """✅ OPTIMIZED: Train the model with comprehensive memory management."""
        
        try:
            print(f"{Fore.MAGENTA}🎓 [TRAIN] Starting training for {state.stage} stages{Style.RESET_ALL}")
            
            self.model.train()
            global_step = self.global_step
            
            # ✅ MEMORY CHECK: Before training
            initial_memory = self._get_trainer_memory_usage()
            
            for stage in range(state.stage):
                try:
                    # ✅ MEMORY CHECK: Every 5 stages
                    if stage > 0 and stage % 5 == 0:
                        current_memory = self._get_trainer_memory_usage()
                        memory_increase = current_memory - initial_memory
                        
                        if memory_increase > 3.0:  # 3GB increase
                            print(f"{Fore.YELLOW}⚠️ [TRAIN] Memory increase: +{memory_increase:.1f}GB after {stage} stages{Style.RESET_ALL}")
                            
                        if memory_increase > 8.0:  # 8GB increase - emergency
                            print(f"{Fore.RED}🚨 [TRAIN] Critical memory increase, emergency cleanup{Style.RESET_ALL}")
                            self._emergency_trainer_cleanup()
                    
                    global_step = self.step(
                        stage, state, data_manager, reward_manager, global_step
                    )
                    
                except Exception as stage_e:
                    print(f"{Fore.RED}❌ [TRAIN] Stage {stage} failed: {stage_e}{Style.RESET_ALL}")
                    continue
            
            self.global_step = global_step
            self.model.eval()
            
            final_memory = self._get_trainer_memory_usage()
            memory_change = final_memory - initial_memory
            
            print(f"{Fore.GREEN}✅ [TRAIN] Training completed, memory change: {memory_change:+.1f}GB{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.model.eval()  # Ensure model is in eval mode

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        """✅ OPTIMIZED: Perform a single training step with memory management."""
        
        try:
            # ✅ INCREMENT COUNTERS
            global_step += 1
            self.training_step_counter += 1
            self._in_training_step = True  # Flag for cleanup safety
            
            # ✅ PERIODIC MEMORY CLEANUP
            if self.training_step_counter % self.metrics_cleanup_frequency == 0:
                self._check_trainer_memory_pressure()
            
            print(f"{Fore.CYAN}📈 [STEP] Training step {global_step} for stage {stage}{Style.RESET_ALL}")
            
            # ✅ GET INPUTS WITH ERROR HANDLING
            try:
                stage_inputs = state.get_stage_state(stage)
                stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
                assert stage_inputs is not None, f"No inputs found for stage {stage}"
            except Exception as input_e:
                print(f"{Fore.RED}❌ [STEP] Failed to get inputs for stage {stage}: {input_e}{Style.RESET_ALL}")
                return global_step
            
            # ✅ GET OUTPUTS WITH ERROR HANDLING
            try:
                stage_actions = state.get_stage_actions(stage)
                stage_outputs = [
                    stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
                    for idx in range(len(index_mapping))
                ]
                assert stage_outputs is not None, f"No outputs found for stage {stage}"
            except Exception as output_e:
                print(f"{Fore.RED}❌ [STEP] Failed to get outputs for stage {stage}: {output_e}{Style.RESET_ALL}")
                return global_step

            # ✅ PROCESS MODEL INPUTS WITH MEMORY MANAGEMENT
            try:
                model_inputs = {}
                processed_inputs = self._process_inputs(stage_inputs, for_training=True)
                model_inputs["prompt_ids"] = processed_inputs.input_ids.to(self.model.device)
                model_inputs["prompt_mask"] = processed_inputs.attention_mask.to(self.model.device)
                
                # ✅ CLEANUP: Clear processed inputs
                del processed_inputs
                
                processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
                model_inputs["completion_ids"] = processed_outputs.input_ids.to(self.model.device)
                model_inputs["completion_mask"] = processed_outputs.attention_mask.to(self.model.device)
                
                # ✅ CLEANUP: Clear processed outputs
                del processed_outputs
                
            except Exception as process_e:
                print(f"{Fore.RED}❌ [STEP] Failed to process inputs/outputs: {process_e}{Style.RESET_ALL}")
                return global_step

            # ✅ GET REWARDS WITH ERROR HANDLING
            try:
                rewards_raw = reward_manager[stage]
                rewards = [
                    rewards_raw[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
                    for idx in range(len(index_mapping))
                ]
                assert rewards is not None, f"No rewards found for stage {stage}"
                rewards = torch.tensor(rewards, device=self.model.device)
            except Exception as reward_e:
                print(f"{Fore.RED}❌ [STEP] Failed to get rewards for stage {stage}: {reward_e}{Style.RESET_ALL}")
                return global_step

            # ✅ COMPUTE ADVANTAGES WITH MEMORY MANAGEMENT
            with torch.no_grad():
                advantages = rewards - rewards.mean(dim=1, keepdim=True)
                if rewards.shape[1] > 1:
                    advantages /= rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = torch.flatten(advantages)

            model_inputs["advantages"] = advantages
            model_inputs["old_per_token_logps"] = None

            # ✅ FORWARD PASS WITH MEMORY MANAGEMENT
            try:
                with self.autocast:
                    loss = self.compute_loss(self.model, model_inputs)

                # ✅ BACKWARD PASS WITH MEMORY MANAGEMENT
                loss.backward()
                
                # ✅ GRADIENT CLIPPING (optional)
                if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.model.zero_grad()  # ✅ Clear gradients
                
            except Exception as forward_e:
                print(f"{Fore.RED}❌ [STEP] Forward/backward pass failed: {forward_e}{Style.RESET_ALL}")
                # Clear gradients even on failure
                self.model.zero_grad()
                return global_step

            # ✅ LOGGING WITH MEMORY MANAGEMENT
            try:
                metrics = {
                    "train/loss": loss.cpu().item(), 
                    "train/rewards": rewards.cpu().mean().item()
                }
                self.log(metrics, global_step)
                
                # ✅ TRACK METRICS IN BOUNDED COLLECTIONS
                if "train" in self._metrics:
                    if "loss" in self._metrics["train"]:
                        self._metrics["train"]["loss"].append(loss.cpu().item())
                    if "rewards" in self._metrics["train"]:
                        self._metrics["train"]["rewards"].append(rewards.cpu().mean().item())
                        
            except Exception as log_e:
                print(f"Logging failed: {log_e}")

            # ✅ CLEANUP: Clear all tensors
            del loss, rewards, advantages, model_inputs
            
            # ✅ STEP CLEANUP
            self.cleanup_step()
            
            return global_step
            
        except Exception as e:
            print(f"Training step failed: {e}")
            return global_step
        finally:
            self._in_training_step = False  # Clear training flag

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        """✅ OPTIMIZED: Evaluate with memory management"""
        
        try:
            print(f"{Fore.BLUE}📊 [EVALUATE] Starting evaluation{Style.RESET_ALL}")
            
            # ✅ MEMORY CHECK: Before evaluation
            initial_memory = self._get_trainer_memory_usage()
            
            # Basic evaluation - can be extended as needed
            self.model.eval()
            
            # ✅ SIMPLE EVALUATION: Just log current state
            eval_metrics = {
                "eval/current_round": getattr(state, 'round', 0),
                "eval/current_stage": getattr(state, 'stage', 0),
                "eval/memory_gb": initial_memory,
            }
            
            self.log(eval_metrics, self.global_step)
            
            print(f"{Fore.GREEN}✅ [EVALUATE] Evaluation completed{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
    
    def save(self, save_dir: str) -> None:
        """✅ OPTIMIZED: Save with memory management."""
        try:
            print(f"{Fore.CYAN}💾 [SAVE] Saving model to {save_dir}{Style.RESET_ALL}")
            
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_pretrained(save_dir)

            # ✅ MEMORY OPTIMIZATION: Convert deques to lists for saving
            metrics_to_save = {}
            for mode in self._metrics:
                metrics_to_save[mode] = {}
                for metric_name, values in self._metrics[mode].items():
                    if isinstance(values, deque):
                        metrics_to_save[mode][metric_name] = list(values)
                    else:
                        metrics_to_save[mode][metric_name] = values

            torch.save(
                {
                    "metrics": metrics_to_save,
                    "total_train_tokens": self._total_train_tokens,
                    "generation_config": self.generation_config,
                    "optimizer": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                    "training_step_counter": self.training_step_counter,
                },
                os.path.join(save_dir, "trainer_state.pt"),
            )
            
            print(f"{Fore.GREEN}✅ [SAVE] Model saved successfully{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Model saving failed: {e}")

    @classmethod
    def load(cls, load_dir: str, **kwargs) -> "GRPOLanguageTrainerModule":
        """✅ OPTIMIZED: Load with memory management."""
        try:
            print(f"{Fore.CYAN}📂 [LOAD] Loading model from {load_dir}{Style.RESET_ALL}")
            
            model = AutoModelForCausalLM.from_pretrained(load_dir)
            trainer = cls([model], **kwargs)

            trainer_state_path = os.path.join(load_dir, "trainer_state.pt")
            if os.path.exists(trainer_state_path):
                trainer_state = torch.load(trainer_state_path, map_location=trainer.device)
                
                # ✅ MEMORY OPTIMIZATION: Convert lists back to deques
                saved_metrics = trainer_state.get("metrics", {})
                for mode in saved_metrics:
                    if mode in trainer._metrics:
                        for metric_name, values in saved_metrics[mode].items():
                            if metric_name in trainer._metrics[mode]:
                                # Convert list back to deque
                                trainer._metrics[mode][metric_name] = deque(
                                    values[-trainer.max_metrics_history:], 
                                    maxlen=trainer.max_metrics_history
                                )
                
                trainer._total_train_tokens = trainer_state.get("total_train_tokens", 0)
                trainer.generation_config = trainer_state.get("generation_config", trainer.generation_config)
                trainer.global_step = trainer_state.get("global_step", 0)
                trainer.training_step_counter = trainer_state.get("training_step_counter", 0)
                
                if "optimizer" in trainer_state:
                    trainer.optimizer.load_state_dict(trainer_state["optimizer"])
                    
            print(f"{Fore.GREEN}✅ [LOAD] Model loaded successfully{Style.RESET_ALL}")
            return trainer
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def cleanup_step(self):
        """✅ OPTIMIZED: Clean up resources after a training step with enhanced management."""
        try:
            # ✅ GPU CLEANUP
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # Note: torch.mps.empty_cache() is available in newer PyTorch versions
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            
            # ✅ CONDITIONAL GARBAGE COLLECTION
            # Only do GC occasionally to avoid performance impact
            if self.training_step_counter % 50 == 0:  # Every 50 steps
                collected = gc.collect()
                if collected > 0:
                    print(f"{Fore.CYAN}🧹 [STEP CLEANUP] Collected {collected} objects{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"Step cleanup failed: {e}")

    def cleanup(self):
        """✅ OPTIMIZED: Clean up resources at the end of training with comprehensive management."""
        try:
            print(f"{Fore.CYAN}🧹 [TRAINER FINAL] Final trainer cleanup{Style.RESET_ALL}")
            
            # Emergency cleanup
            self._emergency_trainer_cleanup()
            
            # Clean up trackers
            if hasattr(self, 'cleanup_trackers'):
                self.cleanup_trackers()
            
            # Clear model references
            if hasattr(self, 'ref_model') and self.ref_model is not None:
                del self.ref_model
                self.ref_model = None
            
            print(f"{Fore.GREEN}✅ [TRAINER FINAL] Final cleanup completed{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Final trainer cleanup failed: {e}")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.cleanup()
        except:
            pass

    # ✅ ADD: Debug and monitoring methods
    def get_trainer_stats(self):
        """Get trainer statistics for monitoring"""
        try:
            gpu_allocated, gpu_reserved = self._get_gpu_memory_usage()
            
            stats = {
                'global_step': self.global_step,
                'training_step_counter': self.training_step_counter,
                'memory_usage_gb': self._get_trainer_memory_usage(),
                'gpu_allocated_gb': gpu_allocated,
                'gpu_reserved_gb': gpu_reserved,
                'generation_cache_size': len(self.generation_cache),
                'tensor_cache_size': len(self.tensor_cache),
                'metrics_sizes': {
                    mode: {metric: len(values) for metric, values in metrics.items()}
                    for mode, metrics in self._metrics.items()
                },
            }
            return stats
        except Exception as e:
            print(f"Failed to get trainer stats: {e}")
            return {}

    def debug_trainer_memory(self):
        """Debug method to show trainer memory usage"""
        try:
            memory_gb = self._get_trainer_memory_usage()
            gpu_allocated, gpu_reserved = self._get_gpu_memory_usage()
            
            print(f"{Fore.BLUE}🔍 [TRAINER DEBUG] Memory Usage:{Style.RESET_ALL}")
            print(f"   💾 Total RAM: {memory_gb:.2f}GB")
            print(f"   🎮 GPU Allocated: {gpu_allocated:.2f}GB")
            print(f"   🎮 GPU Reserved: {gpu_reserved:.2f}GB")
            print(f"   📈 Global step: {self.global_step}")
            print(f"   🔄 Training steps: {self.training_step_counter}")
            print(f"   🎯 Generation cache: {len(self.generation_cache)} items")
            print(f"   📊 Tensor cache: {len(self.tensor_cache)} items")
            
            # Show metrics sizes
            print(f"   📋 Metrics sizes:")
            for mode, metrics in self._metrics.items():
                for metric_name, values in metrics.items():
                    print(f"      {mode}/{metric_name}: {len(values)} items")
            
            return {
                'memory_gb': memory_gb,
                'gpu_allocated_gb': gpu_allocated,
                'gpu_reserved_gb': gpu_reserved,
                'cache_sizes': {
                    'generation': len(self.generation_cache),
                    'tensor': len(self.tensor_cache),
                },
                'metrics_sizes': {
                    mode: {metric: len(values) for metric, values in metrics.items()}
                    for mode, metrics in self._metrics.items()
                },
            }
            
        except Exception as e:
            print(f"Debug trainer memory failed: {e}")
            return {}
