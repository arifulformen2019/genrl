import abc
import hashlib
import gc  # ✅ ADD: Memory cleanup
import time  # ✅ ADD: Time tracking
import weakref  # ✅ ADD: Weak references
from collections import deque  # ✅ ADD: Bounded collections
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

from datasets import Dataset, load_dataset
from numpy import ndarray
from torch import Tensor

from genrl.communication import Payload
from genrl.data import DataManager
from genrl.misc_utils.utils import generate_md5_hash_id
from genrl.state import GameState, WorldState

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

# ✅ ADD: Disable HuggingFace caching globally to prevent cache bloat
try:
    from datasets import disable_caching
    disable_caching()
except ImportError:
    pass


class LocalMemoryTextDataManager(DataManager):
    def __init__(
        self,
        train_dataset: str | None,
        evaluation_dataset: str | None = None,
        num_train_samples: int | None = 5,
        num_evaluation_samples: int | None = None,
        column_name_map: Dict[str, str] = None,
        column_preprocessing_map: Dict[str, Callable] | None = None,
        seed: int | None = None,
        batch_item_id_column: str | None = None,
        data_generator: Callable | None = None,
        **kwargs,
    ):
        super().__init__()

        self.datasets = {"train": train_dataset, "evaluation": evaluation_dataset}
        self.num_samples = {
            "train": num_train_samples,
            "evaluation": num_evaluation_samples,
        }
        self.column_map = {
            "names": column_name_map,
            "preprocessing": column_preprocessing_map,
        }
        self.seed = seed
        self.batch_item_id_column = batch_item_id_column
        if (data_generator is None) and (not isinstance(train_dataset, str)):
            raise ValueError(
                "Provided train dataset is not a string, but no data generating function was provided. Please provide an appropriate path/dataset ID for your desired training data OR provide a function for generating data at the start of a round."
            )
        self.data_generator = data_generator or self.load_HF_dataset
        # Optional properties
        self.data_subset = kwargs.get("subsets", None)

        # ✅ CRITICAL MEMORY LEAK FIX: Add memory management
        self._initialize_memory_management()

    def _initialize_memory_management(self):
        """Initialize memory management for DataManager"""
        try:
            # ✅ DATASET CACHE MANAGEMENT
            self.max_cached_datasets = 2  # Keep only 2 recent datasets (train + eval)
            self.dataset_cache = deque(maxlen=self.max_cached_datasets)
            self.cached_raw_datasets = {}  # For HF dataset caching with limits
            
            # ✅ PROCESSED DATA MANAGEMENT
            self.max_processed_data_cache = 3  # Keep only 3 recent processed datasets
            self.processed_data_cache = deque(maxlen=self.max_processed_data_cache)
            
            # ✅ CLEANUP FREQUENCY CONTROL
            self.data_load_counter = 0
            self.cleanup_frequency = 5  # Cleanup every 5 data loads
            self.last_memory_cleanup = time.time()
            self.memory_cleanup_interval = 300  # 5 minutes
            
            # ✅ MEMORY PRESSURE THRESHOLDS
            self.memory_pressure_threshold = 15.0  # GB - trigger aggressive cleanup
            self.emergency_memory_threshold = 25.0  # GB - trigger emergency cleanup
            
            # ✅ HF CACHE MANAGEMENT
            self.max_hf_cache_size_gb = 5.0  # Limit HF cache to 5GB
            
            print(f"{Fore.GREEN}🚀 [DATA MANAGER] Memory optimization initialized{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ [DATA MANAGER] Memory init warning: {e}{Style.RESET_ALL}")

    def _get_memory_usage(self):
        """Get current memory usage in GB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024**3
            except:
                pass
        return 0.0

    def _check_memory_pressure(self):
        """Check if memory cleanup is needed"""
        current_memory = self._get_memory_usage()
        
        # Emergency cleanup
        if current_memory > self.emergency_memory_threshold:
            print(f"{Fore.RED}🚨 [DATA EMERGENCY] Memory: {current_memory:.1f}GB - Emergency cleanup!{Style.RESET_ALL}")
            self._emergency_memory_cleanup()
            return True
            
        # High memory pressure cleanup
        elif current_memory > self.memory_pressure_threshold:
            print(f"{Fore.YELLOW}⚠️ [DATA PRESSURE] Memory: {current_memory:.1f}GB - Pressure cleanup{Style.RESET_ALL}")
            self._aggressive_memory_cleanup()
            return True
            
        # Time-based cleanup
        elif time.time() - self.last_memory_cleanup > self.memory_cleanup_interval:
            print(f"{Fore.CYAN}🧹 [DATA CLEANUP] Periodic cleanup - Memory: {current_memory:.1f}GB{Style.RESET_ALL}")
            self._periodic_memory_cleanup()
            return True
            
        return False

    def _periodic_memory_cleanup(self):
        """Periodic memory cleanup - safe and conservative"""
        try:
            # Clean old processed data
            if len(self.processed_data_cache) >= self.max_processed_data_cache:
                # Cache is full, let deque handle it automatically
                pass
            
            # Clean HF cache if it exists
            self._clean_hf_cache()
            
            # Light garbage collection
            collected = gc.collect()
            
            self.last_memory_cleanup = time.time()
            
            if collected > 0:
                print(f"{Fore.CYAN}🧹 [DATA CLEANUP] Collected {collected} objects{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"Periodic cleanup failed: {e}")

    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup for high memory pressure"""
        try:
            print(f"{Fore.YELLOW}💥 [DATA AGGRESSIVE] Starting aggressive cleanup{Style.RESET_ALL}")
            
            # Clear all caches
            self.dataset_cache.clear()
            self.processed_data_cache.clear()
            self.cached_raw_datasets.clear()
            
            # Clear HF cache aggressively
            self._clean_hf_cache(aggressive=True)
            
            # Force garbage collection multiple times
            total_collected = 0
            for _ in range(5):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
            
            self.last_memory_cleanup = time.time()
            
            print(f"{Fore.GREEN}✅ [DATA AGGRESSIVE] Collected {total_collected} objects{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Aggressive cleanup failed: {e}")

    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup - nuclear option"""
        try:
            print(f"{Fore.RED}💥 [DATA EMERGENCY] NUCLEAR CLEANUP INITIATED{Style.RESET_ALL}")
            
            # Clear everything
            self.dataset_cache.clear()
            self.processed_data_cache.clear()
            self.cached_raw_datasets.clear()
            
            # Try to clear HF datasets cache directory
            try:
                import os
                import shutil
                cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
                if os.path.exists(cache_dir):
                    # Don't delete everything, just old files
                    import time
                    current_time = time.time()
                    for root, dirs, files in os.walk(cache_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                # Delete files older than 1 hour
                                if current_time - os.path.getmtime(file_path) > 3600:
                                    os.remove(file_path)
                            except:
                                pass
            except Exception as e:
                print(f"HF cache cleanup failed: {e}")
            
            # Nuclear garbage collection
            total_collected = 0
            for _ in range(10):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
                time.sleep(0.1)  # Give system time to clean up
            
            print(f"{Fore.GREEN}✅ [DATA EMERGENCY] Nuclear cleanup completed - {total_collected} objects{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Emergency cleanup failed: {e}")

    def _clean_hf_cache(self, aggressive=False):
        """Clean HuggingFace datasets cache"""
        try:
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            
            if os.path.exists(cache_dir):
                # Calculate cache size
                cache_size = 0
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        try:
                            cache_size += os.path.getsize(os.path.join(root, file))
                        except:
                            pass
                
                cache_size_gb = cache_size / 1024**3
                
                if cache_size_gb > self.max_hf_cache_size_gb or aggressive:
                    print(f"{Fore.YELLOW}📦 [HF CACHE] Size: {cache_size_gb:.1f}GB - Cleaning...{Style.RESET_ALL}")
                    
                    # Clean old cache files
                    files_removed = 0
                    current_time = time.time()
                    
                    for root, dirs, files in os.walk(cache_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                # Remove files older than threshold
                                age_hours = (current_time - os.path.getmtime(file_path)) / 3600
                                max_age = 1 if aggressive else 24  # 1 hour if aggressive, 24 hours otherwise
                                
                                if age_hours > max_age:
                                    os.remove(file_path)
                                    files_removed += 1
                            except:
                                pass
                    
                    if files_removed > 0:
                        print(f"{Fore.GREEN}✅ [HF CACHE] Removed {files_removed} old cache files{Style.RESET_ALL}")
                        
        except Exception as e:
            print(f"HF cache cleanup failed: {e}")

    # --- Helper Methods ---
    def load_HF_dataset(
        self,
        dataset_id_or_path: str,
        subset: str | None = None,
        split: str | None = "train",
        num_samples: int | None = None,
    ) -> Dataset:
        """✅ OPTIMIZED: Load dataset with memory management"""
        
        # ✅ INCREMENT COUNTER AND CHECK MEMORY
        self.data_load_counter += 1
        
        # ✅ CHECK FOR MEMORY PRESSURE BEFORE LOADING
        if self.data_load_counter % self.cleanup_frequency == 0:
            self._check_memory_pressure()
        
        # ✅ CREATE CACHE KEY
        cache_key = f"{dataset_id_or_path}_{subset}_{split}_{num_samples}_{self.seed}"
        
        # ✅ CHECK CACHE FIRST
        if cache_key in self.cached_raw_datasets:
            print(f"{Fore.BLUE}💾 [DATA CACHE] Using cached dataset: {dataset_id_or_path}{Style.RESET_ALL}")
            return self.cached_raw_datasets[cache_key]
        
        try:
            # ✅ LOAD DATASET WITH MEMORY MONITORING
            print(f"{Fore.CYAN}📦 [DATA LOAD] Loading: {dataset_id_or_path} (samples: {num_samples}){Style.RESET_ALL}")
            
            initial_memory = self._get_memory_usage()
            
            # Load dataset from HuggingFace
            if subset is not None:
                dataset_raw = load_dataset(dataset_id_or_path, subset, split=split)
            else:
                dataset_raw = load_dataset(dataset_id_or_path, split=split)
                
            if self.seed is not None:
                dataset_raw = dataset_raw.shuffle(seed=self.seed)
                
            if num_samples is not None:
                dataset_raw = dataset_raw.select(range(num_samples))
                
            # ✅ CACHE WITH SIZE LIMIT
            if len(self.cached_raw_datasets) >= self.max_cached_datasets:
                # Remove oldest cache entry
                oldest_key = next(iter(self.cached_raw_datasets))
                del self.cached_raw_datasets[oldest_key]
                print(f"{Fore.CYAN}🧹 [DATA CACHE] Removed oldest cache entry{Style.RESET_ALL}")
            
            self.cached_raw_datasets[cache_key] = dataset_raw
            
            final_memory = self._get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            print(f"{Fore.GREEN}✅ [DATA LOAD] Loaded successfully (+{memory_increase:.1f}GB RAM){Style.RESET_ALL}")
            
            return dataset_raw
            
        except Exception as e:
            print(f"{Fore.RED}❌ [DATA LOAD] Failed to load {dataset_id_or_path}: {e}{Style.RESET_ALL}")
            raise

    def filter_swarm_states(
        self, swarm_states: Dict[Any, Any], batch_id: Any
    ) -> List[str]:
        """
        ✅ OPTIMIZED: Filter swarm states with memory cleanup
        Consumes data received from the communication step and unpacking it into something that will be combined with prior world-state to form a world-state for the next stage
        """
        opponent_responses = []
        
        try:
            for agent_id in swarm_states:
                if batch_id in swarm_states[agent_id]:
                    for node_idx, _ in enumerate(swarm_states[agent_id][batch_id]):
                        agent_action = swarm_states[agent_id][batch_id][node_idx]
                        if isinstance(agent_action, Payload) and hasattr(
                            agent_action, "actions"
                        ):
                            agent_action = agent_action.actions
                        if isinstance(agent_action, str):
                            opponent_responses.append(agent_action)
                        elif isinstance(agent_action, list):
                            for response in agent_action:
                                if isinstance(response, str):
                                    opponent_responses.append(response)
                                    
                        # ✅ MEMORY OPTIMIZATION: Clear processed action reference
                        del agent_action
                        
            # ✅ LIMIT RESPONSE SIZE TO PREVENT MEMORY EXPLOSION
            if len(opponent_responses) > 1000:  # Arbitrary limit
                print(f"{Fore.YELLOW}⚠️ [SWARM FILTER] Too many responses ({len(opponent_responses)}), limiting to 1000{Style.RESET_ALL}")
                opponent_responses = opponent_responses[:1000]
                
        except Exception as e:
            print(f"Filter swarm states failed: {e}")
            
        return opponent_responses

    def flatten_tree_input(
        self, inputs: Dict[Any, Dict[Any, List[Tuple[Any]]]], stage: int
    ) -> Tuple[Dict[str, List[Any]], Dict[int, Tuple[int, int, int]]]:
        """✅ OPTIMIZED: Flatten tree input with memory management"""
        
        input_flattened, index_mapping = {}, {}
        cur_idx = 0
        
        try:
            for agent in inputs:
                for batch_id in inputs[agent]:
                    for node_idx, state in enumerate(inputs[agent][batch_id]):
                        input_flattened = self.flatten_states(input_flattened, state, stage)
                        index_mapping[cur_idx] = (agent, batch_id, node_idx)
                        cur_idx += 1
                        
                        # ✅ MEMORY CHECK: Prevent runaway flattening
                        if cur_idx > 10000:  # Arbitrary limit
                            print(f"{Fore.RED}🚨 [FLATTEN] Too many inputs ({cur_idx}), stopping to prevent memory explosion{Style.RESET_ALL}")
                            break
                            
        except Exception as e:
            print(f"Flatten tree input failed: {e}")
            
        return input_flattened, index_mapping

    # --- Optional Methods ---
    # NOTE: These require different approaches depending on the game being played/usecase, so should be hijacked when appropriate!
    def batch_item_id_generator(self, hashable_obj: Any) -> Any:
        """
        Generates unique hashes for a given batch item.
        """
        return generate_md5_hash_id(hashable_obj)

    def prompt_map(
        self, flattened_data: Any
    ) -> Any:  # TODO: Come up with a better term than "flattened" data
        """
        Maps flattened data into a prompt that will be consumed by the LLM
        """
        prompt = [
            {"role": "system", "content": flattened_data["system_prompt"]},
            {"role": "user", "content": flattened_data["user_prompt"]},
        ]
        return {"prompt": prompt}

    def merge_swarm_and_node_states(
        self,
        node_state: WorldState,
        swarm_states: Dict[Any, Any],
        stage: int,
        agent: Any,
        batch_id: Any,
    ) -> WorldState:
        """
        ✅ OPTIMIZED: Merge states with memory management
        Parses states from a node of in game tree as well as data coming from communication, and merges them into states that will be appended to nodes in the upcoming stage
        """
        try:
            environment_state = self.prepare_environment(
                node_state, swarm_states, stage, agent, batch_id
            )
            environment_state["prior_stage_input_states"] = deepcopy(node_state)
            opponent_state = self.prepare_opponent(
                node_state, swarm_states, stage, agent, batch_id
            )
            personal_state = self.prepare_personal(
                node_state, swarm_states, stage, agent, batch_id
            )
            world_state = WorldState(
                environment_states=environment_state,
                opponent_states=opponent_state,
                personal_states=personal_state,
            )
            return world_state
            
        except Exception as e:
            print(f"Merge states failed: {e}")
            # Return minimal safe state
            return WorldState(
                environment_states=node_state.environment_states if hasattr(node_state, 'environment_states') else {},
                opponent_states=None,
                personal_states=None,
            )

    def prepare_data(self, dataset_raw):
        """✅ OPTIMIZED: Prepare data with memory management"""
        
        try:
            print(f"{Fore.CYAN}🔧 [DATA PREP] Processing {len(dataset_raw)} samples{Style.RESET_ALL}")
            
            initial_memory = self._get_memory_usage()
            dataset_processed = []
            
            for idx, datum in enumerate(dataset_raw):
                try:
                    # Fill environment state with input data about the start of the round.
                    if self.column_map["names"] is not None:
                        env_state = {
                            key: datum[self.column_map["names"][key]]
                            for key in self.column_map["names"]
                        }
                    elif getattr(dataset_raw, "column_names", False):
                        env_state = {key: datum[key] for key in dataset_raw.column_names}
                    else:
                        raise AttributeError(
                            'No mapping for column names were provided and generated dataset object doesn\'t have a "column_names" method for inferring desired column names.'
                        )
                        
                    # Preprocess any columns if desired
                    if self.column_map["preprocessing"] is not None:
                        for col in self.column_map["preprocessing"]:
                            if col in env_state:
                                env_state[col] = self.column_map["preprocessing"][col](
                                    env_state[col]
                                )
                            else:
                                raise ValueError(
                                    f"Received a column preprocessing function for column == {col}, but this column doesn't exist in your environment states whose columns are: {env_state.keys()}"
                                )
                                
                    world_state = WorldState(
                        environment_states=env_state, opponent_states=None, personal_states=None
                    )
                    
                    if self.batch_item_id_column is not None:
                        item = (
                            self.batch_item_id_generator(env_state[self.batch_item_id_column]),
                            world_state,
                        )
                    else:
                        item = (idx, world_state,)
                        
                    dataset_processed.append(item)
                    
                    # ✅ PERIODIC CLEANUP DURING PROCESSING
                    if idx > 0 and idx % 1000 == 0:
                        gc.collect()
                        current_memory = self._get_memory_usage()
                        print(f"{Fore.BLUE}📊 [DATA PREP] Processed {idx} samples, RAM: {current_memory:.1f}GB{Style.RESET_ALL}")
                        
                        # Emergency stop if memory too high
                        if current_memory > self.emergency_memory_threshold:
                            print(f"{Fore.RED}🚨 [DATA PREP] Memory limit reached, stopping at {idx} samples{Style.RESET_ALL}")
                            break
                    
                    # Clear references
                    del datum, env_state, world_state
                    
                except Exception as e:
                    print(f"Failed to process sample {idx}: {e}")
                    continue
            
            final_memory = self._get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            print(f"{Fore.GREEN}✅ [DATA PREP] Processed {len(dataset_processed)} samples (+{memory_increase:.1f}GB RAM){Style.RESET_ALL}")
            
            # ✅ CACHE PROCESSED DATA WITH LIMITS
            if len(self.processed_data_cache) >= self.max_processed_data_cache:
                # Deque will automatically remove oldest
                pass
            self.processed_data_cache.append(('prepared', dataset_processed))
            
            return dataset_processed
            
        except Exception as e:
            print(f"Data preparation failed: {e}")
            return []

    # --- Main DataManager Methods ---
    def get_round_data(self, **kwargs) -> List[Tuple[Any, Any, Any, Any]]:
        """✅ OPTIMIZED: Get round data with memory management"""
        
        try:
            print(f"{Fore.MAGENTA}🎮 [ROUND DATA] Loading training data for round{Style.RESET_ALL}")
            
            # ✅ CHECK MEMORY BEFORE LOADING
            self._check_memory_pressure()
            
            dataset_raw = self.data_generator(
                dataset_id_or_path=self.datasets["train"],
                subset=self.data_subset,
                split=kwargs.get("split", "train"),
                num_samples=self.num_samples["train"],
            )
            
            prepared_data = self.prepare_data(dataset_raw)
            
            # ✅ CLEANUP AFTER LOADING
            del dataset_raw
            gc.collect()
            
            return prepared_data
            
        except Exception as e:
            print(f"Get round data failed: {e}")
            return []

    def get_eval_data(self, **kwargs) -> List[Tuple[Any, Any, Any, Any]]:
        """✅ OPTIMIZED: Get eval data with memory management"""
        
        try:
            print(f"{Fore.MAGENTA}📊 [EVAL DATA] Loading evaluation data{Style.RESET_ALL}")
            
            # ✅ CHECK MEMORY BEFORE LOADING
            self._check_memory_pressure()
            
            dataset_raw = self.data_generator(
                dataset_id_or_path=self.datasets["evaluation"],
                subset=self.data_subset,
                split=kwargs.get("split", "test"),
                num_samples=self.num_samples["evaluation"],
            )
            
            prepared_data = self.prepare_data(dataset_raw)
            
            # ✅ CLEANUP AFTER LOADING
            del dataset_raw
            gc.collect()
            
            return prepared_data
            
        except Exception as e:
            print(f"Get eval data failed: {e}")
            return []

    def prepare_input(
        self, inputs: Dict[Any, Dict[Any, List[Tuple[Any]]]], stage: int = None
    ) -> Tuple[Dataset, Dict[int, Tuple[int, int, int]]]:
        """✅ OPTIMIZED: Prepare input with memory management"""
        
        try:
            input_flattened, index_mapping = self.flatten_tree_input(inputs, stage)
            
            # ✅ CHECK INPUT SIZE
            if not input_flattened:
                print(f"{Fore.YELLOW}⚠️ [PREP INPUT] Empty input, returning empty dataset{Style.RESET_ALL}")
                return Dataset.from_dict({}), {}
            
            input_flattened = Dataset.from_dict(input_flattened)
            input_prepared = input_flattened.map(self.prompt_map)
            
            # ✅ CLEANUP INTERMEDIATE DATA
            del input_flattened
            gc.collect()
            
            return input_prepared, index_mapping
            
        except Exception as e:
            print(f"Prepare input failed: {e}")
            return Dataset.from_dict({}), {}

    def prepare_actions(
        self, outputs: Any, index_mapping: Dict[int, Tuple[Any]]
    ) -> Dict[Any, List[List[Any]]]:
        """✅ OPTIMIZED: Prepare actions with memory management"""
        
        try:
            if isinstance(outputs, Tensor | ndarray):
                outputs = outputs.tolist()
                
            actions = {}
            
            for idx, model_output in enumerate(outputs):
                if idx not in index_mapping:
                    continue
                    
                agent, batch_id, node_idx = index_mapping[idx]
                
                if agent not in actions:
                    actions[agent] = {}
                if batch_id not in actions[agent]:
                    actions[agent][batch_id] = {}
                    
                actions[agent][batch_id][node_idx] = model_output
                
                # ✅ MEMORY CHECK: Prevent runaway action preparation
                if len(actions) > 1000:  # Arbitrary limit
                    print(f"{Fore.YELLOW}⚠️ [PREP ACTIONS] Too many actions, limiting to prevent memory explosion{Style.RESET_ALL}")
                    break
            
            return actions
            
        except Exception as e:
            print(f"Prepare actions failed: {e}")
            return {}

    def prepare_states(
        self, current_state: GameState, swarm_states: Dict[Any, Any]
    ) -> Dict[Any, Dict[Any, List[Tuple[Any]]]]:
        """✅ OPTIMIZED: Prepare states with memory management"""
        
        try:
            latest_state = current_state.get_latest_state()
            
            for agent in latest_state:
                for batch_id in latest_state[agent]:
                    for node_idx, node_state in enumerate(latest_state[agent][batch_id]):
                        latest_state[agent][batch_id][node_idx] = (
                            self.merge_swarm_and_node_states(
                                node_state,
                                swarm_states,
                                current_state.stage,
                                agent,
                                batch_id,
                            )
                        )
                        
            return latest_state
            
        except Exception as e:
            print(f"Prepare states failed: {e}")
            return {}

    # ✅ ADD: Cleanup methods
    def cleanup(self):
        """Manual cleanup method"""
        try:
            print(f"{Fore.CYAN}🧹 [DATA MANAGER] Manual cleanup initiated{Style.RESET_ALL}")
            self._emergency_memory_cleanup()
        except Exception as e:
            print(f"Manual cleanup failed: {e}")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.cleanup()
        except:
            pass

    # --- Required Game-Dependant Methods ---
    @abc.abstractmethod
    def flatten_states(
        self, flattened_input: Dict[str, List[Any]], state: List[Any], stage: int
    ) -> Dict[str, List[Any]]:
        """Return a dictionary keyed on columns for batched input to the model, where each key points to a ordered list of values each row of that column will have"""
        pass

    @abc.abstractmethod
    def prepare_environment(
        self,
        node_states: List[Any],
        swarm_states: Dict[Any, Any],
        stage: int,
        agent: Any,
        batch_id: Any,
    ) -> Any:
        """
        Returns data that should be passed onto a node's children as an environment state when starting the next stage of the game.
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        pass

    @abc.abstractmethod
    def prepare_opponent(
        self,
        node_states: List[Any],
        swarm_states: Dict[Any, Any],
        stage: int,
        agent: Any,
        batch_id: Any,
    ) -> Any:
        """
        Returns data that should be passed onto a node's children as an opponent state when starting the next stage of the game.
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        pass

    @abc.abstractmethod
    def prepare_personal(
        self,
        node_states: List[Any],
        swarm_states: Dict[Any, Any],
        stage: int,
        agent: Any,
        batch_id: Any,
    ) -> Any:
        """
        Returns data that should be passed onto a node's children as an personal state when starting the next stage of the game.
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        pass


class SimpleTextDataManager(LocalMemoryTextDataManager):
    """
    ✅ OPTIMIZED: A simple data manager for text-based games with memory management
    This data manager assumes there is only a single stage.
    """

    def __init__(
        self,
        train_dataset: str | None,
        evaluation_dataset: str | None = None,
        num_train_samples: int | None = 5,
        num_evaluation_samples: int | None = None,
        column_name_map: Dict[str, str] = None,
        column_preprocessing_map: Dict[str, Callable] | None = None,
        seed: int | None = None,
        batch_item_id_column: str | None = None,
        data_generator: Callable | None = None,
        answer_extractor: Callable | None = None,
        system_prompt: str | None = "",
        **kwargs,
    ):

        super().__init__(
            train_dataset=train_dataset,
            evaluation_dataset=evaluation_dataset,
            num_train_samples=num_train_samples,
            num_evaluation_samples=num_evaluation_samples,
            column_name_map=column_name_map,
            column_preprocessing_map=column_preprocessing_map,
            seed=seed,
            batch_item_id_column=batch_item_id_column,
            data_generator=data_generator,
            subsets=kwargs.get("data_subset", None),
            **kwargs,
        )

        self.answer_extractor = answer_extractor
        self.system_prompt = system_prompt

        # ✅ ADD: Additional memory management for SimpleTextDataManager
        self.simple_data_cache = deque(maxlen=5)  # Keep only 5 recent simple data items
        
        print(f"{Fore.GREEN}🚀 [SIMPLE DATA] Memory-optimized SimpleTextDataManager initialized{Style.RESET_ALL}")

    def state_to_user_prompt(self, state: Tuple[Any, Any, Any]) -> str:
        """✅ SAFE: Extract user prompt from state"""
        try:
            return state.environment_states["question"]
        except (AttributeError, KeyError, TypeError) as e:
            print(f"{Fore.YELLOW}⚠️ [USER PROMPT] Failed to extract question: {e}{Style.RESET_ALL}")
            return ""

    def state_to_answer(self, state: Tuple[Any, Any, Any]) -> str:
        """✅ SAFE: Extract answer from state"""
        try:
            return state.environment_states["answer"]
        except (AttributeError, KeyError, TypeError) as e:
            print(f"{Fore.YELLOW}⚠️ [ANSWER] Failed to extract answer: {e}{Style.RESET_ALL}")
            return ""

    # --- Required Game-Dependant Methods ---
    def flatten_states(
        self, flattened_input: Dict[str, List[Any]], state: List[Any], stage: int
    ) -> Dict[str, List[Any]]:
        """✅ OPTIMIZED: Return a dictionary keyed on columns for batched input to the model, where each key points to a ordered list of values each row of that column will have"""
        
        try:
            if flattened_input == {}:
                flattened_input = {
                    "system_prompt": [],
                    "user_prompt": [],
                    "answer": [],
                    "metadata": [],
                }

            flattened_input["system_prompt"].append(self.system_prompt)
            flattened_input["user_prompt"].append(self.state_to_user_prompt(state))
            flattened_input["answer"].append(self.state_to_answer(state))

            # ✅ SAFE METADATA EXTRACTION
            try:
                if hasattr(state, 'environment_states') and "metadata" in state.environment_states:
                    flattened_input["metadata"].append(state.environment_states["metadata"])
                elif hasattr(state, "metadata") and state.metadata is not None:
                    flattened_input["metadata"].append(state.metadata)
                else:
                    flattened_input["metadata"].append({})
            except Exception as e:
                print(f"Metadata extraction failed: {e}")
                flattened_input["metadata"].append({})

            # ✅ MEMORY CHECK: Prevent unlimited flattening
            if len(flattened_input["system_prompt"]) > 10000:
                print(f"{Fore.YELLOW}⚠️ [FLATTEN] Too many flattened items, may cause memory issues{Style.RESET_ALL}")

            return flattened_input
            
        except Exception as e:
            print(f"flatten_states failed: {e}")
            return flattened_input if flattened_input else {}

    def prepare_environment(
        self,
        node_states: List[Any],
        swarm_states: Dict[Any, Any],
        stage: int,
        agent: Any,
        batch_id: Any,
    ) -> Any:
        """
        ✅ OPTIMIZED: Returns data that should be passed onto a node's children as an environment state when starting the next stage of the game.
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        try:
            if hasattr(node_states, 'environment_states'):
                return node_states.environment_states
            else:
                print(f"{Fore.YELLOW}⚠️ [PREP ENV] node_states has no environment_states, returning empty dict{Style.RESET_ALL}")
                return {}
        except Exception as e:
            print(f"prepare_environment failed: {e}")
            return {}

    def prepare_opponent(
        self,
        node_states: List[Any],
        swarm_states: Dict[Any, Any],
        stage: int,
        agent: Any,
        batch_id: Any,
    ) -> Any:
        """
        ✅ OPTIMIZED: Returns data that should be passed onto a node's children as an opponent state when starting the next stage of the game.
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        # Simple implementation - no opponent state needed for single stage
        return None

    def prepare_personal(
        self,
        node_states: List[Any],
        swarm_states: Dict[Any, Any],
        stage: int,
        agent: Any,
        batch_id: Any,
    ) -> Any:
        """
        ✅ OPTIMIZED: Returns data that should be passed onto a node's children as an personal state when starting the next stage of the game.
        NOTE: Said data can come from a node's states at the current stage, states received from communication with the swarm, and/or any other sources you choose to provide
        """
        # Simple implementation - no personal state needed for single stage
        return None

    # ✅ ADD: Enhanced cleanup for SimpleTextDataManager
    def cleanup(self):
        """Enhanced cleanup for SimpleTextDataManager"""
        try:
            print(f"{Fore.CYAN}🧹 [SIMPLE DATA] Enhanced cleanup initiated{Style.RESET_ALL}")
            
            # Clear simple data cache
            self.simple_data_cache.clear()
            
            # Call parent cleanup
            super().cleanup()
            
        except Exception as e:
            print(f"SimpleTextDataManager cleanup failed: {e}")
