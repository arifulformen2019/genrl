import abc
import gc  # ✅ ADD: Memory cleanup
import time  # ✅ ADD: Time tracking
import weakref  # ✅ ADD: Weak references
from collections import deque  # ✅ ADD: Bounded collections
from typing import Any, Callable, Dict, Iterable, List, Union

from genrl.rewards.reward_store import RewardFnStore
from genrl.state import GameState

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


class RewardManager(abc.ABC):
    """✅ OPTIMIZED: Abstract RewardManager with memory management"""
    
    @abc.abstractmethod
    def update_rewards(self, game_state: GameState) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self) -> Any:
        pass

    # ✅ ADD: Abstract cleanup method
    @abc.abstractmethod
    def cleanup_rewards(self) -> None:
        """Cleanup method to prevent memory leaks"""
        pass


class DefaultRewardManager(RewardManager):
    """✅ OPTIMIZED: DefaultRewardManager with comprehensive memory leak prevention"""
    
    def __init__(self, reward_fn_store: RewardFnStore):
        self._round = 0
        self._stage = 0
        
        # ✅ CRITICAL MEMORY LEAK FIX: Replace unlimited list with bounded deque
        self._rewards: deque = deque(maxlen=100)  # Keep only 100 recent rewards
        
        self.reward_fn_store = reward_fn_store

        # ✅ CRITICAL MEMORY LEAK FIX: Add reward memory management
        self._initialize_reward_memory_management()

    def _initialize_reward_memory_management(self):
        """Initialize memory management for RewardManager"""
        try:
            # ✅ REWARD HISTORY MANAGEMENT
            self.max_reward_history = 100           # Keep only 100 recent rewards
            self.max_reward_cache_size = 50         # Limit reward function cache
            self.max_stage_rewards = 20             # Max rewards per stage
            
            # ✅ REWARD COMPUTATION CACHE
            self.reward_computation_cache = {}
            self.max_computation_cache = 100
            
            # ✅ CLEANUP FREQUENCY CONTROL
            self.reward_update_counter = 0
            self.reward_cleanup_frequency = 25      # Cleanup every 25 updates
            self.last_reward_cleanup = time.time()
            self.reward_cleanup_interval = 180      # 3 minutes
            
            # ✅ MEMORY PRESSURE THRESHOLDS
            self.reward_memory_threshold = 8.0      # GB - trigger cleanup
            self.reward_emergency_threshold = 15.0  # GB - emergency cleanup
            
            # ✅ REWARD SIZE TRACKING
            self.large_reward_threshold_mb = 10     # Warn about rewards > 10MB
            self.total_reward_size_mb = 0.0
            
            print(f"{Fore.GREEN}🚀 [REWARD MANAGER] Memory optimization initialized{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Reward manager memory init failed: {e}")

    def _get_reward_memory_usage(self):
        """Get current memory usage in GB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024**3
            except:
                pass
        return 0.0

    def _check_reward_memory_pressure(self):
        """Check if reward memory cleanup is needed"""
        current_memory = self._get_reward_memory_usage()
        
        # Emergency cleanup
        if current_memory > self.reward_emergency_threshold:
            print(f"{Fore.RED}🚨 [REWARD EMERGENCY] Memory: {current_memory:.1f}GB - Emergency cleanup!{Style.RESET_ALL}")
            self._emergency_reward_cleanup()
            return True
            
        # High memory pressure cleanup
        elif current_memory > self.reward_memory_threshold:
            print(f"{Fore.YELLOW}⚠️ [REWARD PRESSURE] Memory: {current_memory:.1f}GB - Pressure cleanup{Style.RESET_ALL}")
            self._aggressive_reward_cleanup()
            return True
            
        # Time-based cleanup
        elif time.time() - self.last_reward_cleanup > self.reward_cleanup_interval:
            print(f"{Fore.CYAN}🧹 [REWARD CLEANUP] Periodic cleanup - Memory: {current_memory:.1f}GB{Style.RESET_ALL}")
            self._periodic_reward_cleanup()
            return True
            
        return False

    def _periodic_reward_cleanup(self):
        """Periodic reward memory cleanup - safe and conservative"""
        try:
            # Clean computation cache
            if len(self.reward_computation_cache) > self.max_computation_cache:
                # Keep only recent computations
                sorted_keys = sorted(self.reward_computation_cache.keys())
                old_keys = sorted_keys[:-self.max_computation_cache]
                for key in old_keys:
                    del self.reward_computation_cache[key]
                
                if old_keys:
                    print(f"{Fore.CYAN}🧹 [REWARD CACHE] Cleaned {len(old_keys)} old computations{Style.RESET_ALL}")
            
            # Light garbage collection
            collected = gc.collect()
            
            self.last_reward_cleanup = time.time()
            
            if collected > 0:
                print(f"{Fore.CYAN}🧹 [REWARD CLEANUP] Collected {collected} objects{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"Periodic reward cleanup failed: {e}")

    def _aggressive_reward_cleanup(self):
        """Aggressive reward memory cleanup for high memory pressure"""
        try:
            print(f"{Fore.YELLOW}💥 [REWARD AGGRESSIVE] Starting aggressive cleanup{Style.RESET_ALL}")
            
            initial_reward_count = len(self._rewards)
            
            # Clear most rewards, keep only recent ones
            if len(self._rewards) > 10:
                # Convert deque to list, keep last 10, convert back
                recent_rewards = list(self._rewards)[-10:]
                self._rewards.clear()
                self._rewards.extend(recent_rewards)
                
                print(f"{Fore.CYAN}💰 [REWARD AGGRESSIVE] Rewards: {initial_reward_count} → {len(self._rewards)}{Style.RESET_ALL}")
            
            # Clear all computation caches
            self.reward_computation_cache.clear()
            
            # Reset size tracking
            self.total_reward_size_mb = 0.0
            
            # Force garbage collection multiple times
            total_collected = 0
            for _ in range(5):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
            
            self.last_reward_cleanup = time.time()
            
            print(f"{Fore.GREEN}✅ [REWARD AGGRESSIVE] Collected {total_collected} objects{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Aggressive reward cleanup failed: {e}")

    def _emergency_reward_cleanup(self):
        """Emergency reward memory cleanup - nuclear option"""
        try:
            print(f"{Fore.RED}💥 [REWARD EMERGENCY] NUCLEAR CLEANUP INITIATED{Style.RESET_ALL}")
            
            initial_count = len(self._rewards)
            
            # Clear everything except absolute essentials
            self._rewards.clear()
            self.reward_computation_cache.clear()
            self.total_reward_size_mb = 0.0
            
            # Reset counters
            self.reward_update_counter = 0
            
            # Nuclear garbage collection
            total_collected = 0
            for _ in range(10):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
                time.sleep(0.1)  # Give system time to clean up
            
            print(f"{Fore.GREEN}✅ [REWARD EMERGENCY] Nuclear cleanup completed - {initial_count} rewards cleared, {total_collected} objects collected{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Emergency reward cleanup failed: {e}")

    def _estimate_reward_size(self, reward):
        """Estimate size of reward object in MB"""
        try:
            if hasattr(reward, '__sizeof__'):
                size_bytes = reward.__sizeof__()
                if hasattr(reward, '__len__'):
                    # For containers, estimate based on length
                    size_bytes += len(reward) * 100  # Rough estimate
                return size_bytes / 1024 / 1024  # Convert to MB
            return 0.1  # Default small size
        except:
            return 0.1

    @property
    def round(self) -> int:
        return self._round

    @round.setter
    def round(self, value: int) -> None:
        if value < 0:
            value = 0
        self._round = value

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        if value < 0:
            value = 0
        self._stage = value

    @property
    def rewards(self) -> deque:
        """✅ OPTIMIZED: Return bounded deque instead of unlimited list"""
        return self._rewards

    @rewards.setter
    def rewards(self, value: List[Any]) -> None:
        """✅ OPTIMIZED: Set rewards with memory management"""
        if not isinstance(value, (list, deque)):
            raise TypeError(f"Expected rewards to be a list or deque, but got {type(value)}")
        
        # ✅ MEMORY OPTIMIZATION: Convert to bounded deque
        self._rewards.clear()
        
        # ✅ LIMIT: Only keep recent rewards
        if isinstance(value, list):
            recent_rewards = value[-self.max_reward_history:] if len(value) > self.max_reward_history else value
        else:
            recent_rewards = list(value)[-self.max_reward_history:] if len(value) > self.max_reward_history else list(value)
        
        self._rewards.extend(recent_rewards)
        
        if len(value) > self.max_reward_history:
            print(f"{Fore.YELLOW}⚠️ [REWARD SET] Limited rewards: {len(value)} → {len(self._rewards)}{Style.RESET_ALL}")

    def __getitem__(self, stage: int) -> Any:
        """✅ OPTIMIZED: Get reward with bounds checking"""
        try:
            if stage >= len(self._rewards):
                print(f"{Fore.YELLOW}⚠️ [REWARD GET] Stage {stage} out of bounds for {len(self._rewards)} rewards{Style.RESET_ALL}")
                raise IndexError(
                    f"Stage {stage} is out of bounds for rewards list of length {len(self._rewards)}"
                )
            return self._rewards[stage]
        except Exception as e:
            print(f"Reward __getitem__ failed: {e}")
            raise

    def set_round_stage(self, round: int, stage: int) -> None:
        """✅ OPTIMIZED: Set round and stage with validation"""
        try:
            self.round = round
            self.stage = stage
        except Exception as e:
            print(f"Set round/stage failed: {e}")

    def dispatch_reward_fn(self, round: int, stage: int) -> Callable:
        """✅ OPTIMIZED: Dispatch reward function with caching"""
        try:
            # ✅ CACHE KEY for reward function lookup
            cache_key = f"{round}_{stage}"
            
            # ✅ CHECK CACHE FIRST
            if cache_key in self.reward_computation_cache:
                return self.reward_computation_cache[cache_key]
            
            # ✅ GET REWARD FUNCTION
            reward_fn = self.reward_fn_store[round].reward_fns[stage]
            
            # ✅ CACHE WITH SIZE LIMIT
            if len(self.reward_computation_cache) < self.max_computation_cache:
                self.reward_computation_cache[cache_key] = reward_fn
            
            return reward_fn
            
        except Exception as e:
            print(f"Reward function dispatch failed: {e}")
            # Return a dummy function that returns empty reward
            return lambda x: {}

    def __call__(
        self, round: int, stage: int, game_state: GameState
    ) -> Union[Iterable, Dict]:
        """
        ✅ OPTIMIZED: Dispatch the reward function with comprehensive memory management
        """
        try:
            # ✅ INCREMENT COUNTER AND CHECK MEMORY
            self.reward_update_counter += 1
            
            # ✅ PERIODIC MEMORY CHECKS
            if self.reward_update_counter % self.reward_cleanup_frequency == 0:
                self._check_reward_memory_pressure()
            
            # ✅ DISPATCH REWARD FUNCTION
            reward_fn = self.dispatch_reward_fn(round, stage)
            
            # ✅ COMPUTE REWARDS WITH ERROR HANDLING
            try:
                rewards = reward_fn(game_state)
            except Exception as reward_e:
                print(f"{Fore.RED}❌ [REWARD COMPUTE] Reward function failed: {reward_e}{Style.RESET_ALL}")
                rewards = {}  # Fallback to empty rewards
            
            # ✅ SIZE CHECK: Monitor large rewards
            reward_size_mb = self._estimate_reward_size(rewards)
            self.total_reward_size_mb += reward_size_mb
            
            if reward_size_mb > self.large_reward_threshold_mb:
                print(f"{Fore.YELLOW}⚠️ [REWARD SIZE] Large reward: {reward_size_mb:.1f}MB{Style.RESET_ALL}")
            
            # ✅ MEMORY-SAFE APPEND: deque automatically handles max size
            self._rewards.append(rewards)
            
            # ✅ MEMORY CHECK: Emergency cleanup if total size too large
            if self.total_reward_size_mb > 1000:  # 1GB of rewards
                print(f"{Fore.YELLOW}⚠️ [REWARD TOTAL] Total reward size: {self.total_reward_size_mb:.1f}MB{Style.RESET_ALL}")
                if self.total_reward_size_mb > 5000:  # 5GB
                    print(f"{Fore.RED}🚨 [REWARD TOTAL] Critical reward size, emergency cleanup{Style.RESET_ALL}")
                    self._emergency_reward_cleanup()
            
            return rewards
            
        except Exception as e:
            print(f"Reward __call__ failed: {e}")
            return {}

    def reset(self) -> None:
        """✅ OPTIMIZED: Reset with memory cleanup"""
        try:
            print(f"{Fore.CYAN}🔄 [REWARD RESET] Resetting rewards and stage{Style.RESET_ALL}")
            
            old_reward_count = len(self._rewards)
            
            self._stage = 0
            self._rewards.clear()  # Clear all rewards
            
            # ✅ RESET SIZE TRACKING
            self.total_reward_size_mb = 0.0
            
            # ✅ LIGHT CLEANUP
            gc.collect()
            
            if old_reward_count > 0:
                print(f"{Fore.GREEN}✅ [REWARD RESET] Cleared {old_reward_count} rewards{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"Reward reset failed: {e}")

    def update_rewards(self, game_state: GameState) -> None:
        """✅ OPTIMIZED: Update rewards with memory management"""
        try:
            print(f"{Fore.CYAN}💰 [REWARD UPDATE] Updating rewards for {game_state.stage} stages{Style.RESET_ALL}")
            
            # ✅ MEMORY CHECK BEFORE PROCESSING
            self._check_reward_memory_pressure()
            
            initial_memory = self._get_reward_memory_usage()
            
            # ✅ LIMIT STAGE PROCESSING to prevent memory explosion
            max_stages_to_process = min(game_state.stage, self.max_stage_rewards)
            
            if game_state.stage > self.max_stage_rewards:
                print(f"{Fore.YELLOW}⚠️ [REWARD UPDATE] Too many stages ({game_state.stage}), limiting to {max_stages_to_process}{Style.RESET_ALL}")
            
            # ✅ PROCESS STAGES WITH MEMORY MONITORING
            processed_stages = 0
            for stage in range(max_stages_to_process):
                try:
                    # ✅ MEMORY CHECK every 5 stages
                    if stage > 0 and stage % 5 == 0:
                        current_memory = self._get_reward_memory_usage()
                        memory_increase = current_memory - initial_memory
                        
                        if memory_increase > 2.0:  # 2GB increase
                            print(f"{Fore.YELLOW}⚠️ [REWARD UPDATE] Memory increase: +{memory_increase:.1f}GB after {stage} stages{Style.RESET_ALL}")
                            
                        if memory_increase > 5.0:  # 5GB increase - emergency stop
                            print(f"{Fore.RED}🚨 [REWARD UPDATE] Critical memory increase, stopping at stage {stage}{Style.RESET_ALL}")
                            break
                    
                    # ✅ CALL REWARD FUNCTION
                    self.__call__(game_state.round, stage, game_state)
                    processed_stages += 1
                    
                except Exception as stage_e:
                    print(f"{Fore.RED}❌ [REWARD UPDATE] Stage {stage} failed: {stage_e}{Style.RESET_ALL}")
                    continue
            
            # ✅ INCREMENT ROUND
            self.round += 1
            
            final_memory = self._get_reward_memory_usage()
            memory_change = final_memory - initial_memory
            
            print(f"{Fore.GREEN}✅ [REWARD UPDATE] Processed {processed_stages} stages, memory change: {memory_change:+.1f}GB{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Reward update failed: {e}")

    # ✅ IMPLEMENT: Required abstract method
    def cleanup_rewards(self) -> None:
        """Manual cleanup method for RewardManager"""
        try:
            print(f"{Fore.CYAN}🧹 [REWARD MANAGER] Manual cleanup initiated{Style.RESET_ALL}")
            
            # Emergency cleanup
            self._emergency_reward_cleanup()
            
            print(f"{Fore.GREEN}✅ [REWARD MANAGER] Cleanup completed{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Manual reward cleanup failed: {e}")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.cleanup_rewards()
        except:
            pass

    # ✅ ADD: Debug and monitoring methods
    def get_reward_stats(self):
        """Get reward statistics for monitoring"""
        try:
            stats = {
                'total_rewards': len(self._rewards),
                'current_round': self._round,
                'current_stage': self._stage,
                'update_counter': self.reward_update_counter,
                'computation_cache_size': len(self.reward_computation_cache),
                'total_reward_size_mb': self.total_reward_size_mb,
                'memory_usage_gb': self._get_reward_memory_usage(),
            }
            return stats
        except Exception as e:
            print(f"Failed to get reward stats: {e}")
            return {}

    def debug_reward_memory(self):
        """Debug method to show reward memory usage"""
        try:
            memory_gb = self._get_reward_memory_usage()
            
            print(f"{Fore.BLUE}🔍 [REWARD DEBUG] Memory Usage:{Style.RESET_ALL}")
            print(f"   💾 Total RAM: {memory_gb:.2f}GB")
            print(f"   💰 Total rewards: {len(self._rewards)}")
            print(f"   🔄 Current round: {self._round}")
            print(f"   📊 Current stage: {self._stage}")
            print(f"   📈 Update counter: {self.reward_update_counter}")
            print(f"   💾 Computation cache: {len(self.reward_computation_cache)} items")
            print(f"   📏 Total reward size: {self.total_reward_size_mb:.1f}MB")
            
            # Show recent reward sizes
            if len(self._rewards) > 0:
                recent_rewards = list(self._rewards)[-5:]  # Last 5 rewards
                print(f"   📋 Recent reward sizes:")
                for i, reward in enumerate(recent_rewards):
                    size_mb = self._estimate_reward_size(reward)
                    print(f"      {i}: {size_mb:.2f}MB")
            
            return {
                'memory_gb': memory_gb,
                'reward_count': len(self._rewards),
                'cache_size': len(self.reward_computation_cache),
                'total_size_mb': self.total_reward_size_mb,
            }
            
        except Exception as e:
            print(f"Debug reward memory failed: {e}")
            return {}
