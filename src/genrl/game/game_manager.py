import abc
import gc  # ✅ ADD: Memory cleanup
import time  # ✅ ADD: Time tracking
import weakref  # ✅ ADD: Weak references
from collections import deque  # ✅ ADD: Bounded collections
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

from genrl.communication import Communication
from genrl.communication.communication import Communication
from genrl.data import DataManager
from genrl.logging_utils.global_defs import get_logger
from genrl.rewards import RewardManager
from genrl.roles import RoleManager  # TODO: Implement RoleManager+Pass to game manager
from genrl.state import GameNode, GameState
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


class RunType(Enum):
    Train = "train"
    Evaluate = "evaluate"
    TrainAndEvaluate = "train_and_evaluate"


class GameManager(abc.ABC):
    def __init__(
        self,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication | None = None,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        rank: int = 0,
        **kwargs,
    ):
        """✅ OPTIMIZED: Initialization method that stores the various managers needed to orchestrate this game"""
        self.state = game_state
        self.rewards = reward_manager
        self.trainer = trainer
        self.data_manager = data_manager
        self.communication = communication or Communication.create(**kwargs)
        self.roles = role_manager
        try:
            self.mode = RunType(run_mode)
        except ValueError:
            get_logger().info(
                f"Invalid run mode: {run_mode}. Defaulting to train only."
            )
            self.mode = RunType.Train
        self._rank = rank or self.communication.get_id()
        self.agent_ids = [self._rank]

        # ✅ CRITICAL MEMORY LEAK FIX: Add game manager memory management
        self._initialize_game_memory_management()

    def _initialize_game_memory_management(self):
        """Initialize memory management for GameManager"""
        try:
            # ✅ STAGE/ROUND HISTORY MANAGEMENT
            self.max_stage_history = 20     # Keep only 20 recent stages
            self.max_round_history = 10     # Keep only 10 recent rounds
            self.stage_history = deque(maxlen=self.max_stage_history)
            self.round_history = deque(maxlen=self.max_round_history)
            
            # ✅ GAME STATE CACHE MANAGEMENT
            self.max_game_state_cache = 5   # Keep only 5 recent game states
            self.game_state_cache = deque(maxlen=self.max_game_state_cache)
            
            # ✅ COMMUNICATION PAYLOAD MANAGEMENT
            self.max_swarm_payloads = 100   # Limit swarm payloads
            self.swarm_payload_history = deque(maxlen=self.max_swarm_payloads)
            
            # ✅ CLEANUP FREQUENCY CONTROL
            self.stage_counter = 0
            self.round_counter = 0
            self.game_cleanup_frequency = 10  # Cleanup every 10 stages
            self.last_game_memory_cleanup = time.time()
            self.game_memory_cleanup_interval = 180  # 3 minutes
            
            # ✅ MEMORY PRESSURE THRESHOLDS  
            self.game_memory_pressure_threshold = 12.0  # GB
            self.game_emergency_threshold = 20.0  # GB
            
            # ✅ WORLD STATE MANAGEMENT
            self.max_world_states_per_agent = 50  # Limit world states per agent
            
            get_logger().info(f"{Fore.GREEN}🚀 [GAME MANAGER] Memory optimization initialized{Style.RESET_ALL}")
            
        except Exception as e:
            get_logger().warning(f"Game manager memory init failed: {e}")

    def _get_game_memory_usage(self):
        """Get current memory usage in GB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024**3
            except:
                pass
        return 0.0

    def _check_game_memory_pressure(self):
        """Check if game memory cleanup is needed"""
        current_memory = self._get_game_memory_usage()
        
        # Emergency cleanup
        if current_memory > self.game_emergency_threshold:
            get_logger().error(f"{Fore.RED}🚨 [GAME EMERGENCY] Memory: {current_memory:.1f}GB - Emergency cleanup!{Style.RESET_ALL}")
            self._emergency_game_memory_cleanup()
            return True
            
        # High memory pressure cleanup
        elif current_memory > self.game_memory_pressure_threshold:
            get_logger().warning(f"{Fore.YELLOW}⚠️ [GAME PRESSURE] Memory: {current_memory:.1f}GB - Pressure cleanup{Style.RESET_ALL}")
            self._aggressive_game_memory_cleanup()
            return True
            
        # Time-based cleanup
        elif time.time() - self.last_game_memory_cleanup > self.game_memory_cleanup_interval:
            get_logger().info(f"{Fore.CYAN}🧹 [GAME CLEANUP] Periodic cleanup - Memory: {current_memory:.1f}GB{Style.RESET_ALL}")
            self._periodic_game_memory_cleanup()
            return True
            
        return False

    def _periodic_game_memory_cleanup(self):
        """Periodic game memory cleanup - safe and conservative"""
        try:
            # Clean stage/round history (handled by deque automatically)
            
            # Clean game state if it has internal caches
            if hasattr(self.state, '_clear_caches'):
                self.state._clear_caches()
            
            # Clean world state caches
            self._clean_world_state_caches()
            
            # Light garbage collection
            collected = gc.collect()
            
            self.last_game_memory_cleanup = time.time()
            
            if collected > 0:
                get_logger().info(f"{Fore.CYAN}🧹 [GAME CLEANUP] Collected {collected} objects{Style.RESET_ALL}")
                
        except Exception as e:
            get_logger().error(f"Periodic game cleanup failed: {e}")

    def _aggressive_game_memory_cleanup(self):
        """Aggressive game memory cleanup for high memory pressure"""
        try:
            get_logger().warning(f"{Fore.YELLOW}💥 [GAME AGGRESSIVE] Starting aggressive cleanup{Style.RESET_ALL}")
            
            # Clear all history caches
            self.stage_history.clear()
            self.round_history.clear()
            self.game_state_cache.clear()
            self.swarm_payload_history.clear()
            
            # Aggressively clean game state
            self._aggressive_clean_game_state()
            
            # Clean communication payloads
            self._clean_communication_payloads()
            
            # Force garbage collection multiple times
            total_collected = 0
            for _ in range(5):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
            
            self.last_game_memory_cleanup = time.time()
            
            get_logger().info(f"{Fore.GREEN}✅ [GAME AGGRESSIVE] Collected {total_collected} objects{Style.RESET_ALL}")
            
        except Exception as e:
            get_logger().error(f"Aggressive game cleanup failed: {e}")

    def _emergency_game_memory_cleanup(self):
        """Emergency game memory cleanup - nuclear option"""
        try:
            get_logger().error(f"{Fore.RED}💥 [GAME EMERGENCY] NUCLEAR CLEANUP INITIATED{Style.RESET_ALL}")
            
            # Clear everything
            self.stage_history.clear()
            self.round_history.clear()
            self.game_state_cache.clear()
            self.swarm_payload_history.clear()
            
            # Nuclear game state cleanup
            self._nuclear_clean_game_state()
            
            # Nuclear garbage collection
            total_collected = 0
            for _ in range(10):
                collected = gc.collect()
                total_collected += collected
                if collected == 0:
                    break
                time.sleep(0.1)  # Give system time to clean up
            
            get_logger().info(f"{Fore.GREEN}✅ [GAME EMERGENCY] Nuclear cleanup completed - {total_collected} objects{Style.RESET_ALL}")
            
        except Exception as e:
            get_logger().error(f"Emergency game cleanup failed: {e}")

    def _clean_world_state_caches(self):
        """Clean world state caches to prevent accumulation"""
        try:
            # Clean game state world state caches
            if hasattr(self.state, 'trees') and isinstance(self.state.trees, dict):
                for agent_id, agent_trees in self.state.trees.items():
                    if isinstance(agent_trees, dict):
                        # Limit trees per agent
                        if len(agent_trees) > self.max_world_states_per_agent:
                            # Keep only recent trees
                            sorted_batches = sorted(agent_trees.keys())
                            old_batches = sorted_batches[:-self.max_world_states_per_agent]
                            
                            for batch_id in old_batches:
                                # Clean up tree before deleting
                                tree = agent_trees[batch_id]
                                if tree is not None and hasattr(tree, 'clear'):
                                    tree.clear()
                                del agent_trees[batch_id]
                            
                            get_logger().info(f"{Fore.CYAN}🌳 [WORLD STATE] Pruned {len(old_batches)} old trees for agent {agent_id}{Style.RESET_ALL}")
            
            # Clean any other world state caches
            if hasattr(self.state, '_world_state_cache'):
                self.state._world_state_cache.clear()
                
        except Exception as e:
            get_logger().error(f"World state cache cleanup failed: {e}")

    def _aggressive_clean_game_state(self):
        """Aggressively clean game state internal structures"""
        try:
            # Clear game state internal caches
            state_attrs = [
                '_cached_states', '_state_cache', '_world_state_cache',
                '_agent_cache', '_batch_cache', '_stage_cache'
            ]
            
            cleaned_attrs = 0
            for attr in state_attrs:
                if hasattr(self.state, attr):
                    cache = getattr(self.state, attr)
                    if hasattr(cache, 'clear'):
                        cache.clear()
                        cleaned_attrs += 1
                    elif isinstance(cache, dict):
                        cache.clear()
                        cleaned_attrs += 1
            
            if cleaned_attrs > 0:
                get_logger().info(f"{Fore.CYAN}🗑️ [GAME STATE] Cleaned {cleaned_attrs} game state caches{Style.RESET_ALL}")
                
        except Exception as e:
            get_logger().error(f"Aggressive game state cleanup failed: {e}")

    def _nuclear_clean_game_state(self):
        """Nuclear game state cleanup - clear everything possible"""
        try:
            # Clear trees completely and recreate minimal structure
            if hasattr(self.state, 'trees'):
                # Keep only current agent's essential data
                if isinstance(self.state.trees, dict) and self._rank in self.state.trees:
                    # Keep only 1 most recent tree for current agent
                    agent_trees = self.state.trees[self._rank]
                    if isinstance(agent_trees, dict) and len(agent_trees) > 1:
                        # Keep only the most recent batch
                        latest_batch = max(agent_trees.keys()) if agent_trees else None
                        if latest_batch is not None:
                            latest_tree = agent_trees[latest_batch]
                            agent_trees.clear()
                            agent_trees[latest_batch] = latest_tree
                        else:
                            agent_trees.clear()
                    
                    # Clear all other agents
                    agents_to_remove = [agent for agent in self.state.trees.keys() if agent != self._rank]
                    for agent in agents_to_remove:
                        del self.state.trees[agent]
                
                get_logger().info(f"{Fore.RED}💥 [NUCLEAR] Game state trees nuclear cleaned{Style.RESET_ALL}")
            
            # Clear all other state attributes
            nuclear_attrs = [
                '_cached_latest_state', '_cached_communication', '_stage_history',
                '_round_history', '_all_caches'
            ]
            
            for attr in nuclear_attrs:
                if hasattr(self.state, attr):
                    try:
                        setattr(self.state, attr, None)
                    except:
                        pass
                        
        except Exception as e:
            get_logger().error(f"Nuclear game state cleanup failed: {e}")

    def _clean_communication_payloads(self):
        """Clean communication payloads to prevent memory accumulation"""
        try:
            # Clear communication internal buffers
            comm_attrs = [
                '_message_buffer', '_payload_cache', '_swarm_payloads',
                '_communication_history', '_gathered_objects'
            ]
            
            cleaned_comm = 0
            for attr in comm_attrs:
                if hasattr(self.communication, attr):
                    cache = getattr(self.communication, attr)
                    if hasattr(cache, 'clear'):
                        cache.clear()
                        cleaned_comm += 1
                    elif isinstance(cache, (list, dict)):
                        if isinstance(cache, list):
                            cache.clear()
                        else:
                            cache.clear()
                        cleaned_comm += 1
            
            if cleaned_comm > 0:
                get_logger().info(f"{Fore.CYAN}📡 [COMMUNICATION] Cleaned {cleaned_comm} communication buffers{Style.RESET_ALL}")
                
        except Exception as e:
            get_logger().error(f"Communication cleanup failed: {e}")

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, rank: int) -> None:
        self._rank = rank

    @abc.abstractmethod
    def end_of_game(self) -> bool:
        """
        Defines conditions for the game to end and no more rounds/stage should begin.
        Return True if conditions imply game should end, else False
        """
        pass

    @abc.abstractmethod
    def end_of_round(self) -> bool:
        """
        Defines conditions for end of a round AND no more stages/"turns" should being for this round AND the game state should be reset for stage 0 of your game.
        Return True if conditions imply game should end and no new round/stage should begin, else False
        """
        pass

    def _hook_after_rewards_updated(self):
        """✅ OPTIMIZED: Hook method called after rewards are updated."""
        try:
            # Add to round history
            if hasattr(self, 'round_history'):
                self.round_history.append(('rewards_updated', time.time()))
            
            # Periodic cleanup after rewards
            if hasattr(self, 'round_counter') and self.round_counter % 5 == 0:
                self._periodic_game_memory_cleanup()
                
        except Exception as e:
            get_logger().error(f"Hook after rewards updated failed: {e}")

    def _hook_after_round_advanced(self):
        """✅ OPTIMIZED: Hook method called after the round is advanced and rewards are reset."""
        try:
            # Increment round counter
            if hasattr(self, 'round_counter'):
                self.round_counter += 1
            else:
                self.round_counter = 1
            
            # Add to round history
            if hasattr(self, 'round_history'):
                self.round_history.append(('round_advanced', time.time(), self.round_counter))
            
            # Memory cleanup after round advancement
            if self.round_counter % 3 == 0:  # Every 3 rounds
                self._check_game_memory_pressure()
                
        except Exception as e:
            get_logger().error(f"Hook after round advanced failed: {e}")

    def _hook_after_game(self):
        """✅ OPTIMIZED: Hook method called after the game is finished."""
        try:
            get_logger().info(f"{Fore.MAGENTA}🎮 [GAME END] Game finished - Final cleanup{Style.RESET_ALL}")
            
            # Final cleanup
            self._emergency_game_memory_cleanup()
            
        except Exception as e:
            get_logger().error(f"Hook after game failed: {e}")

    # Helper methods
    def aggregate_game_state_methods(
        self,
    ) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
        """✅ OPTIMIZED: Aggregate game state methods with memory awareness"""
        world_state_pruners = {
            "environment_pruner": getattr(self, "environment_state_pruner", None),
            "opponent_pruner": getattr(self, "opponent_state_pruner", None),
            "personal_pruner": getattr(self, "personal_state_pruner", None),
        }
        game_tree_brancher = {
            "terminal_node_decision_function": getattr(
                self, "terminal_game_tree_node_decider", None
            ),
            "stage_inheritance_function": getattr(
                self, "stage_inheritance_function", None
            ),
        }
        return world_state_pruners, game_tree_brancher

    # Core (default) game orchestration methods
    def run_game_stage(self):
        """✅ OPTIMIZED: Run game stage with memory management"""
        try:
            # Increment stage counter
            if hasattr(self, 'stage_counter'):
                self.stage_counter += 1
            else:
                self.stage_counter = 1
            
            # Add to stage history
            if hasattr(self, 'stage_history'):
                self.stage_history.append(('stage_start', time.time(), self.stage_counter))
            
            # ✅ MEMORY CHECK: Before expensive operations
            if self.stage_counter % self.game_cleanup_frequency == 0:
                self._check_game_memory_pressure()
            
            # Core stage logic with memory cleanup
            inputs = self.state.get_latest_state()  # Fetches the current world state for all agents
            
            # ✅ MEMORY OPTIMIZATION: Limit input size
            if isinstance(inputs, dict):
                total_inputs = sum(
                    len(agent_data) if isinstance(agent_data, dict) else 0
                    for agent_data in inputs.values()
                )
                if total_inputs > 10000:  # Arbitrary limit
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [STAGE] Too many inputs ({total_inputs}), may cause memory issues{Style.RESET_ALL}")
            
            inputs, index_mapping = self.data_manager.prepare_input(
                inputs, self.state.stage
            )  # Maps game tree states to model ingestable inputs
            
            outputs = self.trainer.generate(
                inputs
            )  # Generates a rollout
            
            actions = self.data_manager.prepare_actions(
                outputs, index_mapping
            )  # Maps model outputs to RL game tree actions
            
            self.state.append_actions(
                actions
            )  # Adds the freshly generated rollout to the game state
            
            # ✅ CLEANUP: Clear intermediate variables
            del inputs, outputs, actions, index_mapping
            
            # Add to stage history
            if hasattr(self, 'stage_history'):
                self.stage_history.append(('stage_end', time.time(), self.stage_counter))
            
            # ✅ MEMORY CLEANUP: Periodic cleanup during stages
            if self.stage_counter % 20 == 0:  # Every 20 stages
                gc.collect()
                
        except Exception as e:
            get_logger().error(f"run_game_stage failed: {e}")
            raise

    def run_game_round(self):
        """✅ OPTIMIZED: Run game round with memory management"""
        try:
            get_logger().info(f"{Fore.MAGENTA}🎮 [ROUND START] Starting round {getattr(self.state, 'round', 'unknown')}{Style.RESET_ALL}")
            
            # Loop through stages until end of round is hit
            while not self.end_of_round():
                self.run_game_stage()  # Generates rollout and updates the game state
                
                # ✅ COMMUNICATION WITH MEMORY MANAGEMENT
                try:
                    latest_comm = self.state.get_latest_communication()
                    if isinstance(latest_comm, dict) and self.rank in latest_comm:
                        swarm_payloads = self.communication.all_gather_object(
                            latest_comm[self.rank]
                        )
                    else:
                        get_logger().warning(f"No communication data for rank {self.rank}")
                        swarm_payloads = {}
                    
                    # ✅ LIMIT SWARM PAYLOADS TO PREVENT MEMORY EXPLOSION
                    if isinstance(swarm_payloads, dict) and len(swarm_payloads) > self.max_swarm_payloads:
                        get_logger().warning(f"{Fore.YELLOW}⚠️ [SWARM] Too many payloads ({len(swarm_payloads)}), limiting{Style.RESET_ALL}")
                        # Keep only recent payloads (by key sorting)
                        sorted_keys = sorted(swarm_payloads.keys())
                        limited_payloads = {k: swarm_payloads[k] for k in sorted_keys[-self.max_swarm_payloads:]}
                        swarm_payloads = limited_payloads
                    
                    # Add to payload history
                    if hasattr(self, 'swarm_payload_history'):
                        self.swarm_payload_history.append(('payloads', time.time(), len(swarm_payloads) if isinstance(swarm_payloads, dict) else 0))
                    
                except Exception as e:
                    get_logger().error(f"Communication failed: {e}")
                    swarm_payloads = {}
                
                # Prepare states with memory management
                try:
                    world_states = self.data_manager.prepare_states(
                        self.state, swarm_payloads
                    )  # Maps states received via communication with the swarm to RL game tree world states
                    
                    self.state.advance_stage(world_states)  # Prepare for next stage
                    
                    # ✅ CLEANUP: Clear intermediate variables
                    del world_states, swarm_payloads
                    
                except Exception as e:
                    get_logger().error(f"State preparation failed: {e}")

            # ✅ REWARD PROCESSING WITH MEMORY MANAGEMENT
            try:
                self.rewards.update_rewards(self.state)  # Compute reward functions
                self._hook_after_rewards_updated()  # Call hook
            except Exception as e:
                get_logger().error(f"Reward update failed: {e}")

            # ✅ TRAINING/EVALUATION WITH MEMORY MANAGEMENT
            try:
                if self.mode in [RunType.Train, RunType.TrainAndEvaluate]:
                    self.trainer.train(self.state, self.data_manager, self.rewards)
                if self.mode in [RunType.Evaluate, RunType.TrainAndEvaluate]:
                    self.trainer.evaluate(self.state, self.data_manager, self.rewards)
            except Exception as e:
                get_logger().error(f"Training/evaluation failed: {e}")

            # ✅ ROUND ADVANCEMENT WITH MEMORY MANAGEMENT
            try:
                round_data = self.data_manager.get_round_data()
                self.state.advance_round(
                    round_data, agent_keys=self.agent_ids
                )  # Resets the game state appropriately
                
                self.rewards.reset()
                self._hook_after_round_advanced()  # Call hook
                
                # ✅ CLEANUP: Clear round data
                del round_data
                
            except Exception as e:
                get_logger().error(f"Round advancement failed: {e}")
            
            get_logger().info(f"{Fore.GREEN}✅ [ROUND END] Round {getattr(self.state, 'round', 'unknown')} completed{Style.RESET_ALL}")
            
        except Exception as e:
            get_logger().error(f"run_game_round failed: {e}")
            raise

    def run_game(self):
        """✅ OPTIMIZED: Run game with comprehensive memory management"""
        try:
            get_logger().info(f"{Fore.MAGENTA}🎮 [GAME START] Starting game with memory optimization{Style.RESET_ALL}")
            
            # Initialize game and/or run specific details of game state
            world_state_pruners, game_tree_brancher = self.aggregate_game_state_methods()
            
            # ✅ GAME INITIALIZATION WITH MEMORY MANAGEMENT
            try:
                round_data = self.data_manager.get_round_data()
                self.state._init_game(
                    round_data,
                    agent_keys=self.agent_ids,
                    world_state_pruners=world_state_pruners,
                    game_tree_brancher=game_tree_brancher,
                )  # Prepare game trees within the game state for the initial round's batch of data
                
                # ✅ CLEANUP: Clear initialization data
                del round_data, world_state_pruners, game_tree_brancher
                
            except Exception as e:
                get_logger().error(f"Game initialization failed: {e}")
                raise
            
            # Loop through rounds until end of the game is hit
            try:
                while not self.end_of_game():
                    current_round = getattr(self.state, 'round', 'unknown')
                    max_round = getattr(self, 'max_round', 'unknown')
                    
                    get_logger().info(
                        f"{Fore.CYAN}🎯 [GAME LOOP] Starting round: {current_round}/{max_round}{Style.RESET_ALL}"
                    )
                    
                    # ✅ PRE-ROUND MEMORY CHECK
                    self._check_game_memory_pressure()
                    
                    self.run_game_round()  # Loops through stages until end of round signal is received
                    
                    # ✅ POST-ROUND CLEANUP
                    if hasattr(self, 'round_counter') and self.round_counter % 5 == 0:
                        gc.collect()
                        get_logger().info(f"{Fore.CYAN}🧹 [GAME] Post-round cleanup after round {self.round_counter}{Style.RESET_ALL}")
                        
            except KeyboardInterrupt:
                get_logger().info(f"{Fore.YELLOW}⏹️ [GAME] Game interrupted by user{Style.RESET_ALL}")
                raise
            except Exception as e:
                get_logger().exception(
                    "Exception occurred during game run.", stack_info=True
                )
                raise
            finally:
                try:
                    self._hook_after_game()
                    if hasattr(self.trainer, 'cleanup'):
                        self.trainer.cleanup()
                    
                    # ✅ FINAL MEMORY CLEANUP
                    self._emergency_game_memory_cleanup()
                    
                    get_logger().info(f"{Fore.GREEN}🎮 [GAME END] Game completed successfully{Style.RESET_ALL}")
                    
                except Exception as cleanup_e:
                    get_logger().error(f"Game cleanup failed: {cleanup_e}")
                    
        except Exception as e:
            get_logger().error(f"run_game failed: {e}")
            raise

    # ✅ ADD: Cleanup methods
    def cleanup_game_manager(self):
        """Manual cleanup method for GameManager"""
        try:
            get_logger().info(f"{Fore.CYAN}🧹 [GAME MANAGER] Manual cleanup initiated{Style.RESET_ALL}")
            self._emergency_game_memory_cleanup()
        except Exception as e:
            get_logger().error(f"Manual game manager cleanup failed: {e}")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.cleanup_game_manager()
        except:
            pass


class DefaultGameManagerMixin:
    """
    ✅ OPTIMIZED: Defines some default behaviour for games with memory management
    """

    # Optional methods
    def environment_state_pruner(self, input: Any) -> Any:
        """
        ✅ OPTIMIZED: Optional pruning function for environment states with memory limits
        """
        try:
            # ✅ MEMORY OPTIMIZATION: Limit input size
            if isinstance(input, (list, dict)):
                if isinstance(input, list) and len(input) > 1000:
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [ENV PRUNER] Large input list ({len(input)}), limiting to 1000{Style.RESET_ALL}")
                    input = input[:1000]
                elif isinstance(input, dict) and len(input) > 1000:
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [ENV PRUNER] Large input dict ({len(input)}), limiting to 1000{Style.RESET_ALL}")
                    # Keep only first 1000 items
                    input = dict(list(input.items())[:1000])
            
            return input
            
        except Exception as e:
            get_logger().error(f"Environment state pruner failed: {e}")
            return input

    def opponent_state_pruner(self, input: Any) -> Any:
        """
        ✅ OPTIMIZED: Optional pruning function for opponent states with memory limits
        """
        try:
            # ✅ MEMORY OPTIMIZATION: Limit opponent states
            if isinstance(input, (list, dict)):
                if isinstance(input, list) and len(input) > 500:
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [OPP PRUNER] Large opponent list ({len(input)}), limiting to 500{Style.RESET_ALL}")
                    input = input[:500]
                elif isinstance(input, dict) and len(input) > 500:
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [OPP PRUNER] Large opponent dict ({len(input)}), limiting to 500{Style.RESET_ALL}")
                    input = dict(list(input.items())[:500])
            
            return input
            
        except Exception as e:
            get_logger().error(f"Opponent state pruner failed: {e}")
            return input

    def personal_state_pruner(self, input: Any) -> Any:
        """
        ✅ OPTIMIZED: Optional pruning function for personal states with memory limits
        """
        try:
            # ✅ MEMORY OPTIMIZATION: Limit personal states
            if isinstance(input, (list, dict)):
                if isinstance(input, list) and len(input) > 200:
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [PERSONAL PRUNER] Large personal list ({len(input)}), limiting to 200{Style.RESET_ALL}")
                    input = input[:200]
                elif isinstance(input, dict) and len(input) > 200:
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [PERSONAL PRUNER] Large personal dict ({len(input)}), limiting to 200{Style.RESET_ALL}")
                    input = dict(list(input.items())[:200])
            
            return input
            
        except Exception as e:
            get_logger().error(f"Personal state pruner failed: {e}")
            return input

    def terminal_game_tree_node_decider(
        self, stage_nodes: List[GameNode]
    ) -> List[GameNode]:
        """
        ✅ OPTIMIZED: Optional function defining whether nodes are terminal with memory limits
        """
        try:
            terminal = []
            
            # ✅ MEMORY OPTIMIZATION: Limit processing of nodes
            if len(stage_nodes) > 10000:
                get_logger().warning(f"{Fore.YELLOW}⚠️ [TERMINAL DECIDER] Too many nodes ({len(stage_nodes)}), limiting to 10000{Style.RESET_ALL}")
                stage_nodes = stage_nodes[:10000]
            
            for node in stage_nodes:
                try:
                    if hasattr(node, 'stage') and hasattr(self, 'max_stage'):
                        if node["stage"] < self.max_stage - 1:
                            pass  # Not terminal
                        else:
                            terminal.append(node)
                    else:
                        # Fallback logic if attributes missing
                        terminal.append(node)
                        
                except Exception as node_e:
                    get_logger().debug(f"Node processing failed: {node_e}")
                    continue
            
            # ✅ MEMORY OPTIMIZATION: Limit terminal nodes
            if len(terminal) > 1000:
                get_logger().warning(f"{Fore.YELLOW}⚠️ [TERMINAL DECIDER] Too many terminal nodes ({len(terminal)}), limiting to 1000{Style.RESET_ALL}")
                terminal = terminal[:1000]
            
            return terminal
            
        except Exception as e:
            get_logger().error(f"Terminal node decider failed: {e}")
            return []

    def stage_inheritance_function(
        self, stage_nodes: List[GameNode]
    ) -> List[List[GameNode]]:
        """
        ✅ OPTIMIZED: Optional function for stage inheritance with memory limits
        """
        try:
            stage_children = []
            
            # ✅ MEMORY OPTIMIZATION: Limit processing of nodes
            if len(stage_nodes) > 5000:
                get_logger().warning(f"{Fore.YELLOW}⚠️ [INHERITANCE] Too many nodes ({len(stage_nodes)}), limiting to 5000{Style.RESET_ALL}")
                stage_nodes = stage_nodes[:5000]
            
            for i, node in enumerate(stage_nodes):
                children = []
                
                try:
                    if hasattr(node, '_is_leaf_node') and not node._is_leaf_node():
                        child = GameNode(
                            stage=node.stage + 1,
                            node_idx=0,  # Will be overwritten by the game tree if not correct
                            world_state=node.world_state,
                            actions=None,
                        )
                        children.append(child)
                        
                        # ✅ MEMORY OPTIMIZATION: Clear reference to parent node's world_state if large
                        if hasattr(node, 'world_state') and hasattr(node.world_state, '__sizeof__'):
                            if node.world_state.__sizeof__() > 1024 * 1024:  # 1MB
                                # Don't clear, but log warning
                                get_logger().debug(f"Large world state detected in node {i}")
                    
                except Exception as node_e:
                    get_logger().debug(f"Node inheritance failed for node {i}: {node_e}")
                    
                stage_children.append(children)
                
                # ✅ MEMORY CHECK: Break if too many children
                if len(stage_children) > 5000:
                    get_logger().warning(f"{Fore.YELLOW}⚠️ [INHERITANCE] Generated too many children ({len(stage_children)}), stopping{Style.RESET_ALL}")
                    break
            
            return stage_children
            
        except Exception as e:
            get_logger().error(f"Stage inheritance function failed: {e}")
            return []


class BaseGameManager(DefaultGameManagerMixin, GameManager):
    """
    ✅ OPTIMIZED: Basic GameManager with memory management and basic functionality baked-in.
    Will end the game when max_rounds is reached, end a round when max_stage is reached.
    """

    def __init__(
        self,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication | None = None,
        role_manager: RoleManager | None = None,
        run_mode: str = "Train",
    ):
        """✅ OPTIMIZED: Init a GameManager with memory management"""
        
        self.max_stage = max_stage
        self.max_round = max_round
        
        kwargs = {
            "game_state": game_state,
            "reward_manager": reward_manager,
            "trainer": trainer,
            "data_manager": data_manager,
            "communication": communication,
            "role_manager": role_manager,
            "run_mode": run_mode,
        }
        
        # Initialize parent with memory management
        super().__init__(**kwargs)
        
        # ✅ BASE GAME MANAGER SPECIFIC MEMORY MANAGEMENT
        self.base_memory_threshold = 8.0  # GB - lower threshold for base manager
        
        get_logger().info(
            f"{Fore.GREEN}🚀 [BASE GAME] Memory-optimized BaseGameManager initialized\n"
            f"   🎯 Max Stage: {max_stage}\n"
            f"   🔄 Max Round: {max_round}\n"
            f"   💾 Memory Threshold: {self.base_memory_threshold}GB{Style.RESET_ALL}"
        )

    def end_of_game(self) -> bool:
        """✅ OPTIMIZED: Check if game should end with memory consideration"""
        try:
            # Standard end condition
            if self.state.round < self.max_round:
                return False
            else:
                get_logger().info(f"{Fore.MAGENTA}🏁 [GAME END] Max rounds reached: {self.state.round}/{self.max_round}{Style.RESET_ALL}")
                return True
                
        except Exception as e:
            get_logger().error(f"end_of_game check failed: {e}")
            # Default to not ending game on error
            return False

    def end_of_round(self) -> bool:
        """✅ OPTIMIZED: Check if round should end with memory consideration"""
        try:
            # Standard end condition
            if self.state.stage < self.max_stage:
                return False
            else:
                get_logger().info(f"{Fore.CYAN}🔄 [ROUND END] Max stages reached: {self.state.stage}/{self.max_stage}{Style.RESET_ALL}")
                
                # ✅ MEMORY CHECK: End round early if memory critical
                current_memory = self._get_game_memory_usage()
                if current_memory > self.game_emergency_threshold:
                    get_logger().warning(f"{Fore.RED}🚨 [MEMORY END] Ending round early due to memory pressure: {current_memory:.1f}GB{Style.RESET_ALL}")
                
                return True
                
        except Exception as e:
            get_logger().error(f"end_of_round check failed: {e}")
            # Default to not ending round on error
            return False

    # ✅ ADD: Enhanced cleanup for BaseGameManager
    def cleanup_base_game_manager(self):
        """Enhanced cleanup for BaseGameManager"""
        try:
            get_logger().info(f"{Fore.CYAN}🧹 [BASE GAME] Enhanced cleanup initiated{Style.RESET_ALL}")
            
            # Call parent cleanup
            self.cleanup_game_manager()
            
            get_logger().info(f"{Fore.GREEN}✅ [BASE GAME] Cleanup completed{Style.RESET_ALL}")
            
        except Exception as e:
            get_logger().error(f"BaseGameManager cleanup failed: {e}")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.cleanup_base_game_manager()
        except:
            pass
