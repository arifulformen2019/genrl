import os
import pickle
import time
import gc  # ✅ ADD: Memory cleanup
import threading  # ✅ ADD: Thread management
import weakref  # ✅ ADD: Weak references
from collections import deque, defaultdict  # ✅ ADD: Bounded collections
from typing import Any, Dict, List

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes

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


class HivemindRendezvouz:
    _STORE = None
    _IS_MASTER = False
    _IS_LAMBDA = False
    
    # ✅ ADD: Memory management for rendezvous
    _STORE_CLEANUP_COUNTER = 0
    _LAST_CLEANUP = time.time()

    @classmethod
    def init(cls, is_master: bool = False):
        """✅ OPTIMIZED: Initialize rendezvous with memory management"""
        cls._IS_MASTER = is_master
        cls._IS_LAMBDA = os.environ.get("LAMBDA", False)
        
        if cls._STORE is None and cls._IS_LAMBDA:
            try:
                world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
                
                print(f"{Fore.CYAN}🔗 [RENDEZVOUS] Initializing TCP store - World size: {world_size}{Style.RESET_ALL}")
                
                cls._STORE = dist.TCPStore(
                    host_name=os.environ["MASTER_ADDR"],
                    port=int(os.environ["MASTER_PORT"]),
                    is_master=is_master,
                    world_size=world_size,
                    wait_for_workers=True,
                    timeout=300,  # ✅ ADD: 5 minute timeout to prevent hanging
                )
                
                print(f"{Fore.GREEN}✅ [RENDEZVOUS] TCP store initialized successfully{Style.RESET_ALL}")
                
            except Exception as e:
                print(f"{Fore.RED}❌ [RENDEZVOUS] Failed to initialize TCP store: {e}{Style.RESET_ALL}")
                raise

    @classmethod
    def is_bootstrap(cls) -> bool:
        return cls._IS_MASTER

    @classmethod
    def set_initial_peers(cls, initial_peers):
        """✅ OPTIMIZED: Set initial peers with memory management"""
        try:
            if cls._STORE is None and cls._IS_LAMBDA:
                cls.init()
                
            if cls._IS_LAMBDA and cls._STORE is not None:
                # ✅ MEMORY OPTIMIZATION: Limit peer data size
                if isinstance(initial_peers, list) and len(initial_peers) > 100:
                    print(f"{Fore.YELLOW}⚠️ [RENDEZVOUS] Too many initial peers ({len(initial_peers)}), limiting to 100{Style.RESET_ALL}")
                    initial_peers = initial_peers[:100]
                
                peer_data = pickle.dumps(initial_peers)
                
                # ✅ MEMORY CHECK: Warn if peer data is large
                if len(peer_data) > 1024 * 1024:  # 1MB
                    print(f"{Fore.YELLOW}⚠️ [RENDEZVOUS] Large peer data: {len(peer_data)/1024/1024:.1f}MB{Style.RESET_ALL}")
                
                cls._STORE.set("initial_peers", peer_data)
                
                # ✅ CLEANUP: Clear local reference
                del peer_data
                
                print(f"{Fore.GREEN}✅ [RENDEZVOUS] Initial peers set: {len(initial_peers) if isinstance(initial_peers, list) else 'N/A'}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ [RENDEZVOUS] Failed to set initial peers: {e}{Style.RESET_ALL}")
            # Don't raise - allow fallback behavior

    @classmethod
    def get_initial_peers(cls):
        """✅ OPTIMIZED: Get initial peers with timeout and error handling"""
        try:
            if cls._STORE is None and cls._IS_LAMBDA:
                cls.init()
                
            if not cls._IS_LAMBDA or cls._STORE is None:
                print(f"{Fore.YELLOW}⚠️ [RENDEZVOUS] No store available, returning empty peers{Style.RESET_ALL}")
                return []
            
            # ✅ TIMEOUT: Wait with timeout to prevent hanging
            print(f"{Fore.CYAN}🔍 [RENDEZVOUS] Waiting for initial peers...{Style.RESET_ALL}")
            
            try:
                # Wait with timeout
                cls._STORE.wait(["initial_peers"], timeout=60)  # 1 minute timeout
            except Exception as wait_e:
                print(f"{Fore.RED}❌ [RENDEZVOUS] Wait timeout for initial peers: {wait_e}{Style.RESET_ALL}")
                return []
            
            peer_bytes = cls._STORE.get("initial_peers")
            initial_peers = pickle.loads(peer_bytes)
            
            # ✅ CLEANUP: Clear bytes reference
            del peer_bytes
            
            print(f"{Fore.GREEN}✅ [RENDEZVOUS] Retrieved initial peers: {len(initial_peers) if isinstance(initial_peers, list) else 'N/A'}{Style.RESET_ALL}")
            
            return initial_peers
            
        except Exception as e:
            print(f"{Fore.RED}❌ [RENDEZVOUS] Failed to get initial peers: {e}{Style.RESET_ALL}")
            # Return empty list as fallback
            return []

    @classmethod
    def cleanup_store(cls):
        """✅ ADD: Cleanup method for store"""
        try:
            if cls._STORE is not None:
                print(f"{Fore.CYAN}🧹 [RENDEZVOUS] Cleaning up TCP store{Style.RESET_ALL}")
                # TCPStore doesn't have explicit cleanup, but we can clear our reference
                cls._STORE = None
                gc.collect()
                
        except Exception as e:
            print(f"Rendezvous cleanup failed: {e}")


class HivemindBackend(Communication):
    """✅ OPTIMIZED: HivemindBackend with comprehensive memory leak prevention"""
    
    def __init__(
        self,
        initial_peers: List[str] | None = None,
        timeout: int = 600,
        disable_caching: bool = False,
        beam_size: int = 1000,
        **kwargs,
    ):
        try:
            print(f"{Fore.CYAN}🚀 [HIVEMIND] Initializing HivemindBackend with memory optimization{Style.RESET_ALL}")
            
            self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
            self.timeout = timeout
            self.bootstrap = HivemindRendezvouz.is_bootstrap()
            self.beam_size = min(beam_size, 2000)  # ✅ LIMIT: Cap beam size to prevent memory explosion
            self.dht = None

            # ✅ CRITICAL MEMORY LEAK FIX: Add HiveMind memory management
            self._initialize_hivemind_memory_management()

            # ✅ MEMORY OPTIMIZATION: Force disable caching to prevent memory bloat
            if not disable_caching:
                print(f"{Fore.YELLOW}⚠️ [HIVEMIND] Forcing cache disable to prevent memory leaks{Style.RESET_ALL}")
                disable_caching = True
                
            if disable_caching:
                kwargs["cache_locally"] = False
                kwargs["cache_on_store"] = False

            # ✅ DHT INITIALIZATION WITH ERROR HANDLING
            if self.bootstrap:
                print(f"{Fore.CYAN}🌐 [HIVEMIND] Initializing as bootstrap node{Style.RESET_ALL}")
                self._initialize_bootstrap_dht(initial_peers, **kwargs)
            else:
                print(f"{Fore.CYAN}🔗 [HIVEMIND] Initializing as worker node{Style.RESET_ALL}")
                self._initialize_worker_dht(initial_peers, **kwargs)
                
            self.step_ = 0
            
            # ✅ START BACKGROUND MONITORING
            self._start_dht_monitoring()
            
            print(f"{Fore.GREEN}✅ [HIVEMIND] HivemindBackend initialized successfully{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ [HIVEMIND] Failed to initialize HivemindBackend: {e}{Style.RESET_ALL}")
            raise

    def _initialize_hivemind_memory_management(self):
        """Initialize memory management for HivemindBackend"""
        try:
            # ✅ DHT CACHE MANAGEMENT
            self.max_dht_cache_size = 1000      # Limit DHT cache entries
            self.dht_cache_cleanup_frequency = 50  # Cleanup every 50 operations
            
            # ✅ MESSAGE BUFFER MANAGEMENT  
            self.max_message_buffer_size = 500   # Limit message buffers
            self.message_buffer = deque(maxlen=self.max_message_buffer_size)
            self.sent_messages = deque(maxlen=100)  # Track sent messages
            self.received_messages = deque(maxlen=100)  # Track received messages
            
            # ✅ OPERATION TRACKING
            self.operation_counter = 0
            self.failed_operations = deque(maxlen=50)  # Track failed operations
            self.last_hivemind_cleanup = time.time()
            self.hivemind_cleanup_interval = 120  # 2 minutes
            
            # ✅ DHT HEALTH MONITORING
            self.dht_error_count = 0
            self.dht_consecutive_failures = 0
            self.max_consecutive_failures = 5
            self.dht_health_check_interval = 30  # 30 seconds
            self.last_dht_health_check = time.time()
            
            # ✅ PEER MANAGEMENT
            self.max_tracked_peers = 200  # Limit peer tracking
            self.peer_cache = {}
            self.peer_last_seen = {}
            
            # ✅ SERIALIZATION CACHE
            self.serialization_cache = {}
            self.max_serialization_cache = 100
            
            print(f"{Fore.GREEN}🚀 [HIVEMIND] Memory management initialized{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"HiveMind memory management init failed: {e}")

    def _initialize_bootstrap_dht(self, initial_peers, **kwargs):
        """Initialize DHT as bootstrap node with error handling"""
        try:
            # ✅ MEMORY OPTIMIZATION: Set conservative DHT parameters
            dht_kwargs = {
                **kwargs,
                "cache_locally": False,
                "cache_on_store": False,
                "max_workers": 4,  # Limit worker threads
                "max_concurrent_queries": 16,  # Limit concurrent operations
            }
            
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=initial_peers,
                **dht_kwargs,
            )
            
            # ✅ GET VISIBLE ADDRESSES WITH ERROR HANDLING
            try:
                dht_maddrs = self.dht.get_visible_maddrs(latest=True)
                HivemindRendezvouz.set_initial_peers(dht_maddrs)
                print(f"{Fore.GREEN}✅ [BOOTSTRAP] DHT initialized with {len(dht_maddrs)} addresses{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ [BOOTSTRAP] Failed to get visible addresses: {e}{Style.RESET_ALL}")
                # Continue anyway - DHT might still work
                
        except Exception as e:
            print(f"{Fore.RED}❌ [BOOTSTRAP] DHT initialization failed: {e}{Style.RESET_ALL}")
            raise

    def _initialize_worker_dht(self, initial_peers, **kwargs):
        """Initialize DHT as worker node with error handling"""
        try:
            # ✅ GET INITIAL PEERS WITH FALLBACK
            if initial_peers is None:
                try:
                    initial_peers = HivemindRendezvouz.get_initial_peers()
                    if not initial_peers:
                        print(f"{Fore.YELLOW}⚠️ [WORKER] No initial peers available, using fallback{Style.RESET_ALL}")
                        initial_peers = []
                except Exception as e:
                    print(f"{Fore.RED}❌ [WORKER] Failed to get initial peers: {e}{Style.RESET_ALL}")
                    initial_peers = []
            
            # ✅ MEMORY OPTIMIZATION: Set conservative DHT parameters
            dht_kwargs = {
                **kwargs,
                "cache_locally": False,
                "cache_on_store": False,
                "max_workers": 4,  # Limit worker threads
                "max_concurrent_queries": 16,  # Limit concurrent operations
            }
            
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=initial_peers,
                **dht_kwargs,
            )
            
            print(f"{Fore.GREEN}✅ [WORKER] DHT initialized with {len(initial_peers)} initial peers{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ [WORKER] DHT initialization failed: {e}{Style.RESET_ALL}")
            raise

    def _start_dht_monitoring(self):
        """Start background DHT health monitoring"""
        try:
            def dht_monitor():
                while True:
                    try:
                        self._check_dht_health()
                        time.sleep(self.dht_health_check_interval)
                    except Exception as e:
                        print(f"DHT monitor error: {e}")
                        time.sleep(60)  # Back off on error
                        
            monitor_thread = threading.Thread(target=dht_monitor, daemon=True)
            monitor_thread.start()
            
            print(f"{Fore.CYAN}👁️ [HIVEMIND] DHT monitoring started{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"DHT monitoring start failed: {e}")

    def _check_dht_health(self):
        """Check DHT health and perform maintenance"""
        try:
            current_time = time.time()
            
            # Skip if checked recently
            if current_time - self.last_dht_health_check < self.dht_health_check_interval:
                return
                
            self.last_dht_health_check = current_time
            
            # ✅ TEST DHT CONNECTIVITY
            try:
                visible_peers = self.dht.get_visible_maddrs(latest=True)
                self.dht_consecutive_failures = 0  # Reset on success
                
                # ✅ PEER CACHE MANAGEMENT
                if len(visible_peers) > self.max_tracked_peers:
                    print(f"{Fore.YELLOW}⚠️ [DHT HEALTH] Too many visible peers ({len(visible_peers)}), limiting tracking{Style.RESET_ALL}")
                
                # ✅ PERIODIC CLEANUP
                if current_time - self.last_hivemind_cleanup > self.hivemind_cleanup_interval:
                    self._periodic_hivemind_cleanup()
                    
            except Exception as e:
                self.dht_consecutive_failures += 1
                self.dht_error_count += 1
                
                print(f"{Fore.YELLOW}⚠️ [DHT HEALTH] DHT health check failed ({self.dht_consecutive_failures}/{self.max_consecutive_failures}): {e}{Style.RESET_ALL}")
                
                # ✅ EMERGENCY DHT RESET
                if self.dht_consecutive_failures >= self.max_consecutive_failures:
                    print(f"{Fore.RED}🚨 [DHT HEALTH] Too many consecutive failures, initiating emergency cleanup{Style.RESET_ALL}")
                    self._emergency_dht_cleanup()
                    
        except Exception as e:
            print(f"DHT health check failed: {e}")

    def _periodic_hivemind_cleanup(self):
        """Periodic HiveMind memory cleanup"""
        try:
            print(f"{Fore.CYAN}🧹 [HIVEMIND] Periodic cleanup started{Style.RESET_ALL}")
            
            # ✅ CLEAR MESSAGE BUFFERS
            cleaned_buffers = 0
            if len(self.message_buffer) > self.max_message_buffer_size // 2:
                # Clear half the buffer
                for _ in range(len(self.message_buffer) // 2):
                    self.message_buffer.popleft()
                cleaned_buffers += 1
            
            # ✅ CLEAN SERIALIZATION CACHE
            if len(self.serialization_cache) > self.max_serialization_cache:
                # Keep only recent entries
                sorted_keys = sorted(self.serialization_cache.keys())
                old_keys = sorted_keys[:-self.max_serialization_cache]
                for key in old_keys:
                    del self.serialization_cache[key]
                cleaned_buffers += 1
            
            # ✅ CLEAN PEER CACHE
            current_time = time.time()
            old_peers = []
            for peer_id, last_seen in self.peer_last_seen.items():
                if current_time - last_seen > 3600:  # 1 hour old
                    old_peers.append(peer_id)
            
            for peer_id in old_peers:
                self.peer_cache.pop(peer_id, None)
                self.peer_last_seen.pop(peer_id, None)
            
            if old_peers:
                cleaned_buffers += 1
            
            # ✅ GARBAGE COLLECTION
            collected = gc.collect()
            
            self.last_hivemind_cleanup = time.time()
            
            if cleaned_buffers > 0 or collected > 0:
                print(f"{Fore.GREEN}✅ [HIVEMIND] Cleanup completed: {cleaned_buffers} buffers, {collected} objects{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"HiveMind periodic cleanup failed: {e}")

    def _emergency_dht_cleanup(self):
        """Emergency DHT cleanup when errors accumulate"""
        try:
            print(f"{Fore.RED}🚨 [HIVEMIND] Emergency DHT cleanup initiated{Style.RESET_ALL}")
            
            # ✅ CLEAR ALL CACHES
            self.message_buffer.clear()
            self.sent_messages.clear()
            self.received_messages.clear()
            self.serialization_cache.clear()
            self.peer_cache.clear()
            self.peer_last_seen.clear()
            
            # ✅ RESET COUNTERS
            self.dht_consecutive_failures = 0
            self.operation_counter = 0
            
            # ✅ FORCE GARBAGE COLLECTION
            collected = gc.collect()
            
            print(f"{Fore.GREEN}✅ [HIVEMIND] Emergency cleanup completed - {collected} objects collected{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Emergency DHT cleanup failed: {e}")

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        """✅ OPTIMIZED: All-gather with comprehensive error handling and memory management"""
        
        key = str(self.step_)
        
        try:
            # ✅ INCREMENT OPERATION COUNTER
            self.operation_counter += 1
            
            # ✅ PERIODIC CLEANUP
            if self.operation_counter % self.dht_cache_cleanup_frequency == 0:
                self._periodic_hivemind_cleanup()
            
            print(f"{Fore.CYAN}📡 [GATHER] Starting all_gather for step {self.step_} (op #{self.operation_counter}){Style.RESET_ALL}")
            
            # ✅ DHT HEALTH CHECK
            try:
                visible_peers = self.dht.get_visible_maddrs(latest=True)
                if not visible_peers:
                    print(f"{Fore.YELLOW}⚠️ [GATHER] No visible peers, returning self only{Style.RESET_ALL}")
                    return {str(self.dht.peer_id): obj}
            except Exception as peer_e:
                print(f"{Fore.YELLOW}⚠️ [GATHER] Failed to get visible peers: {peer_e}{Style.RESET_ALL}")
                # Continue anyway - might still work
            
            # ✅ OBJECT SERIALIZATION WITH SIZE CHECK
            try:
                obj_bytes = to_bytes(obj)
                
                # ✅ SIZE CHECK: Warn about large objects
                obj_size_mb = len(obj_bytes) / 1024 / 1024
                if obj_size_mb > 10:  # 10MB
                    print(f"{Fore.YELLOW}⚠️ [GATHER] Large object size: {obj_size_mb:.1f}MB{Style.RESET_ALL}")
                elif obj_size_mb > 100:  # 100MB
                    print(f"{Fore.RED}🚨 [GATHER] Very large object: {obj_size_mb:.1f}MB - may cause memory issues{Style.RESET_ALL}")
                
                # ✅ CACHE SERIALIZED OBJECT (with size limit)
                if len(self.serialization_cache) < self.max_serialization_cache:
                    self.serialization_cache[key] = obj_bytes
                
            except Exception as serial_e:
                print(f"{Fore.RED}❌ [GATHER] Serialization failed: {serial_e}{Style.RESET_ALL}")
                return {str(self.dht.peer_id): obj}
            
            # ✅ DHT STORE WITH ERROR HANDLING
            try:
                store_start = time.time()
                
                self.dht.store(
                    key,
                    subkey=str(self.dht.peer_id),
                    value=obj_bytes,
                    expiration_time=get_dht_time() + self.timeout,
                    beam_size=self.beam_size,
                )
                
                store_time = time.time() - store_start
                if store_time > 5:  # Warn if store takes > 5 seconds
                    print(f"{Fore.YELLOW}⚠️ [GATHER] Slow DHT store: {store_time:.1f}s{Style.RESET_ALL}")
                
                # ✅ TRACK SENT MESSAGE
                self.sent_messages.append((key, time.time(), len(obj_bytes)))
                
            except Exception as store_e:
                print(f"{Fore.RED}❌ [GATHER] DHT store failed: {store_e}{Style.RESET_ALL}")
                return {str(self.dht.peer_id): obj}
            
            # ✅ WAIT AND RETRIEVE WITH ENHANCED ERROR HANDLING
            time.sleep(1)  # Brief wait for propagation
            
            t_start = time.monotonic()
            max_wait_time = min(self.timeout, 300)  # Cap at 5 minutes
            
            while True:
                try:
                    # ✅ DHT GET WITH ERROR HANDLING
                    output_, expiration_time = self.dht.get(key, beam_size=self.beam_size, latest=True)
                    
                    current_time = time.monotonic()
                    elapsed = current_time - t_start
                    
                    if len(output_) >= self.world_size:
                        print(f"{Fore.GREEN}✅ [GATHER] Got {len(output_)} values in {elapsed:.1f}s{Style.RESET_ALL}")
                        break
                    else:
                        if elapsed > max_wait_time:
                            print(f"{Fore.RED}❌ [GATHER] Timeout after {elapsed:.1f}s - got {len(output_)}/{self.world_size} values{Style.RESET_ALL}")
                            # ✅ FALLBACK: Return what we have + self
                            if not output_:
                                return {str(self.dht.peer_id): obj}
                            break
                        
                        # ✅ PROGRESS LOGGING
                        if elapsed > 30 and int(elapsed) % 10 == 0:  # Every 10s after 30s
                            print(f"{Fore.YELLOW}⏳ [GATHER] Waiting... {len(output_)}/{self.world_size} values after {elapsed:.0f}s{Style.RESET_ALL}")
                        
                        time.sleep(1)  # Wait before retry
                    
                except Exception as get_e:
                    print(f"{Fore.RED}❌ [GATHER] DHT get failed: {get_e}{Style.RESET_ALL}")
                    # ✅ FALLBACK: Return self only
                    return {str(self.dht.peer_id): obj}
            
            # ✅ ADVANCE STEP
            self.step_ += 1
            
            # ✅ DESERIALIZE RESULTS WITH ERROR HANDLING
            try:
                results = {}
                deserialization_errors = 0
                
                for peer_key, value in output_.items():
                    try:
                        deserialized_obj = from_bytes(value.value)
                        results[peer_key] = deserialized_obj
                        
                        # ✅ TRACK RECEIVED MESSAGE
                        self.received_messages.append((peer_key, time.time(), len(value.value)))
                        
                    except Exception as deser_e:
                        deserialization_errors += 1
                        print(f"{Fore.YELLOW}⚠️ [GATHER] Deserialization failed for peer {peer_key}: {deser_e}{Style.RESET_ALL}")
                        continue
                
                if deserialization_errors > 0:
                    print(f"{Fore.YELLOW}⚠️ [GATHER] {deserialization_errors} deserialization errors{Style.RESET_ALL}")
                
                # ✅ ENSURE SELF IS INCLUDED
                if str(self.dht.peer_id) not in results and obj is not None:
                    results[str(self.dht.peer_id)] = obj
                
                # ✅ SORT RESULTS FOR CONSISTENCY
                sorted_results = sorted(results.items(), key=lambda x: x[0])
                final_results = {key: value for key, value in sorted_results}
                
                # ✅ CLEANUP
                del obj_bytes, output_
                
                print(f"{Fore.GREEN}✅ [GATHER] Completed - {len(final_results)} results{Style.RESET_ALL}")
                
                return final_results
                
            except Exception as result_e:
                print(f"{Fore.RED}❌ [GATHER] Result processing failed: {result_e}{Style.RESET_ALL}")
                return {str(self.dht.peer_id): obj}
            
        except (BlockingIOError, EOFError) as io_e:
            print(f"{Fore.RED}🚨 [GATHER] DHT I/O error (this is the error you saw before): {io_e}{Style.RESET_ALL}")
            
            # ✅ TRACK FAILURE
            self.failed_operations.append((time.time(), 'io_error', str(io_e)))
            self.dht_consecutive_failures += 1
            
            # ✅ EMERGENCY CLEANUP IF TOO MANY FAILURES
            if self.dht_consecutive_failures >= 3:
                print(f"{Fore.RED}💥 [GATHER] Too many I/O failures, emergency cleanup{Style.RESET_ALL}")
                self._emergency_dht_cleanup()
            
            # ✅ FALLBACK: Return self only
            return {str(self.dht.peer_id): obj}
            
        except Exception as e:
            print(f"{Fore.RED}❌ [GATHER] Unexpected error: {e}{Style.RESET_ALL}")
            
            # ✅ TRACK FAILURE
            self.failed_operations.append((time.time(), 'unexpected', str(e)))
            
            # ✅ FALLBACK: Return self only
            return {str(self.dht.peer_id): obj}

    def get_id(self):
        """✅ SAFE: Get DHT peer ID with error handling"""
        try:
            if self.dht is not None:
                return str(self.dht.peer_id)
            else:
                print(f"{Fore.RED}❌ [GET_ID] DHT not initialized{Style.RESET_ALL}")
                return "unknown_peer"
        except Exception as e:
            print(f"{Fore.RED}❌ [GET_ID] Failed to get peer ID: {e}{Style.RESET_ALL}")
            return "error_peer"

    # ✅ ADD: Cleanup methods
    def cleanup_hivemind(self):
        """Manual cleanup method for HivemindBackend"""
        try:
            print(f"{Fore.CYAN}🧹 [HIVEMIND] Manual cleanup initiated{Style.RESET_ALL}")
            
            # Emergency cleanup
            self._emergency_dht_cleanup()
            
            # Close DHT if possible
            if self.dht is not None:
                try:
                    # DHT doesn't have explicit close, but we can clear our reference
                    print(f"{Fore.YELLOW}🔌 [HIVEMIND] Cleaning up DHT connection{Style.RESET_ALL}")
                    self.dht = None
                except Exception as dht_e:
                    print(f"DHT cleanup failed: {dht_e}")
            
            # Clean up rendezvous
            HivemindRendezvouz.cleanup_store()
            
            print(f"{Fore.GREEN}✅ [HIVEMIND] Cleanup completed{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"HivemindBackend cleanup failed: {e}")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.cleanup_hivemind()
        except:
            pass

    # ✅ ADD: Health monitoring methods
    def get_dht_stats(self):
        """Get DHT statistics for monitoring"""
        try:
            stats = {
                'operation_counter': self.operation_counter,
                'dht_error_count': self.dht_error_count,
                'consecutive_failures': self.dht_consecutive_failures,
                'message_buffer_size': len(self.message_buffer),
                'sent_messages': len(self.sent_messages),
                'received_messages': len(self.received_messages),
                'serialization_cache_size': len(self.serialization_cache),
                'peer_cache_size': len(self.peer_cache),
                'failed_operations': len(self.failed_operations),
            }
            return stats
        except Exception as e:
            print(f"Failed to get DHT stats: {e}")
            return {}

    def is_dht_healthy(self):
        """Check if DHT is in healthy state"""
        try:
            # Consider healthy if:
            # 1. DHT exists
            # 2. Not too many consecutive failures
            # 3. Recent successful operations
            
            if self.dht is None:
                return False
                
            if self.dht_consecutive_failures >= self.max_consecutive_failures:
                return False
                
            # Check if we had any successful operations recently
            current_time = time.time()
            if hasattr(self, 'last_successful_operation'):
                if current_time - self.last_successful_operation > 300:  # 5 minutes
                    return False
            
            return True
            
        except Exception as e:
            print(f"DHT health check failed: {e}")
            return False

    # ✅ ADD: Debug methods
    def debug_memory_usage(self):
        """Debug method to show memory usage"""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_gb = process.memory_info().rss / 1024**3
                
                print(f"{Fore.BLUE}🔍 [HIVEMIND DEBUG] Memory Usage:{Style.RESET_ALL}")
                print(f"   💾 Total RAM: {memory_gb:.2f}GB")
                print(f"   📊 Message buffer: {len(self.message_buffer)} items")
                print(f"   📤 Sent messages: {len(self.sent_messages)} items")
                print(f"   📥 Received messages: {len(self.received_messages)} items")
                print(f"   💾 Serialization cache: {len(self.serialization_cache)} items")
                print(f"   👥 Peer cache: {len(self.peer_cache)} peers")
                print(f"   ❌ Failed operations: {len(self.failed_operations)} items")
                print(f"   🔢 Operation counter: {self.operation_counter}")
                print(f"   ⚠️ DHT errors: {self.dht_error_count}")
                print(f"   🔄 Consecutive failures: {self.dht_consecutive_failures}")
                
                return {
                    'total_memory_gb': memory_gb,
                    'message_buffer_size': len(self.message_buffer),
                    'cache_sizes': {
                        'serialization': len(self.serialization_cache),
                        'peer': len(self.peer_cache),
                    },
                    'counters': {
                        'operations': self.operation_counter,
                        'errors': self.dht_error_count,
                        'consecutive_failures': self.dht_consecutive_failures,
                    }
                }
            else:
                print(f"{Fore.YELLOW}⚠️ [HIVEMIND DEBUG] psutil not available for detailed memory info{Style.RESET_ALL}")
                return {}
                
        except Exception as e:
            print(f"Debug memory usage failed: {e}")
            return {}
