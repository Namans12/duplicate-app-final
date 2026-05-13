#!/usr/bin/env python3
"""
Memory Optimization and Streaming Module
Handles large files efficiently without loading entirely into memory
"""
import os
import hashlib
import mmap
from typing import Generator, Optional, Tuple, Any, Dict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
logger = logging.getLogger(__name__)



class StreamingHasher:
    """Efficient hashing for large files using streaming and memory mapping"""
    
    def __init__(self, chunk_size: int = 8192, max_memory_mb: int = 100):
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory_usage = 0
        self.memory_lock = threading.Lock()
        
    def stream_hash_md5(self, file_path: str) -> str:
        """Compute MD5 hash using streaming for large files"""
        try:
            file_size = os.path.getsize(file_path)
            
            # Use memory mapping for very large files
            if file_size > self.max_memory_usage:
                return self._memory_mapped_hash(file_path, hashlib.md5())
            else:
                return self._streaming_hash(file_path, hashlib.md5())
                
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
            return ""
            
    def stream_hash_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash using streaming for large files"""
        try:
            file_size = os.path.getsize(file_path)
            
            if file_size > self.max_memory_usage:
                return self._memory_mapped_hash(file_path, hashlib.sha256())
            else:
                return self._streaming_hash(file_path, hashlib.sha256())
                
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
            return ""
            
    def _streaming_hash(self, file_path: str, hasher) -> str:
        """Hash file using streaming chunks"""
        try:
            with open(file_path, 'rb') as f:
                chunk_idx = 0
                yield_every = 256  # yield once every ~2MB (256 * 8KB) instead of every chunk
                on_worker = threading.current_thread() != threading.main_thread()
                while chunk := f.read(self.chunk_size):
                    hasher.update(chunk)
                    chunk_idx += 1
                    # Yield control periodically (worker threads only) to keep
                    # the UI responsive without paying a sleep on every read.
                    if on_worker and chunk_idx % yield_every == 0:
                        time.sleep(0.0001)

            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Streaming hash error for {file_path}: {e}")
            return ""
            
    def _memory_mapped_hash(self, file_path: str, hasher) -> str:
        """Hash file using memory mapping for very large files"""
        try:
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Process in chunks to avoid loading entire file
                    for i in range(0, len(mm), self.chunk_size):
                        chunk_end = min(i + self.chunk_size, len(mm))
                        hasher.update(mm[i:chunk_end])
                        
                        # Yield control periodically
                        if i % (self.chunk_size * 100) == 0:  # Every 100 chunks
                            time.sleep(0.001)
                            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Memory mapped hash error for {file_path}: {e}")
            return ""
            
    def sample_hash(self, file_path: str, sample_size: int = 64 * 1024) -> str:
        """Create hash from file samples for very large files"""
        try:
            file_size = os.path.getsize(file_path)
            
            if file_size <= sample_size * 3:
                # Small file, hash entirely
                return self.stream_hash_md5(file_path)
                
            hasher = hashlib.md5()
            
            with open(file_path, 'rb') as f:
                # Hash beginning
                start_sample = f.read(sample_size)
                hasher.update(start_sample)
                
                # Hash middle
                if file_size > sample_size * 2:
                    f.seek(file_size // 2 - sample_size // 2)
                    middle_sample = f.read(sample_size)
                    hasher.update(middle_sample)
                
                # Hash end
                f.seek(-sample_size, 2)
                end_sample = f.read(sample_size)
                hasher.update(end_sample)
                
                # Include file size in hash for uniqueness
                hasher.update(str(file_size).encode())
                
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Sample hash error for {file_path}: {e}")
            return ""

class MemoryManager:
    """Manages memory usage during large-scale operations
    
    Thread-safe: All dictionary accesses are protected by usage_lock.
    """
    
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_usage = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.usage_lock = threading.RLock()  # RLock to allow re-entry from allocate_memory -> _cleanup_old_cache
        self.cached_data: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.cache_sizes: Dict[str, int] = {}
        
    def allocate_memory(self, size_bytes: int, identifier: Optional[str] = None) -> bool:
        """Try to allocate memory, return False if would exceed limit"""
        with self.usage_lock:
            if self.current_usage + size_bytes > self.max_memory_usage:
                # Try to free some memory first
                self._cleanup_old_cache_unlocked()
                
                if self.current_usage + size_bytes > self.max_memory_usage:
                    return False
                    
            self.current_usage += size_bytes
            if identifier:
                self.access_times[identifier] = time.time()
            return True
            
    def free_memory(self, size_bytes: int, identifier: Optional[str] = None):
        """Free allocated memory"""
        with self.usage_lock:
            self.current_usage = max(0, self.current_usage - size_bytes)
            if identifier and identifier in self.cached_data:
                del self.cached_data[identifier]
                if identifier in self.access_times:
                    del self.access_times[identifier]
                if identifier in self.cache_sizes:
                    del self.cache_sizes[identifier]
                
    def cache_data(self, key: str, data: Any, size_bytes: int) -> bool:
        """Cache data if memory allows (thread-safe)"""
        with self.usage_lock:
            if self.current_usage + size_bytes > self.max_memory_usage:
                self._cleanup_old_cache_unlocked()
                if self.current_usage + size_bytes > self.max_memory_usage:
                    return False
            
            self.current_usage += size_bytes
            self.cached_data[key] = data
            self.access_times[key] = time.time()
            self.cache_sizes[key] = size_bytes
            return True
        
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data and update access time (thread-safe)"""
        with self.usage_lock:
            if key in self.cached_data:
                self.access_times[key] = time.time()
                return self.cached_data[key]
            return None
        
    def _cleanup_old_cache_unlocked(self):
        """Remove oldest cached items to free memory (must hold usage_lock)"""
        if not self.access_times:
            return
            
        # Sort by access time and remove oldest 25%
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        items_to_remove = max(1, len(sorted_items) // 4)
        
        # Create list of keys to remove to avoid modifying dict during iteration
        keys_to_remove = [key for key, _ in sorted_items[:items_to_remove]]
        
        for key in keys_to_remove:
            if key in self.cached_data:
                self.current_usage = max(0, self.current_usage - self.cache_sizes.get(key, 0))
                del self.cached_data[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.cache_sizes:
                del self.cache_sizes[key]
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        with self.usage_lock:
            return {
                "current_usage_mb": self.current_usage / (1024 * 1024),
                "max_usage_mb": self.max_memory_usage / (1024 * 1024),
                "usage_percent": (self.current_usage / self.max_memory_usage) * 100,
                "cached_items": len(self.cached_data),
                "available_mb": (self.max_memory_usage - self.current_usage) / (1024 * 1024)
            }

class IncrementalScanner:
    """Handles incremental scanning to only process new/modified files"""
    
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.streaming_hasher = StreamingHasher()
        self.memory_manager = MemoryManager()
        
    def scan_directory_incremental(self, directory: str, last_scan_time: float = 0) -> Tuple[list, list, list]:
        """
        Scan directory incrementally, returning new, modified, and unchanged files
        Returns: (new_files, modified_files, unchanged_files)
        """
        new_files = []
        modified_files = []
        unchanged_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    if not os.path.exists(file_path):
                        continue
                        
                    stat_info = os.stat(file_path)
                    modification_time = stat_info.st_mtime
                    file_size = stat_info.st_size
                    
                    # Check if file is new or modified
                    if modification_time > last_scan_time:
                        # Check if we have cached info
                        if self.cache_manager:
                            cached_info = self.cache_manager.get_cached_file_info(file_path)
                            
                            if cached_info and cached_info['modification_time'] == modification_time:
                                unchanged_files.append(file_path)
                            else:
                                if cached_info:
                                    modified_files.append(file_path)
                                else:
                                    new_files.append(file_path)
                        else:
                            new_files.append(file_path)
                    else:
                        unchanged_files.append(file_path)
                        
                except Exception as e:
                    logger.error(f"Error checking {file_path}: {e}")
                    
        return new_files, modified_files, unchanged_files
        
    def process_files_efficiently(self, file_paths: list, algorithm: str = "md5") -> Dict[str, str]:
        """Process files efficiently based on size and type"""
        results = {}
        
        # Group files by size for efficient processing
        size_groups = self._group_files_by_size(file_paths)
        
        # Process small files first (they're fast)
        for size_category, files in size_groups.items():
            if size_category == "small":
                results.update(self._process_small_files(files, algorithm))
            elif size_category == "medium":
                results.update(self._process_medium_files(files, algorithm))
            elif size_category == "large":
                results.update(self._process_large_files(files, algorithm))
                
        return results
        
    def _group_files_by_size(self, file_paths: list) -> Dict[str, list]:
        """Group files by size categories for efficient processing"""
        small_files = []    # < 10MB
        medium_files = []   # 10MB - 100MB
        large_files = []    # > 100MB
        
        for file_path in file_paths:
            try:
                file_size = os.path.getsize(file_path)
                
                if file_size < 10 * 1024 * 1024:  # 10MB
                    small_files.append(file_path)
                elif file_size < 100 * 1024 * 1024:  # 100MB
                    medium_files.append(file_path)
                else:
                    large_files.append(file_path)
                    
            except Exception:
                small_files.append(file_path)  # Default to small if can't determine size
                
        return {
            "small": small_files,
            "medium": medium_files,
            "large": large_files
        }
        
    def _process_small_files(self, files: list, algorithm: str) -> Dict[str, str]:
        """Process small files with standard hashing"""
        results = {}
        
        for file_path in files:
            try:
                if algorithm.lower() == "md5":
                    hash_value = self.streaming_hasher.stream_hash_md5(file_path)
                elif algorithm.lower() == "sha256":
                    hash_value = self.streaming_hasher.stream_hash_sha256(file_path)
                else:
                    hash_value = self.streaming_hasher.stream_hash_md5(file_path)
                    
                if hash_value:
                    results[file_path] = hash_value
                    
            except Exception as e:
                logger.error(f"Error processing small file {file_path}: {e}")
                
        return results
        
    def _process_medium_files(self, files: list, algorithm: str) -> Dict[str, str]:
        """Process medium files with threading"""
        results = {}
        
        def process_file(file_path):
            try:
                if algorithm.lower() == "md5":
                    return file_path, self.streaming_hasher.stream_hash_md5(file_path)
                elif algorithm.lower() == "sha256":
                    return file_path, self.streaming_hasher.stream_hash_sha256(file_path)
                else:
                    return file_path, self.streaming_hasher.stream_hash_md5(file_path)
            except Exception as e:
                logger.error(f"Error processing medium file {file_path}: {e}")
                return file_path, ""
                
        # Use thread pool for medium files
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_file, fp) for fp in files]
            
            for future in futures:
                try:
                    file_path, hash_value = future.result(timeout=30)
                    if hash_value:
                        results[file_path] = hash_value
                except Exception as e:
                    logger.error(f"Thread processing error: {e}")
                    
        return results
        
    def _process_large_files(self, files: list, algorithm: str) -> Dict[str, str]:
        """Process large files with sample hashing for speed"""
        results = {}
        
        for file_path in files:
            try:
                # For very large files, use sample hashing
                file_size = os.path.getsize(file_path)
                
                if file_size > 500 * 1024 * 1024:  # 500MB
                    # Use sample hash for very large files
                    hash_value = self.streaming_hasher.sample_hash(file_path)
                else:
                    # Use streaming hash for large but manageable files
                    if algorithm.lower() == "md5":
                        hash_value = self.streaming_hasher.stream_hash_md5(file_path)
                    elif algorithm.lower() == "sha256":
                        hash_value = self.streaming_hasher.stream_hash_sha256(file_path)
                    else:
                        hash_value = self.streaming_hasher.stream_hash_md5(file_path)
                        
                if hash_value:
                    results[file_path] = hash_value
                    
                # Report progress for large files
                logger.info(f"Processed large file: {os.path.basename(file_path)} ({file_size / (1024*1024):.1f} MB)")
                
            except Exception as e:
                logger.error(f"Error processing large file {file_path}: {e}")
                
        return results
        
    def estimate_processing_time(self, file_paths: list) -> Dict[str, float]:
        """Estimate processing time based on file sizes and types"""
        size_groups = self._group_files_by_size(file_paths)
        
        # Estimated processing rates (files per second)
        rates = {
            "small": 50,    # 50 small files per second
            "medium": 5,    # 5 medium files per second  
            "large": 0.5    # 0.5 large files per second
        }
        
        estimates = {}
        total_time = 0
        
        for category, files in size_groups.items():
            if files:
                category_time = len(files) / rates[category]
                estimates[category] = {
                    "files": len(files),
                    "estimated_seconds": category_time
                }
                total_time += category_time
                
        estimates["total_estimated_seconds"] = total_time
        estimates["estimated_minutes"] = total_time / 60
        
        return estimates

# Lazy singletons: avoid eager construction at import time. Existing callers
# that do ``from memory_optimizer import streaming_hasher`` still work via
# PEP 562 module ``__getattr__`` below.
_streaming_hasher_instance = None
_memory_manager_instance = None
_incremental_scanner_instance = None


def get_streaming_hasher() -> 'StreamingHasher':
    global _streaming_hasher_instance
    if _streaming_hasher_instance is None:
        _streaming_hasher_instance = StreamingHasher()
    return _streaming_hasher_instance


def get_memory_manager() -> 'MemoryManager':
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance


def get_incremental_scanner() -> 'IncrementalScanner':
    global _incremental_scanner_instance
    if _incremental_scanner_instance is None:
        _incremental_scanner_instance = IncrementalScanner()
    return _incremental_scanner_instance


def __getattr__(name):
    if name == 'streaming_hasher':
        return get_streaming_hasher()
    if name == 'memory_manager':
        return get_memory_manager()
    if name == 'incremental_scanner':
        return get_incremental_scanner()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
