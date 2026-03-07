#!/usr/bin/env python3
"""
Example Custom Hashing Plugin
Demonstrates how to create a custom hashing algorithm
"""
import hashlib
import os
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from plugin_system import HashingPlugin

class ExampleHashPlugin(HashingPlugin):
    @property
    def name(self) -> str:
        return "Example Custom Hash"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    @property
    def description(self) -> str:
        return "Example custom hashing algorithm for demonstration"
        
    @property
    def author(self) -> str:
        return "Plugin Developer"
        
    @property
    def supported_extensions(self) -> List[str]:
        return ['.txt', '.py', '.js', '.html', '.css']
        
    def compute_hash(self, file_path: str) -> str:
        """Custom hash combining file size and content sample"""
        try:
            file_size = os.path.getsize(file_path)
            
            # Read first and last 1024 bytes
            with open(file_path, 'rb') as f:
                start_bytes = f.read(1024)
                f.seek(-min(1024, file_size), 2)
                end_bytes = f.read(1024)
                
            # Combine size and content for hash
            hash_input = f"{file_size}_{len(start_bytes)}_{len(end_bytes)}"
            combined_bytes = hash_input.encode() + start_bytes + end_bytes
            
            return hashlib.sha256(combined_bytes).hexdigest()
            
        except Exception:
            return ""
            
    def compare_hashes(self, hash1: str, hash2: str) -> float:
        """Simple exact match comparison"""
        if not hash1 or not hash2:
            return 0.0
        return 1.0 if hash1 == hash2 else 0.0

# Plugin entry point
def create_plugin():
    return ExampleHashPlugin()
