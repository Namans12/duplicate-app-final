#!/usr/bin/env python3
"""
Example File Type Plugin
Demonstrates custom file type support
"""
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from plugin_system import FileTypePlugin

class JSONFilePlugin(FileTypePlugin):
    @property
    def name(self) -> str:
        return "JSON File Analyzer"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    @property
    def description(self) -> str:
        return "Analyzes JSON files for structural similarity"
        
    @property
    def author(self) -> str:
        return "Plugin Developer"
        
    @property
    def supported_extensions(self) -> List[str]:
        return ['.json', '.jsonl']
        
    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """Extract structural features from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            features = {
                "keys": self._extract_keys(data),
                "value_types": self._extract_value_types(data),
                "structure_depth": self._get_depth(data),
                "array_lengths": self._get_array_lengths(data)
            }
            
            return features
            
        except Exception:
            return {}
            
    def _extract_keys(self, data, prefix="") -> List[str]:
        """Extract all keys from nested JSON"""
        keys = []
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                keys.extend(self._extract_keys(value, full_key))
        elif isinstance(data, list) and data:
            keys.extend(self._extract_keys(data[0], f"{prefix}[0]"))
        return keys
        
    def _extract_value_types(self, data) -> Dict[str, int]:
        """Count value types in JSON"""
        type_counts = {}
        if isinstance(data, dict):
            for value in data.values():
                type_name = type(value).__name__
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                nested_counts = self._extract_value_types(value)
                for k, v in nested_counts.items():
                    type_counts[k] = type_counts.get(k, 0) + v
        elif isinstance(data, list):
            for item in data:
                nested_counts = self._extract_value_types(item)
                for k, v in nested_counts.items():
                    type_counts[k] = type_counts.get(k, 0) + v
        return type_counts
        
    def _get_depth(self, data) -> int:
        """Get maximum nesting depth"""
        if isinstance(data, dict):
            return 1 + max((self._get_depth(v) for v in data.values()), default=0)
        elif isinstance(data, list):
            return 1 + max((self._get_depth(item) for item in data), default=0)
        return 0
        
    def _get_array_lengths(self, data) -> List[int]:
        """Get all array lengths in structure"""
        lengths = []
        if isinstance(data, dict):
            for value in data.values():
                lengths.extend(self._get_array_lengths(value))
        elif isinstance(data, list):
            lengths.append(len(data))
            for item in data:
                lengths.extend(self._get_array_lengths(item))
        return lengths
        
    def compare_features(self, features1: Dict, features2: Dict) -> float:
        """Compare JSON structural features"""
        if not features1 or not features2:
            return 0.0
            
        # Compare key similarity
        keys1 = set(features1.get("keys", []))
        keys2 = set(features2.get("keys", []))
        key_similarity = len(keys1 & keys2) / max(len(keys1 | keys2), 1)
        
        # Compare structure depth
        depth1 = features1.get("structure_depth", 0)
        depth2 = features2.get("structure_depth", 0)
        depth_similarity = 1.0 - abs(depth1 - depth2) / max(depth1, depth2, 1)
        
        # Compare value types
        types1 = features1.get("value_types", {})
        types2 = features2.get("value_types", {})
        all_types = set(types1.keys()) | set(types2.keys())
        type_similarity = 0.0
        if all_types:
            type_matches = sum(min(types1.get(t, 0), types2.get(t, 0)) for t in all_types)
            type_total = sum(max(types1.get(t, 0), types2.get(t, 0)) for t in all_types)
            type_similarity = type_matches / max(type_total, 1)
            
        # Weighted average
        return (key_similarity * 0.5 + depth_similarity * 0.2 + type_similarity * 0.3)
        
    def get_preview_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get JSON preview data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return {
                "type": "json",
                "keys": list(data.keys()) if isinstance(data, dict) else ["[array]"],
                "size": len(str(data)),
                "preview": json.dumps(data, indent=2)[:500] + "..." if len(str(data)) > 500 else json.dumps(data, indent=2)
            }
        except Exception:
            return None

def create_plugin():
    return JSONFilePlugin()
