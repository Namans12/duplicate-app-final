#!/usr/bin/env python3
"""
Plugin System for Custom Algorithms and File Type Support
Extensible architecture for user-created plugins
"""
import hashlib
import os
import sys
import importlib.util
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
import json
from pathlib import Path
import logging

# Module logger
logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
        
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass
        
    @property
    @abstractmethod
    def author(self) -> str:
        """Plugin author"""
        pass

class HashingPlugin(PluginInterface):
    """Base class for custom hashing algorithms"""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions"""
        pass
        
    @abstractmethod
    def compute_hash(self, file_path: str) -> str:
        """Compute hash for a file"""
        pass
        
    @abstractmethod
    def compare_hashes(self, hash1: str, hash2: str) -> float:
        """Compare two hashes and return similarity (0.0-1.0)"""
        pass
        
    def validate_file(self, file_path: str) -> bool:
        """Validate if file is supported by this plugin"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions
        
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file (optional)"""
        return {}

class FileTypePlugin(PluginInterface):
    """Base class for custom file type support"""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions"""
        pass
        
    @abstractmethod
    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """Extract features from file for comparison"""
        pass
        
    @abstractmethod
    def compare_features(self, features1: Dict, features2: Dict) -> float:
        """Compare features and return similarity (0.0-1.0)"""
        pass
        
    def get_preview_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get preview data for UI display (optional)"""
        return None

class SimilarityPlugin(PluginInterface):
    """Base class for custom similarity algorithms"""
    
    @abstractmethod
    def calculate_similarity(self, file1: str, file2: str) -> float:
        """Calculate similarity between two files"""
        pass
        
    @abstractmethod
    def supports_file_type(self, file_path: str) -> bool:
        """Check if plugin supports this file type"""
        pass
        
    def batch_calculate_similarity(self, file_pairs: List[tuple]) -> List[float]:
        """Calculate similarity for multiple file pairs (optional optimization)"""
        return [self.calculate_similarity(f1, f2) for f1, f2 in file_pairs]

class PluginManager:
    """Manages loading and execution of plugins"""
    
    def __init__(self, plugin_directory: Optional[str] = None):
        # Resolve the plugin directory relative to this module so the manager
        # works regardless of the caller's current working directory.
        if plugin_directory is None:
            self.plugin_directory = Path(__file__).resolve().parent / "plugins"
        else:
            candidate = Path(plugin_directory)
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parent / candidate
            self.plugin_directory = candidate
        self.plugin_directory.mkdir(exist_ok=True)

        self.hashing_plugins: Dict[str, HashingPlugin] = {}
        self.file_type_plugins: Dict[str, FileTypePlugin] = {}
        self.similarity_plugins: Dict[str, SimilarityPlugin] = {}

        self.plugin_configs: Dict[str, Dict] = {}
        self.plugin_errors: List[Dict] = []

        # Security gate: load only allowlisted plugin filenames unless the
        # user has explicitly disabled the check in plugin_config.json.
        self.security_check: bool = True
        self.enabled_plugins: set = set()

        self._create_plugin_structure()
        self._load_security_config()
        self.load_all_plugins()

    def _load_security_config(self):
        """Read plugin_config.json and populate the security allowlist."""
        config_file = self.plugin_directory / "plugin_config.json"
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            # No config -> conservative defaults: security on, empty allowlist.
            self.security_check = True
            self.enabled_plugins = set()
            return
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not parse plugin_config.json ({e}); enforcing security defaults")
            self.security_check = True
            self.enabled_plugins = set()
            return

        # Default to True if the key is missing.
        self.security_check = bool(config.get("security_check", True))
        self.enabled_plugins = set(config.get("enabled_plugins", []) or [])
        self.plugin_configs = config.get("plugin_settings", {}) or {}
        
    def _create_plugin_structure(self):
        """Create plugin directory structure"""
        # Create subdirectories for different plugin types
        (self.plugin_directory / "hashing").mkdir(exist_ok=True)
        (self.plugin_directory / "file_types").mkdir(exist_ok=True)
        (self.plugin_directory / "similarity").mkdir(exist_ok=True)
        (self.plugin_directory / "examples").mkdir(exist_ok=True)
        
        # Create example plugins
        self._create_example_plugins()
        
        # Create plugin configuration file
        config_file = self.plugin_directory / "plugin_config.json"
        if not config_file.exists():
            default_config = {
                "enabled_plugins": [],
                "plugin_settings": {},
                "auto_load": True,
                "security_check": True
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                
    def _create_example_plugins(self):
        """Create example plugins for reference"""
        # Example hashing plugin
        example_hash_plugin = '''#!/usr/bin/env python3
"""
Example Custom Hashing Plugin
Demonstrates how to create a custom hashing algorithm
"""
import hashlib
import os
from typing import List
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
'''
        
        example_file = self.plugin_directory / "examples" / "example_hash_plugin.py"
        if not example_file.exists():
            with open(example_file, 'w', encoding='utf-8') as f:
                f.write(example_hash_plugin)
                
        # Example file type plugin
        example_filetype_plugin = '''#!/usr/bin/env python3
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
'''
        
        example_json_file = self.plugin_directory / "examples" / "json_plugin.py"
        if not example_json_file.exists():
            with open(example_json_file, 'w', encoding='utf-8') as f:
                f.write(example_filetype_plugin)
                
    def load_all_plugins(self):
        """Load all plugins from plugin directory"""
        self.plugin_errors.clear()
        
        # Load from each subdirectory
        for plugin_type, plugin_dict, base_class in [
            ("hashing", self.hashing_plugins, HashingPlugin),
            ("file_types", self.file_type_plugins, FileTypePlugin),
            ("similarity", self.similarity_plugins, SimilarityPlugin)
        ]:
            plugin_dir = self.plugin_directory / plugin_type
            if plugin_dir.exists():
                self._load_plugins_from_directory(plugin_dir, plugin_dict, base_class)
                
        logger.info(f"✅ Loaded {len(self.hashing_plugins)} hashing plugins")
        logger.info(f"✅ Loaded {len(self.file_type_plugins)} file type plugins")
        logger.info(f"✅ Loaded {len(self.similarity_plugins)} similarity plugins")
        
        if self.plugin_errors:
            logger.warning(f"⚠️ {len(self.plugin_errors)} plugin loading errors")
            
    def _load_plugins_from_directory(self, directory: Path, plugin_dict: Dict, base_class):
        """Load plugins from a specific directory.

        Security:
          * Symlinks that resolve outside the plugin directory are skipped.
          * When `security_check` is enabled, only filenames listed in
            `enabled_plugins` are executed. This blocks arbitrary .py files
            dropped into plugins/ from being auto-executed (RCE protection).
        """
        try:
            directory_resolved = directory.resolve()
        except OSError as e:
            logger.warning(f"Cannot resolve plugin directory {directory}: {e}")
            return

        for plugin_file in directory.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            # Reject symlinks (and any path) whose resolved target escapes
            # the plugin directory tree.
            try:
                resolved = plugin_file.resolve()
            except OSError as e:
                logger.warning(f"  ⚠️ Skipping {plugin_file.name}: cannot resolve ({e})")
                continue
            try:
                resolved.relative_to(directory_resolved)
            except ValueError:
                logger.warning(
                    f"  ⚠️ Skipping {plugin_file.name}: symlink target escapes plugin dir"
                )
                continue
            if plugin_file.is_symlink():
                logger.warning(
                    f"  ⚠️ Skipping {plugin_file.name}: symlinks not permitted"
                )
                continue

            # Security allowlist gate.
            if self.security_check and plugin_file.name not in self.enabled_plugins:
                logger.info(
                    f"  ⏭️ Skipping {plugin_file.name}: not in enabled_plugins allowlist "
                    f"(set security_check=false in plugin_config.json to disable)"
                )
                continue

            try:
                plugin = self._load_plugin_file(plugin_file, base_class)
                if plugin:
                    if plugin.name in plugin_dict:
                        logger.warning(
                            f"  ⚠️ Overwriting existing plugin '{plugin.name}' "
                            f"with definition from {plugin_file.name}"
                        )
                    plugin_dict[plugin.name] = plugin
                    logger.info(f"  ✅ Loaded {plugin.name} v{plugin.version}")

            except Exception as e:
                error_info = {
                    "file": str(plugin_file),
                    "error": str(e),
                    "type": base_class.__name__
                }
                self.plugin_errors.append(error_info)
                logger.error(f"  ❌ Failed to load {plugin_file.name}: {e}")
                
    def _load_plugin_file(self, plugin_file: Path, base_class) -> Optional[PluginInterface]:
        """Load a single plugin file"""
        spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
        if not spec or not spec.loader:
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for create_plugin function
        if hasattr(module, 'create_plugin'):
            plugin_instance = module.create_plugin()
            if isinstance(plugin_instance, base_class):
                return plugin_instance
                
        # Look for plugin classes that are *defined* in this module — not
        # base classes or other plugin types imported from plugin_system.
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module.__name__
                and issubclass(obj, base_class)
                and obj is not base_class
            ):
                return obj()

        return None
        
    def get_available_plugins(self) -> Dict[str, List[Dict[str, str]]]:
        """Get information about all available plugins"""
        return {
            "hashing": [
                {
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "author": plugin.author,
                    "extensions": getattr(plugin, 'supported_extensions', [])
                }
                for plugin in self.hashing_plugins.values()
            ],
            "file_types": [
                {
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "author": plugin.author,
                    "extensions": getattr(plugin, 'supported_extensions', [])
                }
                for plugin in self.file_type_plugins.values()
            ],
            "similarity": [
                {
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "author": plugin.author
                }
                for plugin in self.similarity_plugins.values()
            ]
        }
        
    def get_plugin_for_file(self, file_path: str, plugin_type: str = "hashing") -> Optional[PluginInterface]:
        """Get the best plugin for a specific file"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if plugin_type == "hashing":
            plugins = self.hashing_plugins
        elif plugin_type == "file_types":
            plugins = self.file_type_plugins
        elif plugin_type == "similarity":
            plugins = self.similarity_plugins
        else:
            return None
            
        # Find plugins that support this extension
        compatible_plugins = []
        for plugin in plugins.values():
            if hasattr(plugin, 'supported_extensions'):
                if ext in plugin.supported_extensions:
                    compatible_plugins.append(plugin)
            elif hasattr(plugin, 'supports_file_type'):
                if plugin.supports_file_type(file_path):
                    compatible_plugins.append(plugin)
                    
        # Return first compatible plugin (could be enhanced with priority system)
        return compatible_plugins[0] if compatible_plugins else None
        
    def compute_hash_with_plugins(self, file_path: str) -> Dict[str, str]:
        """Compute hashes using all applicable plugins"""
        results = {}
        
        for plugin_name, plugin in self.hashing_plugins.items():
            if plugin.validate_file(file_path):
                try:
                    hash_value = plugin.compute_hash(file_path)
                    if hash_value:
                        results[f"plugin_{plugin_name}"] = hash_value
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} error: {e}")
                    
        return results
        
    def extract_features_with_plugins(self, file_path: str) -> Dict[str, Dict]:
        """Extract features using all applicable plugins"""
        results = {}
        
        for plugin_name, plugin in self.file_type_plugins.items():
            ext = os.path.splitext(file_path)[1].lower()
            if hasattr(plugin, 'supported_extensions') and ext in plugin.supported_extensions:
                try:
                    features = plugin.extract_features(file_path)
                    if features:
                        results[f"plugin_{plugin_name}"] = features
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} error: {e}")
                    
        return results
        
    def calculate_similarity_with_plugins(self, file1: str, file2: str) -> Dict[str, float]:
        """Calculate similarity using all applicable plugins"""
        results = {}
        
        for plugin_name, plugin in self.similarity_plugins.items():
            if plugin.supports_file_type(file1) and plugin.supports_file_type(file2):
                try:
                    similarity = plugin.calculate_similarity(file1, file2)
                    results[f"plugin_{plugin_name}"] = similarity
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} error: {e}")
                    
        return results
        
    def reload_plugins(self):
        """Reload all plugins"""
        self.hashing_plugins.clear()
        self.file_type_plugins.clear()
        self.similarity_plugins.clear()
        self.load_all_plugins()
        
    def get_plugin_errors(self) -> List[Dict]:
        """Get list of plugin loading errors"""
        return self.plugin_errors.copy()
        
    def install_plugin(self, plugin_file_path: str, plugin_type: str) -> bool:
        """Install a new plugin.

        Refuses to follow symlinks or to silently overwrite an existing
        plugin file. Logs the SHA256 of the source for audit trails.
        """
        try:
            source_path = Path(plugin_file_path)
            if not source_path.exists():
                logger.error(f"Plugin install failed: source not found: {source_path}")
                return False

            # Reject symlinked sources to avoid TOCTOU / path-escape attacks.
            if source_path.is_symlink():
                logger.error(
                    f"Plugin install refused: source is a symlink ({source_path})"
                )
                return False

            target_dir = self.plugin_directory / plugin_type
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name

            # Never silently overwrite an existing plugin.
            if target_path.exists():
                logger.error(
                    f"Plugin install refused: target already exists ({target_path}); "
                    f"remove the existing file first"
                )
                return False

            # Compute and log SHA256 of the source before copying.
            sha = hashlib.sha256()
            with open(source_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            digest = sha.hexdigest()
            logger.info(f"Installing plugin {source_path.name} (sha256={digest})")

            # Copy plugin file
            import shutil
            shutil.copy2(source_path, target_path)

            # Reload plugins
            self.reload_plugins()

            return True

        except Exception as e:
            logger.error(f"Plugin installation failed: {e}")
            return False

# Lazy global plugin manager — do not run plugin discovery on import.
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Return the process-wide PluginManager, constructing it on first use."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def __getattr__(name):
    """Backwards-compat shim: `from plugin_system import plugin_manager`."""
    if name == "plugin_manager":
        return get_plugin_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
