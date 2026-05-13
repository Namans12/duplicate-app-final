#!/usr/bin/env python3
"""
Intelligent Duplicate Merging System
Auto-suggests merging similar files with difference highlighting
"""
import os
import copy
import difflib
import functools
import shutil
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
import tempfile

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

class FileMerger:
    """Handles intelligent merging of duplicate and similar files"""
    
    def __init__(self):
        self.merge_strategies = {
            "text": "text_merge",
            "image": "image_merge", 
            "json": "json_merge",
            "config": "config_merge",
            "data": "data_merge"
        }
        
    def analyze_mergeable_files(self, file_group: List[str]) -> Dict[str, Any]:
        """Analyze a group of duplicate files for merge potential"""
        if len(file_group) < 2:
            return {"mergeable": False, "reason": "Not enough files to merge"}
            
        analysis = {
            "file_count": len(file_group),
            "files": file_group,
            "merge_suggestions": [],
            "differences": {},
            "recommended_action": "keep_separate",
            "confidence": 0.0
        }
        
        try:
            # Determine file type and merging strategy
            file_type = self._determine_file_type(file_group[0])
            analysis["file_type"] = file_type
            
            if file_type == "text":
                analysis.update(self._analyze_text_files(file_group))
            elif file_type == "image":
                analysis.update(self._analyze_image_files(file_group))
            elif file_type == "json":
                analysis.update(self._analyze_json_files(file_group))
            elif file_type == "config":
                analysis.update(self._analyze_config_files(file_group))
            else:
                analysis.update(self._analyze_binary_files(file_group))
                
        except Exception as e:
            analysis["error"] = str(e)
            
        return analysis
        
    def _determine_file_type(self, file_path: str) -> str:
        """Determine the type of file for merging strategy"""
        ext = os.path.splitext(file_path)[1].lower()
        
        text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.md', '.rst', '.log'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        json_extensions = {'.json', '.jsonl'}
        config_extensions = {'.ini', '.cfg', '.conf', '.xml', '.yaml', '.yml'}
        
        if ext in text_extensions:
            return "text"
        elif ext in image_extensions:
            return "image"
        elif ext in json_extensions:
            return "json"
        elif ext in config_extensions:
            return "config"
        else:
            return "binary"
            
    def _analyze_text_files(self, files: List[str]) -> Dict[str, Any]:
        """Analyze text files for merging potential"""
        analysis = {
            "mergeable": False,
            "differences": {},
            "similarity_matrix": {},
            "merge_suggestions": []
        }

        try:
            file_contents = {}
            file_lines = {}
            file_decode_errors = {}

            # Read all files
            READ_CAP = 1 * 1024 * 1024  # 1MB cap for similarity reading
            for file_path in files:
                try:
                    fsize = os.path.getsize(file_path)
                    with open(file_path, 'rb') as f:
                        raw = f.read(READ_CAP) if fsize > READ_CAP else f.read()
                    decoded = raw.decode('utf-8', errors='replace')
                    had_errors = '�' in decoded
                    file_contents[file_path] = decoded
                    file_lines[file_path] = decoded.splitlines()
                    file_decode_errors[file_path] = had_errors
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
            analysis["_decode_errors"] = file_decode_errors
                    
            if len(file_contents) < 2:
                return analysis
                
            # Calculate similarities and differences
            file_list = list(file_contents.keys())
            LARGE_FILE_THRESHOLD = 5 * 1024 * 1024  # 5MB
            for i, file1 in enumerate(file_list):
                for j, file2 in enumerate(file_list[i+1:], i+1):
                    try:
                        size1 = os.path.getsize(file1)
                        size2 = os.path.getsize(file2)
                    except OSError:
                        size1 = size2 = 0
                    if size1 > LARGE_FILE_THRESHOLD or size2 > LARGE_FILE_THRESHOLD:
                        size_sim = 1 - abs(size1 - size2) / max(size1, size2, 1)
                        lc1 = len(file_lines[file1])
                        lc2 = len(file_lines[file2])
                        line_sim = 1 - abs(lc1 - lc2) / max(lc1, lc2, 1)
                        similarity = (size_sim + line_sim) / 2
                    else:
                        similarity = self._calculate_text_similarity(
                            file_lines[file1], file_lines[file2]
                        )
                    analysis["similarity_matrix"][f"{os.path.basename(file1)}_vs_{os.path.basename(file2)}"] = similarity
                    
                    if similarity > 0.8:  # High similarity
                        diff_info = self._generate_text_diff(file1, file2, file_lines)
                        analysis["differences"][f"{file1}_vs_{file2}"] = diff_info
                        
                        if diff_info["minor_differences"]:
                            suggestion = self._suggest_text_merge(file1, file2, diff_info)
                            analysis["merge_suggestions"].append(suggestion)
                            
            # Determine if files are mergeable
            if analysis["merge_suggestions"]:
                analysis["mergeable"] = True
                analysis["recommended_action"] = "merge_with_review"
                analysis["confidence"] = max(s["confidence"] for s in analysis["merge_suggestions"])
                
        except Exception as e:
            analysis["error"] = str(e)
            
        return analysis
        
    def _calculate_text_similarity(self, lines1: List[str], lines2: List[str]) -> float:
        """Calculate similarity between two text files"""
        if not lines1 and not lines2:
            return 1.0
        if not lines1 or not lines2:
            return 0.0
            
        # Use difflib to calculate similarity
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        return matcher.ratio()
        
    def _generate_text_diff(self, file1: str, file2: str, file_lines: Dict) -> Dict[str, Any]:
        """Generate detailed difference information for text files"""
        lines1 = file_lines[file1]
        lines2 = file_lines[file2]
        
        diff = list(difflib.unified_diff(
            lines1, lines2,
            fromfile=os.path.basename(file1),
            tofile=os.path.basename(file2),
            lineterm=''
        ))
        
        # Analyze the differences
        additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        total_lines = max(len(lines1), len(lines2))
        
        diff_info = {
            "unified_diff": diff,
            "additions": additions,
            "deletions": deletions,
            "total_changes": additions + deletions,
            "change_percentage": (additions + deletions) / max(total_lines, 1) * 100,
            "minor_differences": (additions + deletions) / max(total_lines, 1) < 0.1  # Less than 10% changes
        }
        
        return diff_info
        
    def _suggest_text_merge(self, file1: str, file2: str, diff_info: Dict) -> Dict[str, Any]:
        """Suggest how to merge two similar text files"""
        suggestion = {
            "type": "text_merge",
            "primary_file": file1,
            "secondary_file": file2,
            "confidence": 0.0,
            "merge_strategy": "manual_review",
            "preview": None
        }
        
        if diff_info["change_percentage"] < 5:  # Very minor differences
            suggestion["merge_strategy"] = "auto_merge_newer"
            suggestion["confidence"] = 0.9
            suggestion["description"] = "Files are nearly identical, suggest keeping newer version"
        elif diff_info["change_percentage"] < 10:  # Minor differences
            suggestion["merge_strategy"] = "merge_with_comments"
            suggestion["confidence"] = 0.7
            suggestion["description"] = "Minor differences detected, suggest merging with conflict markers"
        else:
            suggestion["merge_strategy"] = "manual_review"
            suggestion["confidence"] = 0.5
            suggestion["description"] = "Significant differences require manual review"
            
        # Generate preview of merged content
        try:
            suggestion["preview"] = self._create_merge_preview(file1, file2, diff_info)
        except Exception as e:
            suggestion["preview_error"] = str(e)
            
        return suggestion
        
    def _create_merge_preview(self, file1: str, file2: str, diff_info: Dict) -> str:
        """Create a preview of what the merged file would look like"""
        try:
            with open(file1, 'r', encoding='utf-8', errors='replace') as f:
                content1 = f.read()
            with open(file2, 'r', encoding='utf-8', errors='replace') as f:
                content2 = f.read()
                
            # Simple merge strategy: use newer file as base and add conflict markers
            file1_mtime = os.path.getmtime(file1)
            file2_mtime = os.path.getmtime(file2)
            
            if file1_mtime > file2_mtime:
                base_content = content1
                other_content = content2
                base_name = os.path.basename(file1)
                other_name = os.path.basename(file2)
            else:
                base_content = content2
                other_content = content1
                base_name = os.path.basename(file2)
                other_name = os.path.basename(file1)
                
            # For now, return simple conflict markers
            preview = f"""
# Merged content from {base_name} and {other_name}
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<<<<<<< {base_name}
{base_content}
=======
{other_content}
>>>>>>> {other_name}
"""
            
            return preview.strip()
            
        except Exception as e:
            return f"Error creating preview: {e}"
            
    def _analyze_image_files(self, files: List[str]) -> Dict[str, Any]:
        """Analyze image files for merging potential"""
        analysis = {
            "mergeable": False,
            "differences": {},
            "merge_suggestions": []
        }
        
        if not HAS_PILLOW:
            analysis["error"] = "PIL not available for image analysis"
            return analysis
            
        try:
            image_info = {}
            
            # Analyze each image
            for file_path in files:
                try:
                    if HAS_PILLOW:
                        with Image.open(file_path) as img:
                            image_info[file_path] = {
                                "size": img.size,
                                "mode": img.mode,
                                "format": img.format,
                                "file_size": os.path.getsize(file_path)
                            }
                except Exception as e:
                    print(f"Error analyzing image {file_path}: {e}")
                    
            if len(image_info) < 2:
                return analysis
                
            # Look for same images in different formats/qualities
            file_list = list(image_info.keys())
            for i, file1 in enumerate(file_list):
                for j, file2 in enumerate(file_list[i+1:], i+1):
                    info1 = image_info[file1]
                    info2 = image_info[file2]
                    
                    # Check if same dimensions
                    if info1["size"] == info2["size"]:
                        suggestion = {
                            "type": "image_quality_merge",
                            "files": [file1, file2],
                            "description": "Same dimensions, different file sizes - possible quality difference",
                            "recommended_action": "keep_file1" if info1["file_size"] > info2["file_size"] else "keep_file2",
                            "confidence": 0.8
                        }
                        analysis["merge_suggestions"].append(suggestion)
                        analysis["mergeable"] = True
                        
        except Exception as e:
            analysis["error"] = str(e)
            
        return analysis
        
    def _analyze_json_files(self, files: List[str]) -> Dict[str, Any]:
        """Analyze JSON files for merging potential"""
        analysis = {
            "mergeable": False,
            "differences": {},
            "merge_suggestions": []
        }
        
        try:
            json_data = {}
            
            # Load all JSON files
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data[file_path] = json.load(f)
                except Exception as e:
                    print(f"Error loading JSON {file_path}: {e}")
                    
            if len(json_data) < 2:
                return analysis
                
            # Compare JSON structures
            file_list = list(json_data.keys())
            for i, file1 in enumerate(file_list):
                for j, file2 in enumerate(file_list[i+1:], i+1):
                    data1 = json_data[file1]
                    data2 = json_data[file2]
                    
                    # Check structural similarity
                    similarity = self._compare_json_structure(data1, data2)
                    
                    if similarity > 0.8:
                        diff_info = self._generate_json_diff(data1, data2)
                        analysis["differences"][f"{file1}_vs_{file2}"] = diff_info
                        
                        if diff_info["mergeable"]:
                            suggestion = {
                                "type": "json_merge",
                                "files": [file1, file2],
                                "description": "Similar JSON structure with mergeable differences",
                                "merge_strategy": "deep_merge",
                                "confidence": similarity
                            }
                            analysis["merge_suggestions"].append(suggestion)
                            analysis["mergeable"] = True
                            
        except Exception as e:
            analysis["error"] = str(e)
            
        return analysis
        
    def _structure_signature(self, obj: Any, path: tuple = ()) -> set:
        sig = set()
        if isinstance(obj, dict):
            sig.add((path, 'dict'))
            for k, v in obj.items():
                sig |= self._structure_signature(v, path + (('key', k),))
        elif isinstance(obj, list):
            sig.add((path, 'list'))
            for idx, v in enumerate(obj):
                sig |= self._structure_signature(v, path + (('idx', idx),))
        else:
            sig.add((path, type(obj).__name__))
        return sig

    def _compare_json_structure(self, data1: Any, data2: Any) -> float:
        """Compare structural similarity of JSON data"""
        try:
            sig1 = self._structure_signature(data1)
            sig2 = self._structure_signature(data2)

            if not sig1 and not sig2:
                return 1.0
            if not sig1 or not sig2:
                return 0.0

            intersection = len(sig1 & sig2)
            union = len(sig1 | sig2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0
            
    def _generate_json_diff(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """Generate difference information for JSON data"""
        diff_info = {
            "mergeable": False,
            "conflicts": [],
            "additions": {},
            "deletions": {},
            "modifications": {}
        }
        
        try:
            # Simple difference detection (can be enhanced)
            if isinstance(data1, dict) and isinstance(data2, dict):
                all_keys = set(data1.keys()) | set(data2.keys())
                
                for key in all_keys:
                    if key in data1 and key in data2:
                        if data1[key] != data2[key]:
                            diff_info["modifications"][key] = {
                                "old_value": data1[key],
                                "new_value": data2[key]
                            }
                    elif key in data1:
                        diff_info["deletions"][key] = data1[key]
                    else:
                        diff_info["additions"][key] = data2[key]
                        
                # Determine if mergeable (no conflicts)
                diff_info["mergeable"] = len(diff_info["modifications"]) == 0
                
        except Exception as e:
            diff_info["error"] = str(e)
            
        return diff_info
        
    def _analyze_config_files(self, files: List[str]) -> Dict[str, Any]:
        """Analyze configuration files for merging potential"""
        # Similar to text files but with config-specific logic
        return self._analyze_text_files(files)
        
    def _analyze_binary_files(self, files: List[str]) -> Dict[str, Any]:
        """Analyze binary files (limited merging potential)"""
        analysis = {
            "mergeable": False,
            "reason": "Binary files cannot be automatically merged",
            "recommendations": []
        }
        
        # Check for identical files
        file_sizes = {}
        for file_path in files:
            try:
                file_sizes[file_path] = os.path.getsize(file_path)
            except Exception:
                pass
                
        if len(set(file_sizes.values())) == 1:
            analysis["recommendations"].append({
                "action": "keep_newest",
                "reason": "All files have identical size, suggest keeping newest",
                "confidence": 0.9
            })
        else:
            analysis["recommendations"].append({
                "action": "manual_review",
                "reason": "Different file sizes, manual review required",
                "confidence": 0.3
            })
            
        return analysis
        
    def execute_merge(self, merge_suggestion: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute a merge suggestion"""
        result = {
            "success": False,
            "output_file": None,
            "backup_files": [],
            "error": None
        }
        
        try:
            merge_type = merge_suggestion.get("type", "unknown")
            
            if merge_type == "text_merge":
                result = self._execute_text_merge(merge_suggestion, output_path)
            elif merge_type == "image_quality_merge":
                result = self._execute_image_merge(merge_suggestion, output_path)
            elif merge_type == "json_merge":
                result = self._execute_json_merge(merge_suggestion, output_path)
            else:
                result["error"] = f"Unknown merge type: {merge_type}"
                
        except Exception as e:
            result["error"] = str(e)
            
        return result
        
    def _execute_text_merge(self, suggestion: Dict, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute text file merge"""
        result = {"success": False, "backup_files": []}

        try:
            primary_file = suggestion["primary_file"]
            secondary_file = suggestion["secondary_file"]

            backup_dir = os.path.join(os.path.expanduser('~'), '.duplicate_finder_merge_backups')
            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup1 = os.path.join(backup_dir, f"{os.path.basename(primary_file)}.{timestamp}.backup")
            backup2 = os.path.join(backup_dir, f"{os.path.basename(secondary_file)}.{timestamp}.backup")

            shutil.copy2(primary_file, backup1)
            shutil.copy2(secondary_file, backup2)
            result["backup_files"] = [backup1, backup2]

            # Check inputs for decode errors before any write
            had_decode_errors = False
            for fp in (primary_file, secondary_file):
                try:
                    with open(fp, 'rb') as f:
                        if '�' in f.read().decode('utf-8', errors='replace'):
                            had_decode_errors = True
                            break
                except Exception:
                    pass

            # Execute merge based on strategy
            strategy = suggestion.get("merge_strategy", "manual_review")

            if strategy == "auto_merge_newer":
                # Avoid writing the conflict-preview over the original
                if output_path is None or output_path == primary_file:
                    output_path = primary_file + '.merged'
                    print(f"Writing merged output to: {output_path}")

                if had_decode_errors:
                    raise ValueError("Source file had decode errors; refusing to write merge result")

                # Keep the newer file
                file1_mtime = os.path.getmtime(primary_file)
                file2_mtime = os.path.getmtime(secondary_file)

                src = primary_file if file1_mtime > file2_mtime else secondary_file
                if os.path.abspath(src) == os.path.abspath(output_path):
                    result["success"] = True
                    result["output_file"] = output_path
                    return result
                try:
                    shutil.copy2(src, output_path)
                except shutil.SameFileError:
                    pass

            elif strategy == "merge_with_comments":
                preview = suggestion.get("preview")
                if not preview:
                    raise ValueError("Empty preview; cannot write merge_with_comments output")
                if had_decode_errors:
                    raise ValueError("Source file had decode errors; refusing to write merge result")
                # Always write to a sibling, never the original
                conflict_path = (output_path + '.merge_conflict') if output_path else (primary_file + '.merge_conflict.txt')
                if conflict_path == primary_file or conflict_path == secondary_file:
                    conflict_path = primary_file + '.merge_conflict.txt'
                with open(conflict_path, 'w', encoding='utf-8') as f:
                    f.write(preview)
                output_path = conflict_path
                print(f"Wrote conflict-marker preview to: {conflict_path}")
            else:
                if output_path is None:
                    raise ValueError("output_path required")

            result["success"] = True
            result["output_file"] = output_path

        except Exception as e:
            result["error"] = str(e)

        return result
        
    def _execute_image_merge(self, suggestion: Dict, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute image merge (usually just keep best quality)"""
        result = {"success": False, "backup_files": []}
        
        try:
            files = suggestion["files"]
            
            # Find file with largest size (assuming better quality)
            best_file = max(files, key=lambda f: os.path.getsize(f))
            
            if output_path is None:
                raise ValueError("output_path required")

            if output_path != best_file:
                if os.path.abspath(best_file) == os.path.abspath(output_path):
                    pass
                else:
                    try:
                        shutil.copy2(best_file, output_path)
                    except shutil.SameFileError:
                        pass
                
            result["success"] = True
            result["output_file"] = output_path
            result["kept_file"] = best_file
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
        
    def _execute_json_merge(self, suggestion: Dict, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute JSON merge"""
        result = {"success": False, "backup_files": []}
        
        try:
            files = suggestion["files"]
            
            # Load JSON data
            json_objects = []
            had_decode_errors = False
            for file_path in files:
                with open(file_path, 'rb') as f:
                    raw = f.read()
                decoded = raw.decode('utf-8', errors='replace')
                if '�' in decoded:
                    had_decode_errors = True
                json_objects.append(json.loads(decoded))

            # Deep merge JSON objects
            merged_data = self._deep_merge_json(json_objects)

            # Determine output file
            if output_path is None:
                raise ValueError("output_path required")

            if had_decode_errors:
                raise ValueError("Source file had decode errors; refusing to write merge result")

            # Write merged JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
                
            result["success"] = True
            result["output_file"] = output_path
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
        
    def _deep_merge_json(self, json_objects: List[Any]) -> Any:
        """Deep merge multiple JSON objects"""
        if not json_objects:
            return {}
            
        result = copy.deepcopy(json_objects[0])
        
        for obj in json_objects[1:]:
            if isinstance(result, dict) and isinstance(obj, dict):
                for key, value in obj.items():
                    if key in result:
                        if isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = self._deep_merge_json([result[key], value])
                        else:
                            # Keep the newer value (from later object)
                            result[key] = value
                    else:
                        result[key] = value
                        
        return result

@functools.lru_cache(maxsize=1)
def get_file_merger() -> FileMerger:
    return FileMerger()
