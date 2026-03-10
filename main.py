#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duplicate File Finder with Advanced Comparison and File Management
Features: Recycle Bin, Scan Filters, Progress with ETA, Export Results
Technical: CUDA GPU Acceleration, Parallel Processing, Memory Optimization
"""
import sys
import os
import hashlib
import shutil
import json
import csv
import time
import threading
import multiprocessing
from copy import deepcopy
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from functools import partial
from queue import Queue
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QListWidget, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox, QProgressBar, QLabel, QHBoxLayout, QComboBox,
    QTextEdit, QSplitter, QCheckBox, QGroupBox, QGridLayout, QSpinBox,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QSlider, QLineEdit, QFrame,
    QScrollArea, QMenu, QToolBar, QStatusBar, QToolButton, QStyle,
    QSystemTrayIcon, QWidgetAction, QInputDialog, QDialog, QSizePolicy,
    QButtonGroup
)
from PyQt6.QtCore import QEvent, QObject, QThread, pyqtSignal, Qt, QTimer, QSettings, QSize, QMimeData
from PyQt6.QtGui import QPixmap, QAction, QShortcut, QKeySequence, QIcon, QDragEnterEvent, QDropEvent, QPalette, QColor, QFont

# Initialize logger early since it's needed for import error handling
import logging
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from PIL import Image
    import imagehash
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    Image = None
    imagehash = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from pydub import AudioSegment
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Try to import numpy for optimized operations (independent of pydub)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# CUDA / GPU Acceleration imports
HAS_CUDA = False
CUDA_BACKEND = 'cpu'

# Video fingerprinting support
HAS_VIDEO_FINGERPRINTER = False
video_fingerprinter = None

# Audio fingerprinting support
HAS_AUDIO_FINGERPRINTER = False
audio_fingerprinter = None

# Semantic similarity support
HAS_SEMANTIC_SIMILARITY = False
semantic_engine = None

# Pipeline status function
get_acceleration_summary = None

# Import GPU acceleration status from gpu_accelerator module
try:
    from gpu_accelerator import (
        video_fingerprinter as _vfp, 
        gpu_accelerator as _gpu_accel, 
        semantic_engine as _sem_engine,
        audio_fingerprinter as _audio_fp,
        get_acceleration_summary as _get_accel_summary,
        HAS_CUDA as _HAS_CUDA,
        BACKEND as _BACKEND
    )
    video_fingerprinter = _vfp
    HAS_VIDEO_FINGERPRINTER = _vfp.has_cv2
    audio_fingerprinter = _audio_fp
    HAS_AUDIO_FINGERPRINTER = _audio_fp.has_chromaprint or _audio_fp.has_librosa
    semantic_engine = _sem_engine
    HAS_SEMANTIC_SIMILARITY = _sem_engine.is_available
    get_acceleration_summary = _get_accel_summary
    
    # Reuse GPU status from gpu_accelerator
    HAS_CUDA = _HAS_CUDA
    CUDA_BACKEND = _BACKEND
    
except ImportError as e:
    logger.info(f"[GPU] ℹ️ GPU accelerator not available: {e}")
    HAS_CUDA = False
    CUDA_BACKEND = 'cpu'
except Exception as e:
    logger.warning(f"[GPU] ⚠️ GPU accelerator import error: {e}")
    HAS_CUDA = False
    CUDA_BACKEND = 'cpu'

# Import enterprise modules
HAS_ENTERPRISE_MODULES = False
DatabaseManager = None
PluginManager = None
MemoryManager = None
StreamingHasher = None
FileMerger = None

try:
    from database_manager import DatabaseManager
    from plugin_system import PluginManager
    from memory_optimizer import MemoryManager, StreamingHasher
    from duplicate_merger import FileMerger
    HAS_ENTERPRISE_MODULES = True
    logger.info("Advanced modules loaded successfully")
except ImportError as e:
    logger.info(f"Advanced modules not available: {e}")
except Exception as e:
    logger.warning(f"Advanced modules error: {e}")

# Dataclass for structured progress reporting
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class ScanProgress:
    """Structured progress information for scan operations"""
    current_file: str = ""
    files_processed: int = 0
    total_files: int = 0
    current_phase: str = "Initializing"
    start_time: float = 0.0
    duplicates_found: int = 0
    bytes_saved: int = 0


# Import shared utilities from dependencies module
from dependencies import (
    AppConfig, HAS_CUDA as DEP_HAS_CUDA, CUDA_BACKEND as DEP_CUDA_BACKEND,
    HAS_SEND2TRASH as DEP_SEND2TRASH, send2trash_func
)

# CF-2: Fall back to dependency-level GPU detection if gpu_accelerator failed to load
if not HAS_CUDA and DEP_HAS_CUDA:
    HAS_CUDA = DEP_HAS_CUDA
    CUDA_BACKEND = DEP_CUDA_BACKEND

# CF-3: Consolidate send2trash — use single source from dependencies module
HAS_SEND2TRASH = DEP_SEND2TRASH
send2trash = send2trash_func  # Alias so existing call sites work unchanged
from audio_quality_analyzer import AudioQualityAnalyzer
from media_profiles import (
    compare_archive_files,
    compare_document_files,
    compare_image_files,
    compute_exact_hash,
    get_default_media_profiles,
    get_media_type,
    get_optional_engine_status,
    get_profile_for_path,
    merge_media_profiles,
    summarize_media_profile,
)

# Use centralized get_optimal_workers
get_optimal_workers = AppConfig.get_optimal_workers

# Get optimal worker count
CPU_COUNT = multiprocessing.cpu_count()
OPTIMAL_WORKERS = get_optimal_workers(HAS_CUDA)
OPTIMAL_WORKERS_CPU_ONLY = get_optimal_workers(False)


if HAS_CUDA:
    logger.info(f"Using {OPTIMAL_WORKERS} worker threads (GPU mode, {CPU_COUNT} cores available)")
else:
    logger.info(f"Using {OPTIMAL_WORKERS} worker threads (CPU-only mode, {CPU_COUNT} cores available)")

# Theme definitions
DARK_THEME = """
QMainWindow, QWidget {
    background-color: #17191d;
    color: #f3f6fb;
}
QGroupBox {
    border: 1px solid #323846;
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 24px;
    font-weight: bold;
    color: #78c9ff;
    background-color: #1c2027;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 2px 8px 4px 8px;
}
QPushButton, QToolButton {
    background-color: #234a7a;
    color: white;
    border: 1px solid #35649c;
    padding: 9px 16px;
    border-radius: 8px;
    font-weight: 600;
    min-height: 20px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #2b5f9a;
    border-color: #4f88c5;
}
QPushButton:pressed, QToolButton:pressed {
    background-color: #1c3f67;
}
QPushButton:checked, QToolButton:checked {
    background-color: #0f79b8;
    border-color: #55b3e8;
}
QPushButton:disabled, QToolButton:disabled {
    background-color: #2a2f39;
    color: #6d7480;
    border-color: #2f3540;
}
QLineEdit, QSpinBox, QComboBox {
    background-color: #20242c;
    border: 1px solid #394150;
    border-radius: 8px;
    padding: 7px 10px;
    color: #f3f6fb;
    min-height: 20px;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #78c9ff;
}
QComboBox {
    padding-right: 34px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: 1px solid #394150;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background-color: #262b35;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #20242c;
    color: #f3f6fb;
    border: 1px solid #394150;
    selection-background-color: #2d5d95;
    selection-color: white;
    outline: 0;
}
QTreeWidget, QListWidget, QTextEdit {
    background-color: #1d2128;
    border: 1px solid #323846;
    border-radius: 8px;
    color: #f3f6fb;
}
QTreeWidget::item:selected, QListWidget::item:selected {
    background-color: #295e96;
}
QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #282d36;
}
QHeaderView::section {
    background-color: #232833;
    color: #d8e2ee;
    padding: 8px 10px;
    border: none;
    border-right: 1px solid #323846;
    border-bottom: 1px solid #323846;
    font-weight: 600;
}
QProgressBar {
    border: 1px solid #394150;
    border-radius: 7px;
    text-align: center;
    background-color: #20242c;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #2f9d68;
    border-radius: 6px;
}
QTabWidget::pane {
    border: 1px solid #323846;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background-color: #232833;
    color: #dce6f1;
    padding: 10px 18px;
    border: 1px solid #323846;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background-color: #295e96;
    color: white;
}
QTabBar::tab:hover {
    background-color: #2b3140;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #323846;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #78c9ff;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QCheckBox {
    color: #f3f6fb;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
}
QCheckBox::indicator:unchecked {
    border: 2px solid #727b88;
    border-radius: 3px;
    background: transparent;
}
QCheckBox::indicator:checked {
    border: 2px solid #2f9d68;
    border-radius: 3px;
    background: #2f9d68;
}
QToolBar {
    background-color: #1c2027;
    border: none;
    spacing: 8px;
    padding: 8px;
}
QStatusBar {
    background-color: #14171c;
    color: #9ea7b3;
}
QToolTip {
    background-color: #232833;
    color: white;
    border: 1px solid #78c9ff;
    padding: 6px 8px;
    border-radius: 6px;
}
QMenu {
    background-color: #20242c;
    border: 1px solid #323846;
    color: white;
    padding: 6px;
}
QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: #295e96;
}
#settingsHelpText {
    background-color: #202733;
    color: #d9e4f0;
    padding: 12px;
    border: 1px solid #323846;
    border-radius: 8px;
    line-height: 1.35em;
}
"""

LIGHT_THEME = """
QMainWindow, QWidget {
    background-color: #f3f6fb;
    color: #243142;
}
QGroupBox {
    border: 1px solid #cfd8e3;
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 24px;
    font-weight: bold;
    color: #1f5f95;
    background-color: #fbfcfe;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 2px 8px 4px 8px;
}
QPushButton, QToolButton {
    background-color: #2e6ea8;
    color: white;
    border: 1px solid #4a84ba;
    padding: 9px 16px;
    border-radius: 8px;
    font-weight: 600;
    min-height: 20px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #3a7dbc;
    border-color: #5c97cc;
}
QPushButton:pressed, QToolButton:pressed {
    background-color: #265f92;
}
QPushButton:checked, QToolButton:checked {
    background-color: #1676bf;
    border-color: #3a93d8;
}
QPushButton:disabled, QToolButton:disabled {
    background-color: #d9e0e8;
    color: #7d8792;
    border-color: #d0d7df;
}
QLineEdit, QSpinBox, QComboBox {
    background-color: white;
    border: 1px solid #c8d1dc;
    border-radius: 8px;
    padding: 7px 10px;
    color: #243142;
    min-height: 20px;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #2f84c8;
}
QComboBox {
    padding-right: 34px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: 1px solid #c8d1dc;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background-color: #eef3f8;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: white;
    color: #243142;
    border: 1px solid #c8d1dc;
    selection-background-color: #d7ebfb;
    selection-color: #243142;
    outline: 0;
}
QTreeWidget, QListWidget, QTextEdit {
    background-color: white;
    border: 1px solid #cfd8e3;
    border-radius: 8px;
    color: #243142;
}
QTreeWidget::item:selected, QListWidget::item:selected {
    background-color: #d8ebfa;
    color: #243142;
}
QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #eef5fb;
}
QHeaderView::section {
    background-color: #eef3f8;
    color: #304155;
    padding: 8px 10px;
    border: none;
    border-right: 1px solid #d6dee8;
    border-bottom: 1px solid #d6dee8;
    font-weight: 600;
}
QProgressBar {
    border: 1px solid #c8d1dc;
    border-radius: 7px;
    text-align: center;
    background-color: #e9eef4;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #309866;
    border-radius: 6px;
}
QTabWidget::pane {
    border: 1px solid #d0d8e3;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background-color: #e8eef5;
    color: #314155;
    padding: 10px 18px;
    border: 1px solid #d0d8e3;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background-color: #2e6ea8;
    color: white;
}
QTabBar::tab:hover {
    background-color: #dce6f0;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #cfd8e3;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2e84c8;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QCheckBox {
    color: #243142;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
}
QCheckBox::indicator:unchecked {
    border: 2px solid #8390a0;
    border-radius: 3px;
    background: transparent;
}
QCheckBox::indicator:checked {
    border: 2px solid #309866;
    border-radius: 3px;
    background: #309866;
}
QToolBar {
    background-color: #eef3f8;
    border: none;
    spacing: 8px;
    padding: 8px;
}
QStatusBar {
    background-color: #e8eef5;
    color: #5f6d7d;
}
QToolTip {
    background-color: #243142;
    color: white;
    border: 1px solid #2e84c8;
    padding: 6px 8px;
    border-radius: 6px;
}
QMenu {
    background-color: white;
    border: 1px solid #cfd8e3;
    color: #243142;
    padding: 6px;
}
QMenu::item {
    padding: 8px 24px 8px 12px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: #d8ebfa;
}
#settingsHelpText {
    background-color: #eef4fa;
    color: #304155;
    padding: 12px;
    border: 1px solid #d0d8e3;
    border-radius: 8px;
    line-height: 1.35em;
}
"""


class FileComparator:
    """Advanced file comparison utility"""
    
    @staticmethod
    def compare_by_name(file1, file2, similarity_threshold=0.8):
        """Compare files by name similarity"""
        name1 = os.path.basename(file1).lower()
        name2 = os.path.basename(file2).lower()
        ratio = SequenceMatcher(None, name1, name2).ratio()
        return ratio >= similarity_threshold
    
    @staticmethod
    def compare_by_size(file1, file2, tolerance_mb=1):
        """Compare files by size with tolerance"""
        try:
            size1 = os.path.getsize(file1)
            size2 = os.path.getsize(file2)
            diff_mb = abs(size1 - size2) / (1024 * 1024)
            return diff_mb <= tolerance_mb
        except Exception:
            return False
    
    @staticmethod
    def compare_by_content(file1, file2, chunk_size=8192):
        """Compare files by content (exact match)"""
        try:
            with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                while True:
                    chunk1 = f1.read(chunk_size)
                    chunk2 = f2.read(chunk_size)
                    if chunk1 != chunk2:
                        return False
                    if not chunk1:  # EOF
                        return True
        except Exception:
            return False
    
    @staticmethod
    def compare_text_similarity(file1, file2, similarity_threshold=0.9, use_semantic=True):
        """Compare text files by content similarity.
        
        Uses GPU-accelerated semantic similarity if available, with fallback
        to SequenceMatcher for syntactic comparison.
        
        Args:
            file1: First file path.
            file2: Second file path.
            similarity_threshold: Minimum similarity (0.0 to 1.0).
            use_semantic: Whether to use semantic similarity (GPU-accelerated).
            
        Returns:
            True if files are similar above threshold.
        """
        try:
            ext1 = os.path.splitext(file1)[1].lower()
            ext2 = os.path.splitext(file2)[1].lower()
            
            text_extensions = ['.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.xml', '.csv', '.log', '.yaml', '.yml', '.ts', '.jsx', '.tsx']
            
            if ext1 in text_extensions and ext2 in text_extensions:
                with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                    with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                        content1 = f1.read()
                        content2 = f2.read()
                        
                        # Skip very large files for semantic comparison
                        max_semantic_chars = 50000
                        
                        # Try GPU-accelerated semantic similarity first
                        if use_semantic and HAS_SEMANTIC_SIMILARITY and semantic_engine is not None:
                            if len(content1) <= max_semantic_chars and len(content2) <= max_semantic_chars:
                                try:
                                    similarity = semantic_engine.compare(content1, content2)
                                    return similarity >= similarity_threshold
                                except Exception:
                                    pass  # Fall through to syntactic comparison
                        
                        # Fallback to syntactic comparison (SequenceMatcher)
                        ratio = SequenceMatcher(None, content1, content2).ratio()
                        return ratio >= similarity_threshold
        except Exception:
            pass
        return False


class ScanFilters:
    """File scanning filters"""
    
    # Common file type categories
    FILE_TYPES = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico', '.svg'],
        'documents': ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp'],
        'text': ['.txt', '.md', '.rtf', '.csv', '.json', '.xml', '.html', '.css', '.log'],
        'code': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
        'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'],
        'archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
    }
    
    # Common folders to exclude
    EXCLUDE_FOLDERS = [
        'node_modules', '.git', '.svn', '__pycache__', '.venv', 'venv',
        '.idea', '.vscode', 'build', 'dist', 'target', '.gradle'
    ]
    
    def __init__(self):
        self.enabled_types = set()  # Empty means all types
        self.excluded_folders = set(self.EXCLUDE_FOLDERS)
        self.min_size_bytes = 0
        self.max_size_bytes = float('inf')
        self.custom_extensions = set()
    
    def passes_filter(self, file_path):
        """Check if file passes all filters"""
        # Check excluded folders
        path_parts = file_path.replace('\\', '/').split('/')
        for part in path_parts:
            if part in self.excluded_folders:
                return False
        
        # Check file size
        try:
            size = os.path.getsize(file_path)
            if size < self.min_size_bytes or size > self.max_size_bytes:
                return False
        except Exception:
            return False
        
        # Check file type if filter is enabled
        if self.enabled_types or self.custom_extensions:
            ext = os.path.splitext(file_path)[1].lower()
            
            # Check custom extensions first
            if self.custom_extensions:
                if ext in self.custom_extensions:
                    return True
            
            # Check category types
            if self.enabled_types:
                for type_name in self.enabled_types:
                    if ext in self.FILE_TYPES.get(type_name, []):
                        return True
                return False
        
        return True


CATEGORY_PRESETS = {
    "general": {
        "label": "General Files",
        "types": [],
        "accuracy": {
            "hash_type": "MD5",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.85,
            "compare_by_size": True,
            "size_tolerance": 2,
            "compare_text_content": True,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    },
    "photos": {
        "label": "Photos",
        "types": ["images"],
        "accuracy": {
            "hash_type": "Perceptual",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.8,
            "compare_by_size": True,
            "size_tolerance": 0.5,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "aggressive",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    },
    "videos": {
        "label": "Videos",
        "types": ["video"],
        "accuracy": {
            "hash_type": "Perceptual",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.85,
            "compare_by_size": True,
            "size_tolerance": 10,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "aggressive",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 5,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    },
    "music": {
        "label": "Music",
        "types": ["audio"],
        "accuracy": {
            "hash_type": "Audio",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.85,
            "compare_by_size": True,
            "size_tolerance": 3,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 2,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    },
    "documents": {
        "label": "Documents",
        "types": ["documents", "text"],
        "accuracy": {
            "hash_type": "Semantic",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.85,
            "compare_by_size": True,
            "size_tolerance": 1,
            "compare_text_content": True,
            "content_similarity": 0.92,
            "phash_preset": "balanced",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    },
    "archives": {
        "label": "Archives",
        "types": ["archives"],
        "accuracy": {
            "hash_type": "SHA256",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.85,
            "compare_by_size": True,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    },
    "code": {
        "label": "Code",
        "types": ["code", "text"],
        "accuracy": {
            "hash_type": "SHA256",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.85,
            "compare_by_size": True,
            "size_tolerance": 1,
            "compare_text_content": True,
            "content_similarity": 0.95,
            "phash_preset": "balanced",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    },
    "other": {
        "label": "Other Files",
        "types": [],
        "accuracy": {
            "hash_type": "MD5",
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 0.85,
            "compare_by_size": True,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        },
        "performance": {
            "hash_type": "MD5",
            "find_similar": False,
            "compare_by_name": False,
            "name_similarity": 0.8,
            "compare_by_size": False,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "phash_preset": "balanced",
            "use_gpu": True
        }
    }
}


class UndoManager:
    """Manages file backup and restore operations for undo functionality"""
    
    def __init__(self):
        self.backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".duplicate_finder_backup")
        self.operations = []
        self.max_operations = 100
        self._ensure_backup_dir()
        
    def _ensure_backup_dir(self):
        """Create backup directory if it doesn't exist"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            
    def backup_file(self, file_path):
        """Backup a file before deletion"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(file_path)
            backup_name = f"{timestamp}_{filename}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            shutil.copy2(file_path, backup_path)
            
            operation = {
                'type': 'delete',
                'original_path': file_path,
                'backup_path': backup_path,
                'timestamp': timestamp
            }
            
            self.operations.append(operation)
            
            # Keep only last N operations
            if len(self.operations) > self.max_operations:
                old_op = self.operations.pop(0)
                if os.path.exists(old_op['backup_path']):
                    try:
                        os.remove(old_op['backup_path'])
                    except Exception:
                        pass
                    
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def record_move(self, original_path, new_path):
        """Record a move operation for undo."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            operation = {
                'type': 'move',
                'original_path': original_path,
                'new_path': new_path,
                'timestamp': timestamp
            }
            self.operations.append(operation)
            if len(self.operations) > self.max_operations:
                self.operations.pop(0)
            return True
        except Exception as e:
            logger.error(f"Failed to record move: {e}")
            return False
            
    def undo_last_operation(self):
        """Undo the last file operation"""
        if not self.operations:
            return False, "No operations to undo"
            
        try:
            operation = self.operations.pop()
            
            if operation['type'] == 'delete':
                # Create directory if it doesn't exist
                original_dir = os.path.dirname(operation['original_path'])
                if not os.path.exists(original_dir):
                    os.makedirs(original_dir)
                    
                # Restore file from backup
                shutil.copy2(operation['backup_path'], operation['original_path'])
                os.remove(operation['backup_path'])
                return True, f"Restored: {os.path.basename(operation['original_path'])}"

            if operation['type'] == 'move':
                original_path = operation['original_path']
                new_path = operation['new_path']
                if not os.path.exists(new_path):
                    return False, f"Moved file not found: {os.path.basename(new_path)}"

                original_dir = os.path.dirname(original_path)
                if not os.path.exists(original_dir):
                    os.makedirs(original_dir)

                restore_path = original_path
                if os.path.exists(restore_path):
                    base, ext = os.path.splitext(original_path)
                    counter = 1
                    while os.path.exists(restore_path):
                        restore_path = f"{base}_restored_{counter}{ext}"
                        counter += 1

                shutil.move(new_path, restore_path)
                return True, f"Restored: {os.path.basename(restore_path)}"
                
        except Exception as e:
            return False, f"Undo failed: {e}"
            
        return False, "Unknown operation type"
        
    def get_recent_operations(self, limit=10):
        """Get recent operations for display"""
        return self.operations[-limit:] if self.operations else []
    
    def get_undo_count(self):
        """Get number of available undo operations"""
        return len(self.operations)
    
    def clear_backups(self):
        """Clear all backups and operations"""
        for op in self.operations:
            try:
                if op.get('backup_path') and os.path.exists(op['backup_path']):
                    os.remove(op['backup_path'])
            except Exception:
                pass
        self.operations.clear()


# Standalone hash function for multiprocessing (must be at module level)
def compute_file_hash_worker(args):
    """Worker function for parallel hash computation"""
    file_path, hash_type = args
    try:
        return (file_path, compute_exact_hash(file_path, hash_type or "MD5"))
    except Exception as e:
        return (file_path, None)


def compute_perceptual_hash_worker(file_path):
    """Worker function for perceptual hash computation (GPU-accelerated when available)"""
    try:
        if not HAS_PILLOW:
            return (file_path, None)
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            return (file_path, None)
        
        img = Image.open(file_path)
        
        # GPU-accelerated perceptual hashing
        if HAS_CUDA and CUDA_BACKEND == 'cupy':
            # Use CuPy for GPU acceleration
            img_gray = img.convert('L').resize((32, 32))
            img_array = np.array(img_gray, dtype=np.float32)
            gpu_array = cp.asarray(img_array)
            
            # GPU-based DCT approximation using FFT
            fft_result = cp.fft.fft2(gpu_array)
            dct_approx = cp.real(fft_result)[:8, :8]
            median_val = cp.median(dct_approx)
            binary_hash = (dct_approx > median_val).flatten()
            hash_str = ''.join('1' if b else '0' for b in cp.asnumpy(binary_hash))
            return (file_path, hash_str)
            
        elif HAS_CUDA and CUDA_BACKEND == 'torch':
            # Use PyTorch CUDA for GPU acceleration
            img_gray = img.convert('L').resize((32, 32))
            img_array = np.array(img_gray, dtype=np.float32)
            gpu_tensor = torch_module.from_numpy(img_array).cuda()
            
            # GPU-based FFT for DCT approximation
            fft_result = torch_module.fft.fft2(gpu_tensor)
            dct_approx = torch_module.real(fft_result)[:8, :8]
            median_val = torch_module.median(dct_approx)
            binary_hash = (dct_approx > median_val).flatten()
            hash_str = ''.join('1' if b else '0' for b in binary_hash.cpu().numpy())
            return (file_path, hash_str)
        else:
            # CPU fallback using imagehash
            phash = imagehash.phash(img)
            return (file_path, str(phash))
            
    except Exception as e:
        return (file_path, None)


class GPUHasher:
    """GPU-accelerated batch hash computation"""
    
    def __init__(self):
        self.cuda_available = HAS_CUDA
        self.backend = CUDA_BACKEND
        
    def batch_hash_images_gpu(self, image_paths, batch_size=32):
        """Compute perceptual hashes for batch of images using GPU"""
        if not self.cuda_available or not HAS_PILLOW:
            return self._batch_hash_cpu(image_paths)
        
        results = {}
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                batch_results = self._process_batch_gpu(batch)
                results.update(batch_results)
                
                # Clear GPU memory periodically
                if i % (batch_size * 4) == 0:
                    self._clear_gpu_memory()
                    
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}, falling back to CPU")
            return self._batch_hash_cpu(image_paths)
        
        return results
    
    def _process_batch_gpu(self, image_paths):
        """Process batch of images on GPU"""
        results = {}
        gpu_images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                if not os.path.exists(path):
                    results[path] = None
                    continue
                
                ext = os.path.splitext(path)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                    results[path] = None
                    continue
                
                img = Image.open(path).convert('L').resize((32, 32))
                img_array = np.array(img, dtype=np.float32)
                
                if self.backend == 'cupy':
                    gpu_img = cp.asarray(img_array)
                elif self.backend == 'torch':
                    gpu_img = torch_module.from_numpy(img_array).cuda()
                else:
                    continue
                    
                gpu_images.append(gpu_img)
                valid_paths.append(path)
                
            except Exception as e:
                results[path] = None
        
        if not gpu_images:
            return results
        
        try:
            if self.backend == 'cupy':
                batch_tensor = cp.stack(gpu_images)
                fft_results = cp.fft.fft2(batch_tensor)
                dct_approx = cp.real(fft_results)[:, :8, :8]
                
                for idx, path in enumerate(valid_paths):
                    img_dct = dct_approx[idx]
                    median_val = cp.median(img_dct)
                    binary_hash = (img_dct > median_val).flatten()
                    hash_str = ''.join('1' if b else '0' for b in cp.asnumpy(binary_hash))
                    results[path] = hash_str
                    
            elif self.backend == 'torch':
                batch_tensor = torch_module.stack(gpu_images)
                fft_results = torch_module.fft.fft2(batch_tensor)
                dct_approx = torch_module.real(fft_results)[:, :8, :8]
                
                for idx, path in enumerate(valid_paths):
                    img_dct = dct_approx[idx]
                    median_val = torch_module.median(img_dct)
                    binary_hash = (img_dct > median_val).flatten()
                    hash_str = ''.join('1' if b else '0' for b in binary_hash.cpu().numpy())
                    results[path] = hash_str
                    
        except Exception as e:
            logger.error(f"GPU batch error: {e}")
            for path in valid_paths:
                if path not in results:
                    results[path] = None
        
        return results
    
    def _batch_hash_cpu(self, image_paths):
        """CPU fallback for batch hashing"""
        results = {}
        for path in image_paths:
            try:
                if HAS_PILLOW:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                        img = Image.open(path)
                        phash = imagehash.phash(img)
                        results[path] = str(phash)
                    else:
                        results[path] = None
                else:
                    results[path] = None
            except Exception:
                results[path] = None
        return results
    
    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        try:
            if self.backend == 'cupy':
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            elif self.backend == 'torch':
                torch_module.cuda.empty_cache()
        except Exception:
            pass


class EnhancedScanThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict, dict)  # duplicates, similar_files
    status_update = pyqtSignal(str)
    current_file = pyqtSignal(str)  # Current file being processed
    eta_update = pyqtSignal(str)  # ETA update
    files_processed = pyqtSignal(int, int)  # processed, total
    gpu_status = pyqtSignal(str)  # GPU status updates
    
    def __init__(self, directory, comparison_settings, scan_filters=None, use_parallel=True, use_gpu=True):
        super().__init__()
        self.directory = directory
        self.comparison_settings = comparison_settings
        self.scan_filters = scan_filters or ScanFilters()
        self.is_paused = False
        self.is_cancelled = False
        self.start_time = None
        self.use_parallel = use_parallel
        self.use_gpu = use_gpu and HAS_CUDA
        self.gpu_hasher = GPUHasher() if self.use_gpu else None
        self.media_profiles = merge_media_profiles(self.comparison_settings.get("media_profiles"))
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
    
    def cancel(self):
        self.is_cancelled = True

    def _pause_or_cancel_requested(self) -> bool:
        """Yield while paused and report whether work should stop."""
        while self.is_paused and not self.is_cancelled:
            time.sleep(0.1)
        return self.is_cancelled
        
    def run(self):
        self.start_time = time.time()
        
        # Report GPU status
        if self.use_gpu:
            self.gpu_status.emit(f"🚀 GPU acceleration enabled ({CUDA_BACKEND.upper()})")
        else:
            self.gpu_status.emit("💻 Using CPU processing")
        
        self.status_update.emit("Collecting files...")
        
        # Collect all files with filters
        files = []
        for root, dirs, filenames in os.walk(self.directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.scan_filters.excluded_folders]
            
            for filename in filenames:
                if self.is_cancelled:
                    return
                file_path = os.path.join(root, filename)
                if self.scan_filters.passes_filter(file_path):
                    files.append(file_path)
        
        total_files = len(files)
        self.status_update.emit(f"Found {total_files} files. Analyzing with {'GPU' if self.use_gpu else 'parallel'} processing...")
        self.files_processed.emit(0, total_files)
        
        # Phase 1: Find exact duplicates by hash using parallel processing
        try:
            if self.use_parallel and total_files > 100:
                file_hashes = self._parallel_hash_computation(files, total_files)
            else:
                file_hashes = self._sequential_hash_computation(files, total_files)
        except Exception as e:
            logger.error(f"Hash computation error: {e}")
            import traceback
            traceback.print_exc()
            file_hashes = {}
        
        if self.is_cancelled:
            return
        
        # Ensure file_hashes is not None
        if file_hashes is None:
            logger.warning("Warning: file_hashes is None, using empty dict")
            file_hashes = {}
            
        duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}
        
        # Phase 2: Find similar files (not exact duplicates)
        self.status_update.emit("Finding similar files...")
        similar_files = {}
        
        if self.comparison_settings['find_similar']:
            try:
                similar_files = self.find_similar_files(files, duplicates)
            except MemoryError:
                logger.error("Memory exhausted during similarity scan — results are partial")
                self.status_update.emit("Warning: Memory exhausted; similarity results may be incomplete")
            except Exception as e:
                logger.error("Similarity scan crashed (%s): %s", type(e).__name__, e, exc_info=True)
                self.status_update.emit(f"Warning: Similarity scan stopped ({type(e).__name__})")
        
        self.progress.emit(100)
        elapsed = time.time() - self.start_time
        self.status_update.emit(f"Analysis complete in {elapsed:.1f}s!")
        self.finished.emit(duplicates, similar_files)
    
    def _parallel_hash_computation(self, files, total_files):
        """Compute hashes using parallel processing with thread pool"""
        file_hashes = defaultdict(list)
        hash_type = self.comparison_settings.get('hash_type', 'MD5')
        
        # Separate image, video, and audio files for specialized processing
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
        audio_extensions = {'.mp3', '.flac', '.wav', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff', '.ape', '.alac', '.wv'}
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log', '.ini', '.cfg', '.yaml', '.yml'}
        
        image_files = []
        video_files = []
        audio_files = []
        text_files = []
        other_files = []
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            # Perceptual mode: images and videos
            if ext in image_extensions and hash_type == 'Perceptual':
                image_files.append(f)
            elif ext in video_extensions and hash_type == 'Perceptual' and HAS_VIDEO_FINGERPRINTER:
                video_files.append(f)
            # Audio mode: audio files with fingerprinting
            elif ext in audio_extensions and hash_type == 'Audio' and HAS_AUDIO_FINGERPRINTER:
                audio_files.append(f)
            # Semantic mode: text files
            elif ext in text_extensions and hash_type == 'Semantic' and HAS_SEMANTIC_SIMILARITY:
                text_files.append(f)
            else:
                other_files.append(f)
        
        processed = 0
        
        # GPU batch processing for images if perceptual hashing
        if self.use_gpu and image_files and hash_type == 'Perceptual':
            self.status_update.emit(f"GPU processing {len(image_files)} images...")
            gpu_results = self.gpu_hasher.batch_hash_images_gpu(image_files, batch_size=64)
            
            for path, hash_val in gpu_results.items():
                if self.is_cancelled:
                    return file_hashes
                if hash_val:
                    file_hashes[hash_val].append(path)
                processed += 1
                if processed % 100 == 0:
                    progress = int((processed / total_files) * 50)
                    self.progress.emit(progress)
                    self.files_processed.emit(processed, total_files)
        
        # GPU video fingerprinting if available
        if video_files and HAS_VIDEO_FINGERPRINTER and hash_type == 'Perceptual':
            self.status_update.emit(f"🎬 GPU processing {len(video_files)} videos...")
            self.gpu_status.emit(f"🎬 Video fingerprinting: {len(video_files)} files")
            
            try:
                # Use batch video fingerprinting
                video_results = video_fingerprinter.batch_fingerprint_videos(
                    video_files, 
                    num_keyframes=8,
                    combine_method="xor"  # XOR method for efficient comparison
                )
                
                for path, fingerprint in video_results.items():
                    if self.is_cancelled:
                        return file_hashes
                    if fingerprint:
                        file_hashes[fingerprint].append(path)
                    processed += 1
                    self.current_file.emit(os.path.basename(path))
                    
                    if processed % 10 == 0:
                        progress = int((processed / total_files) * 50)
                        self.progress.emit(progress)
                        self.files_processed.emit(processed, total_files)
                        
            except Exception as e:
                logger.error(f"Video fingerprinting error: {e}")
                # Fall back to adding videos to other_files for regular hashing
                other_files.extend(video_files)
        
        # GPU audio fingerprinting if available
        if audio_files and HAS_AUDIO_FINGERPRINTER and hash_type in ('Audio', 'Perceptual'):
            self.status_update.emit(f"🎵 Processing {len(audio_files)} audio files (GPU MFCC + Chromaprint)...")
            self.gpu_status.emit(f"🎵 Audio fingerprinting: {len(audio_files)} files")
            
            try:
                # HP-2: Run fingerprint computation in parallel across multiple CPU cores.
                # Fingerprinting is CPU-bound (audio decode + MFCC), so multiple workers
                # give a near-linear speed-up up to the number of physical cores.
                # HP-3: Results are saved to self._phase1_audio_fp_cache so Phase 2
                # can reuse them and skip a second full decode pass.
                from concurrent.futures import ThreadPoolExecutor as _TPool, as_completed as _asc
                _n_workers = min(4, os.cpu_count() or 2)
                _phase1_cache: dict = {}

                with _TPool(max_workers=_n_workers) as _fp_exec:
                    _futures = {
                        _fp_exec.submit(
                            audio_fingerprinter.compute_fingerprint, path, "hybrid"
                        ): path
                        for path in audio_files
                    }
                    for _fut in _asc(_futures):
                        path = _futures[_fut]
                        if self.is_cancelled:
                            break
                        try:
                            fingerprint = _fut.result()
                        except Exception:
                            fingerprint = None

                        _phase1_cache[path] = fingerprint
                        self.current_file.emit(os.path.basename(path))

                        if fingerprint:
                            # Create a unique hash from the fingerprint
                            # Combine chromaprint and mfcc for a robust identifier
                            fp_parts = []
                            if fingerprint.get('chromaprint'):
                                fp_parts.append(f"cp:{fingerprint['chromaprint'][:32]}")
                            if fingerprint.get('mfcc'):
                                # Use first few MFCC values as a hash
                                mfcc = fingerprint['mfcc']
                                if isinstance(mfcc, list) and len(mfcc) >= 4:
                                    mfcc_hash = "_".join(f"{v:.2f}" for v in mfcc[:4])
                                    fp_parts.append(f"mfcc:{mfcc_hash}")

                            if fp_parts:
                                audio_hash = f"audio:{'|'.join(fp_parts)}"
                                file_hashes[audio_hash].append(path)

                        processed += 1
                        progress = int((processed / total_files) * 50)
                        self.progress.emit(progress)
                        self.files_processed.emit(processed, total_files)

                # Make Phase 1 fingerprints available to find_similar_files (HP-3)
                self._phase1_audio_fp_cache = _phase1_cache
                        
            except Exception as e:
                logger.error(f"Audio fingerprinting error: {e}")
                # Fall back to adding audio to other_files for regular hashing
                other_files.extend(audio_files)
        
        # Semantic text processing if available
        if text_files and HAS_SEMANTIC_SIMILARITY and hash_type == 'Semantic':
            self.status_update.emit(f"📚 Processing {len(text_files)} text files (GPU semantic embeddings)...")
            self.gpu_status.emit(f"📚 Semantic analysis: {len(text_files)} files")
            
            try:
                # Read text content and compute embeddings
                text_contents = {}
                for path in text_files:
                    if self.is_cancelled:
                        break
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:10000]  # Limit to first 10K chars
                            if content.strip():
                                text_contents[path] = content
                    except Exception:
                        pass
                
                if text_contents:
                    # Compute embeddings in batch
                    paths = list(text_contents.keys())
                    texts = [text_contents[p] for p in paths]
                    
                    embeddings = semantic_engine.compute_embeddings(texts)
                    
                    if embeddings is not None:
                        # Create hash from first few embedding dimensions
                        for i, path in enumerate(paths):
                            if self.is_cancelled:
                                return file_hashes
                            
                            emb = embeddings[i]
                            # Quantize embedding to create a "hash"
                            emb_hash = "_".join(f"{v:.2f}" for v in emb[:8])
                            semantic_hash = f"semantic:{emb_hash}"
                            file_hashes[semantic_hash].append(path)
                            
                            processed += 1
                            self.current_file.emit(os.path.basename(path))
                            
                            if processed % 10 == 0:
                                progress = int((processed / total_files) * 50)
                                self.progress.emit(progress)
                                self.files_processed.emit(processed, total_files)
                                
            except Exception as e:
                logger.error(f"Semantic processing error: {e}")
                other_files.extend(text_files)
        
        # Parallel processing for other files using ThreadPoolExecutor
        mode_str = "GPU mode" if HAS_CUDA else "CPU mode"
        self.status_update.emit(f"Parallel processing {len(other_files)} files with {OPTIMAL_WORKERS} workers ({mode_str})...")
        
        hash_tasks = [
            (f, get_profile_for_path(f, self.media_profiles).get("exact_hash", hash_type))
            for f in other_files
        ]

        with ThreadPoolExecutor(max_workers=OPTIMAL_WORKERS) as executor:
            # Submit all hash computation tasks
            futures = {
                executor.submit(compute_file_hash_worker, task): task[0]
                for task in hash_tasks
            }
            
            for future in as_completed(futures):
                if self.is_cancelled:
                    executor.shutdown(wait=False)
                    return file_hashes
                
                # Handle pause
                while self.is_paused:
                    time.sleep(0.1)
                    if self.is_cancelled:
                        executor.shutdown(wait=False)
                        return file_hashes
                
                try:
                    file_path, file_hash = future.result()
                    if file_hash:
                        file_hashes[file_hash].append(file_path)
                    
                    processed += 1
                    self.current_file.emit(os.path.basename(file_path))
                    
                    if processed % 50 == 0:
                        progress = int((processed / total_files) * 50)
                        self.progress.emit(progress)
                        self.files_processed.emit(processed, total_files)
                        
                        # Calculate ETA
                        elapsed = time.time() - self.start_time
                        if processed > 0:
                            rate = processed / elapsed
                            remaining = (total_files - processed) / rate if rate > 0 else 0
                            eta = timedelta(seconds=int(remaining))
                            self.eta_update.emit(f"ETA: {str(eta)}")
                            
                except Exception as e:
                    logger.error(f"Hash computation error: {e}")
        
        return file_hashes
    
    def _sequential_hash_computation(self, files, total_files):
        """Sequential hash computation for smaller file sets"""
        file_hashes = defaultdict(list)
        
        for i, file_path in enumerate(files):
            # Handle pause/cancel
            while self.is_paused:
                time.sleep(0.1)
                if self.is_cancelled:
                    return file_hashes
            if self.is_cancelled:
                return file_hashes
            
            try:
                # Emit current file
                self.current_file.emit(os.path.basename(file_path))
                
                file_hash = self.compute_hash(file_path)
                if file_hash:
                    file_hashes[file_hash].append(file_path)
                
                progress = int((i + 1) / total_files * 50)  # First 50% for hashing
                self.progress.emit(progress)
                self.files_processed.emit(i + 1, total_files)
                
                # Calculate and emit ETA
                if i > 0:
                    elapsed = time.time() - self.start_time
                    rate = (i + 1) / elapsed
                    remaining = (total_files - i - 1) / rate if rate > 0 else 0
                    eta = timedelta(seconds=int(remaining))
                    self.eta_update.emit(f"ETA: {str(eta)}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        return file_hashes
    
    def compute_hash(self, file_path):
        try:
            profile = get_profile_for_path(file_path, self.media_profiles)
            exact_hash = profile.get("exact_hash")
            if exact_hash:
                return compute_exact_hash(file_path, exact_hash)

            hash_type = self.comparison_settings.get('hash_type', 'MD5')
            if hash_type == "SHA256":
                return self.sha256_hash(file_path)
            return self.md5_hash(file_path)
        except Exception:
            return None
    
    def md5_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def sha256_hash(self, file_path):
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def perceptual_hash(self, file_path):
        if not HAS_PILLOW:
            return self.md5_hash(file_path)
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            try:
                with Image.open(file_path) as img:
                    return str(imagehash.phash(img))
            except Exception:
                return self.md5_hash(file_path)
        else:
            return self.md5_hash(file_path)
    
    def find_similar_files(self, files, duplicates):
        """Find similar (but not identical) files"""
        similar_files = defaultdict(list)
        
        # Get list of files that are not already exact duplicates
        duplicate_files = set()
        for file_list in duplicates.values():
            duplicate_files.update(file_list)
        
        non_duplicate_files = [f for f in files if f not in duplicate_files]
        
        settings = self.comparison_settings
        hash_type = settings.get('hash_type', 'MD5')
        total_comparisons = len(non_duplicate_files) * (len(non_duplicate_files) - 1) // 2
        comparison_count = 0

        # Define extension sets once outside the O(n²) loop (M-5 fix)
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
        audio_extensions = {'.mp3', '.flac', '.wav', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff', '.ape', '.alac', '.wv'}
        enable_audio_similarity = hash_type == 'Audio' and HAS_AUDIO_FINGERPRINTER
        enable_video_similarity = hash_type == 'Perceptual' and HAS_VIDEO_FINGERPRINTER

        # AF-3: Pre-compute fingerprints ONCE per unique file before the O(n²) loop.
        # Without this, with N audio/video files, each file's fingerprint is recomputed
        # N-1 times (O(n²) total decoder calls). With this cache it's exactly O(n).
        # HP-3: Seed audio_fp_cache from Phase 1 results when hash_type == 'Audio',
        # so files already decoded in Phase 1 are never decoded a second time.
        audio_fp_cache: dict = dict(getattr(self, '_phase1_audio_fp_cache', {}))
        video_fp_cache: dict = {}
        audio_candidates = []
        video_candidates = []

        if enable_audio_similarity:
            # Only queue files whose fingerprint was not carried over from Phase 1
            audio_candidates = [
                f for f in non_duplicate_files
                if os.path.splitext(f)[1].lower() in audio_extensions
                and f not in audio_fp_cache
            ]
        if enable_video_similarity:
            video_candidates = [
                f for f in non_duplicate_files
                if os.path.splitext(f)[1].lower() in video_extensions
            ]

        total_precompute = len(audio_candidates) + len(video_candidates)
        precomputed = 0
        phase2_start = 60 if total_precompute else 50
        phase2_span = 35 if total_precompute else 45

        if enable_audio_similarity and audio_candidates:
            self.status_update.emit(
                f"Computing audio fingerprints ({len(audio_candidates)} files)…"
            )
            for _idx, _af in enumerate(audio_candidates, start=1):
                if self._pause_or_cancel_requested():
                    return dict(similar_files)
                audio_fp_cache[_af] = audio_fingerprinter.compute_fingerprint(
                    _af, method="hybrid"
                )
                precomputed += 1
                progress = 50 + int((precomputed / total_precompute) * 10)
                self.progress.emit(min(progress, 60))

        if enable_video_similarity and video_candidates:
            self.status_update.emit(
                f"Computing video fingerprints ({len(video_candidates)} files)…"
            )
            for _idx, _vf in enumerate(video_candidates, start=1):
                if self._pause_or_cancel_requested():
                    return dict(similar_files)
                video_fp_cache[_vf] = video_fingerprinter.compute_video_fingerprint(
                    _vf, num_keyframes=5
                )
                precomputed += 1
                progress = 50 + int((precomputed / total_precompute) * 10)
                self.progress.emit(min(progress, 60))

        # AF-4: Pre-compute per-file derived properties — extension, media type, and
        # profile only depend on the file path yet were being computed O(n²) times inside
        # the inner loop.  Building these dicts once reduces that to O(n).
        ext_cache = {f: os.path.splitext(f)[1].lower() for f in non_duplicate_files}
        media_type_cache = {f: get_media_type(f) for f in non_duplicate_files}
        profile_cache = {f: get_profile_for_path(f, self.media_profiles) for f in non_duplicate_files}

        for i, file1 in enumerate(non_duplicate_files):
            if self._pause_or_cancel_requested():
                return dict(similar_files)
            # AF-4: file1 properties only depend on file1 — constant across all j
            # pairings. Compute once per outer iteration, not once per inner iteration.
            ext1 = ext_cache[file1]
            media_type1 = media_type_cache[file1]
            profile1 = profile_cache[file1]
            for j, file2 in enumerate(non_duplicate_files[i+1:], i+1):
                if self._pause_or_cancel_requested():
                    return dict(similar_files)
                comparison_count += 1

                # Update progress (second 50%)
                if comparison_count % 100 == 0:
                    if total_comparisons:
                        progress = phase2_start + int((comparison_count / total_comparisons) * phase2_span)
                    else:
                        progress = 95
                    self.progress.emit(min(progress, 95))

                is_similar = False
                similarity_reasons = []
                name_similar = False
                size_similar = False
                media_type2 = media_type_cache[file2]
                shared_media_type = media_type1 if media_type1 == media_type2 else None
                profile = profile1 if shared_media_type else None
                # ext1 is pre-fetched in the outer loop (constant for all j)
                ext2 = ext_cache[file2]

                if shared_media_type and profile and profile.get("enable_fuzzy", False):
                    if shared_media_type == "images":
                        try:
                            image_match, image_score, image_reasons = compare_image_files(file1, file2, profile)
                            if image_match:
                                is_similar = True
                                similarity_reasons.extend(image_reasons or [f"Similar image content ({image_score:.0%})"])
                        except Exception:
                            pass
                    elif shared_media_type in {"documents", "code"}:
                        try:
                            doc_match, doc_score, doc_reasons = compare_document_files(
                                file1,
                                file2,
                                profile,
                                semantic_engine=semantic_engine if HAS_SEMANTIC_SIMILARITY else None,
                            )
                            if doc_match:
                                is_similar = True
                                similarity_reasons.extend(doc_reasons or [f"Similar document content ({doc_score:.0%})"])
                        except Exception:
                            pass
                    elif shared_media_type == "archives":
                        try:
                            archive_match, archive_score, archive_reasons = compare_archive_files(file1, file2, profile)
                            if archive_match:
                                is_similar = True
                                similarity_reasons.extend(archive_reasons or [f"Similar archive contents ({archive_score:.0%})"])
                        except Exception:
                            pass

                if (
                    ext1 in video_extensions
                    and ext2 in video_extensions
                    and enable_video_similarity
                    and shared_media_type == "videos"
                    and profile
                    and profile.get("enable_fuzzy", False)
                ):
                    try:
                        fp1 = video_fp_cache.get(file1)
                        fp2 = video_fp_cache.get(file2)
                        
                        if fp1 and fp2:
                            is_sim, similarity, details = video_fingerprinter.are_videos_similar(
                                fp1, fp2, threshold=0.7
                            )
                            if is_sim:
                                is_similar = True
                                similarity_reasons.append(f"Similar video content ({similarity:.0%})")
                    except Exception:
                        pass  # Fall through to other comparison methods
                
                # Check for similar audio using audio fingerprinting
                elif (
                    ext1 in audio_extensions
                    and ext2 in audio_extensions
                    and enable_audio_similarity
                    and shared_media_type == "audio"
                    and profile
                    and profile.get("enable_fuzzy", False)
                ):
                    try:
                        fp1 = audio_fp_cache.get(file1)
                        fp2 = audio_fp_cache.get(file2)
                        
                        if fp1 and fp2:
                            is_sim, similarity, details = audio_fingerprinter.compare_fingerprints(
                                fp1, fp2, method="hybrid"
                            )
                            # AS-2: Trust the engine's determination only — no secondary
                            # low-score fallback.  is_sim=True requires chromaprint≥0.92,
                            # MFCC≥0.90, or hybrid-combined≥0.88 after a 3%-duration gate.
                            if is_sim:
                                is_similar = True
                                method = details.get('method_used', 'hybrid')
                                similarity_reasons.append(f"Similar audio ({similarity:.0%} via {method})")
                    except Exception:
                        pass  # Fall through to other comparison methods
                
                # Check various similarity criteria.
                # AS-2: Skip all name/size/text comparisons for audio pairs — those
                # signals are too noisy for music libraries (same song title ≠ same
                # recording; different bitrate ≠ different song).  Only the acoustic
                # fingerprint (above) is authoritative for audio.
                if shared_media_type != "audio":
                    if settings.get('compare_by_name', False):
                        name_similar = FileComparator.compare_by_name(file1, file2, settings.get('name_similarity', 0.8))
                        if name_similar:
                            is_similar = True
                            similarity_reasons.append("Similar names")

                    if settings.get('compare_by_size', False):
                        size_similar = FileComparator.compare_by_size(file1, file2, settings.get('size_tolerance', 1))
                        if size_similar:
                            is_similar = True
                            similarity_reasons.append("Similar sizes")

                    # Same name but different size (explicit step)
                    if name_similar and not size_similar:
                        try:
                            name1 = os.path.splitext(os.path.basename(file1))[0].lower()
                            name2 = os.path.splitext(os.path.basename(file2))[0].lower()
                            if name1 == name2:
                                size1 = os.path.getsize(file1)
                                size2 = os.path.getsize(file2)
                                if size1 != size2:
                                    is_similar = True
                                    similarity_reasons.append("Same name, different size")
                        except Exception:
                            pass

                    if settings.get('compare_text_content', False):
                        if FileComparator.compare_text_similarity(file1, file2, settings.get('content_similarity', 0.9)):
                            is_similar = True
                            similarity_reasons.append("Similar content")
                
                if is_similar:
                    match_type = "name_size"
                    if "Same name, different size" in similarity_reasons:
                        match_type = "same_name_diff_size"
                    if any(
                        reason.startswith(("Similar audio", "Similar video", "Similar content", "lexical ", "archive ", "phash "))
                        for reason in similarity_reasons
                    ):
                        match_type = "content"
                    key = f"similar_{len(similar_files)}"
                    similar_files[key] = {
                        'files': [file1, file2],
                        'reasons': similarity_reasons,
                        'match_type': match_type
                    }
        
        return dict(similar_files)


class EnterpriseConfigDialog(QDialog):
    """Configuration dialog for advanced features"""
    
    def __init__(self, parent=None, current_config=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Configuration")
        self.setModal(True)
        self.resize(550, 650)
        
        self.config = current_config or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Configure Advanced Features")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(header)
        
        # Status label
        if HAS_ENTERPRISE_MODULES:
            status = QLabel("Advanced modules available")
            status.setStyleSheet("color: #4CAF50; padding: 5px;")
        else:
            status = QLabel("Advanced modules not installed - some features are unavailable")
            status.setStyleSheet("color: #ff9800; padding: 5px;")
        layout.addWidget(status)
        
        # Create tab widget for different configuration sections
        tabs = QTabWidget()
        
        # Database settings tab
        db_tab = QWidget()
        db_layout = QVBoxLayout(db_tab)
        db_layout.setSpacing(10)
        
        self.use_database_cb = QCheckBox("Enable database persistence (cache file hashes)")
        self.use_database_cb.setChecked(self.config.get("use_database", True))
        self.use_database_cb.setEnabled(HAS_ENTERPRISE_MODULES)
        db_layout.addWidget(self.use_database_cb)
        
        self.incremental_scan_cb = QCheckBox("Enable incremental scanning (skip unchanged files)")
        self.incremental_scan_cb.setChecked(self.config.get("incremental_scan", True))
        self.incremental_scan_cb.setEnabled(HAS_ENTERPRISE_MODULES)
        db_layout.addWidget(self.incremental_scan_cb)
        
        db_path_layout = QHBoxLayout()
        db_path_layout.addWidget(QLabel("Database path:"))
        self.db_path_edit = QLineEdit(self.config.get("db_path", "duplicates.db"))
        self.db_path_edit.setEnabled(HAS_ENTERPRISE_MODULES)
        db_path_layout.addWidget(self.db_path_edit)
        db_layout.addLayout(db_path_layout)
        
        db_layout.addStretch()
        tabs.addTab(db_tab, "💾 Database")
        
        # GPU settings tab
        gpu_tab = QWidget()
        gpu_layout = QVBoxLayout(gpu_tab)
        gpu_layout.setSpacing(10)
        
        gpu_status = QLabel(f"GPU Status: {'✅ CUDA Available (' + CUDA_BACKEND + ')' if HAS_CUDA else '❌ CPU Only'}")
        gpu_status.setStyleSheet(f"color: {'#4CAF50' if HAS_CUDA else '#ff5722'}; font-weight: bold;")
        gpu_layout.addWidget(gpu_status)
        
        self.use_gpu_cb = QCheckBox("Enable GPU acceleration for image processing")
        self.use_gpu_cb.setChecked(self.config.get("use_gpu", HAS_CUDA))
        self.use_gpu_cb.setEnabled(HAS_CUDA)
        gpu_layout.addWidget(self.use_gpu_cb)
        
        self.use_gpu_hashing_cb = QCheckBox("Use GPU for perceptual image hashing")
        self.use_gpu_hashing_cb.setChecked(self.config.get("use_gpu_hashing", HAS_CUDA))
        self.use_gpu_hashing_cb.setEnabled(HAS_CUDA)
        gpu_layout.addWidget(self.use_gpu_hashing_cb)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("GPU batch size:"))
        self.gpu_batch_size = QSpinBox()
        self.gpu_batch_size.setRange(1, 1000)
        self.gpu_batch_size.setValue(self.config.get("gpu_batch_size", 32))
        self.gpu_batch_size.setEnabled(HAS_CUDA)
        batch_layout.addWidget(self.gpu_batch_size)
        batch_layout.addStretch()
        gpu_layout.addLayout(batch_layout)
        
        gpu_layout.addStretch()
        tabs.addTab(gpu_tab, "🎮 GPU")
        
        # Plugin settings tab
        plugin_tab = QWidget()
        plugin_layout = QVBoxLayout(plugin_tab)
        plugin_layout.setSpacing(10)
        
        self.use_plugins_cb = QCheckBox("Enable plugin system for extensibility")
        self.use_plugins_cb.setChecked(self.config.get("use_plugins", True))
        self.use_plugins_cb.setEnabled(HAS_ENTERPRISE_MODULES)
        plugin_layout.addWidget(self.use_plugins_cb)
        
        plugin_dir_layout = QHBoxLayout()
        plugin_dir_layout.addWidget(QLabel("Plugin directory:"))
        self.plugin_dir_edit = QLineEdit(self.config.get("plugin_dir", "plugins"))
        self.plugin_dir_edit.setEnabled(HAS_ENTERPRISE_MODULES)
        plugin_dir_layout.addWidget(self.plugin_dir_edit)
        plugin_layout.addLayout(plugin_dir_layout)
        
        plugin_layout.addStretch()
        tabs.addTab(plugin_tab, "🔌 Plugins")
        
        # Memory optimization tab
        memory_tab = QWidget()
        memory_layout = QVBoxLayout(memory_tab)
        memory_layout.setSpacing(10)
        
        self.optimize_memory_cb = QCheckBox("Enable memory optimization for large files")
        self.optimize_memory_cb.setChecked(self.config.get("optimize_memory", True))
        self.optimize_memory_cb.setEnabled(HAS_ENTERPRISE_MODULES)
        memory_layout.addWidget(self.optimize_memory_cb)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Large file threshold (MB):"))
        self.large_file_threshold = QSpinBox()
        self.large_file_threshold.setRange(1, 10000)
        self.large_file_threshold.setValue(self.config.get("large_file_threshold", 100))
        self.large_file_threshold.setEnabled(HAS_ENTERPRISE_MODULES)
        threshold_layout.addWidget(self.large_file_threshold)
        threshold_layout.addStretch()
        memory_layout.addLayout(threshold_layout)
        
        memory_layout.addStretch()
        tabs.addTab(memory_tab, "🧠 Memory")
        
        # Merge settings tab
        merge_tab = QWidget()
        merge_layout = QVBoxLayout(merge_tab)
        merge_layout.setSpacing(10)
        
        self.enable_merging_cb = QCheckBox("Enable file merging analysis")
        self.enable_merging_cb.setChecked(self.config.get("enable_merging", True))
        self.enable_merging_cb.setEnabled(HAS_ENTERPRISE_MODULES)
        merge_layout.addWidget(self.enable_merging_cb)
        
        self.analyze_merges_cb = QCheckBox("Auto-analyze merge opportunities")
        self.analyze_merges_cb.setChecked(self.config.get("analyze_merges", False))
        self.analyze_merges_cb.setEnabled(HAS_ENTERPRISE_MODULES)
        merge_layout.addWidget(self.analyze_merges_cb)
        
        self.backup_before_merge_cb = QCheckBox("Create backups before merging")
        self.backup_before_merge_cb.setChecked(self.config.get("backup_before_merge", True))
        self.backup_before_merge_cb.setEnabled(HAS_ENTERPRISE_MODULES)
        merge_layout.addWidget(self.backup_before_merge_cb)
        
        merge_layout.addStretch()
        tabs.addTab(merge_tab, "🔀 Merging")
        
        # Undo settings tab
        undo_tab = QWidget()
        undo_layout = QVBoxLayout(undo_tab)
        undo_layout.setSpacing(10)
        
        self.enable_undo_cb = QCheckBox("Enable undo for file deletions (backup before delete)")
        self.enable_undo_cb.setChecked(self.config.get("enable_undo", True))
        undo_layout.addWidget(self.enable_undo_cb)
        
        max_undo_layout = QHBoxLayout()
        max_undo_layout.addWidget(QLabel("Maximum undo operations:"))
        self.max_undo_ops = QSpinBox()
        self.max_undo_ops.setRange(10, 1000)
        self.max_undo_ops.setValue(self.config.get("max_undo_operations", 100))
        max_undo_layout.addWidget(self.max_undo_ops)
        max_undo_layout.addStretch()
        undo_layout.addLayout(max_undo_layout)
        
        undo_layout.addStretch()
        tabs.addTab(undo_tab, "↩️ Undo")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_defaults)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def reset_defaults(self):
        """Reset all settings to defaults"""
        self.use_database_cb.setChecked(True)
        self.incremental_scan_cb.setChecked(True)
        self.db_path_edit.setText("duplicates.db")
        self.use_gpu_cb.setChecked(HAS_CUDA)
        self.use_gpu_hashing_cb.setChecked(HAS_CUDA)
        self.gpu_batch_size.setValue(32)
        self.use_plugins_cb.setChecked(True)
        self.plugin_dir_edit.setText("plugins")
        self.optimize_memory_cb.setChecked(True)
        self.large_file_threshold.setValue(100)
        self.enable_merging_cb.setChecked(True)
        self.analyze_merges_cb.setChecked(False)
        self.backup_before_merge_cb.setChecked(True)
        self.enable_undo_cb.setChecked(True)
        self.max_undo_ops.setValue(100)
        
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration from the dialog"""
        return {
            "use_database": self.use_database_cb.isChecked(),
            "incremental_scan": self.incremental_scan_cb.isChecked(),
            "db_path": self.db_path_edit.text(),
            "use_gpu": self.use_gpu_cb.isChecked(),
            "use_gpu_hashing": self.use_gpu_hashing_cb.isChecked(),
            "gpu_batch_size": self.gpu_batch_size.value(),
            "use_plugins": self.use_plugins_cb.isChecked(),
            "plugin_dir": self.plugin_dir_edit.text(),
            "optimize_memory": self.optimize_memory_cb.isChecked(),
            "large_file_threshold": self.large_file_threshold.value(),
            "enable_merging": self.enable_merging_cb.isChecked(),
            "analyze_merges": self.analyze_merges_cb.isChecked(),
            "backup_before_merge": self.backup_before_merge_cb.isChecked(),
            "enable_undo": self.enable_undo_cb.isChecked(),
            "max_undo_operations": self.max_undo_ops.value()
        }


class BulkMoveThread(QThread):
    progress = pyqtSignal(int, int)  # processed, total
    status = pyqtSignal(str)
    result = pyqtSignal(dict)

    def __init__(self, files_to_move, destination):
        super().__init__()
        self.files_to_move = files_to_move or []
        self.destination = destination
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.files_to_move)
        moved_count = 0
        failed_files = []
        moved_pairs = []
        space_freed = 0
        cancelled = False

        if not self.destination:
            self.result.emit({
                "moved_count": 0,
                "failed_files": ["No destination set."],
                "moved_pairs": [],
                "space_freed": 0,
                "cancelled": False
            })
            return

        try:
            os.makedirs(self.destination, exist_ok=True)
        except Exception as e:
            self.result.emit({
                "moved_count": 0,
                "failed_files": [f"Failed to create destination: {e}"],
                "moved_pairs": [],
                "space_freed": 0,
                "cancelled": False
            })
            return

        try:
            free_space = shutil.disk_usage(self.destination).free
        except Exception:
            free_space = None

        for idx, file_path in enumerate(self.files_to_move, 1):
            if self._cancelled:
                cancelled = True
                break

            self.progress.emit(idx, total)
            self.status.emit(f"Moving {os.path.basename(file_path)} ({idx}/{total})")

            try:
                if not os.path.exists(file_path):
                    failed_files.append(f"{file_path}: File not found")
                    continue

                file_size = 0
                try:
                    file_size = os.path.getsize(file_path)
                except Exception:
                    file_size = 0

                if free_space is not None and file_size > free_space:
                    failed_files.append(f"{file_path}: Insufficient disk space")
                    continue

                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.destination, filename)

                # Handle filename conflicts
                counter = 1
                base_name, ext = os.path.splitext(filename)
                while os.path.exists(dest_path):
                    new_filename = f"{base_name}_{counter}{ext}"
                    dest_path = os.path.join(self.destination, new_filename)
                    counter += 1

                shutil.move(file_path, dest_path)
                moved_count += 1
                moved_pairs.append((file_path, dest_path))
                space_freed += file_size

                if free_space is not None:
                    free_space -= file_size

            except PermissionError as e:
                failed_files.append(f"{file_path}: Permission denied ({e})")
            except Exception as e:
                failed_files.append(f"{file_path}: {e}")

        self.result.emit({
            "moved_count": moved_count,
            "failed_files": failed_files,
            "moved_pairs": moved_pairs,
            "space_freed": space_freed,
            "cancelled": cancelled
        })


class NoWheelChangeFilter(QObject):
    """Prevent mouse-wheel scrolling from changing form values."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(obj, event)


class EnhancedDuplicateFinder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.duplicates = {}
        self.similar_files = {}
        self.move_destination = None
        self.scan_filters = ScanFilters()
        self.scan_thread = None
        self.scan_start_time = None
        self.audio_quality_analyzer = AudioQualityAnalyzer()
        self.audio_extensions = {'.mp3', '.flac', '.wav', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff', '.ape', '.alac', '.wv'}
        self._quality_cache = {}
        
        # Settings persistence
        self.settings = QSettings("DuplicateFinder", "EnhancedDuplicateFinder")
        self.is_dark_theme = self.settings.value("dark_theme", True, type=bool)
        self.preset_enabled = self.settings.value("preset_enabled", True, type=bool)
        self.current_category_key = self.settings.value("preset_category", "general")
        self.current_preset_mode = self.settings.value("preset_mode", "accuracy")
        self.media_profiles = get_default_media_profiles()
        self.current_step_filter = "All"
        self.similar_step_filter = None
        self.step_sequence = ["Exact", "Content", "Name/Size"]
        self._suppress_hash_focus_feedback = False
        self._wheel_guard = NoWheelChangeFilter(self)

        saved_media_profiles = self.settings.value("media_profiles", None)
        if saved_media_profiles:
            try:
                self.media_profiles = merge_media_profiles(json.loads(saved_media_profiles))
            except Exception:
                self.media_profiles = get_default_media_profiles()
        
        # Advanced configuration
        self.enterprise_config = {
            "use_database": True,
            "incremental_scan": True,
            "db_path": "duplicates.db",
            "use_gpu": HAS_CUDA,
            "use_gpu_hashing": HAS_CUDA,
            "gpu_batch_size": 32,
            "use_plugins": True,
            "plugin_dir": "plugins",
            "optimize_memory": True,
            "large_file_threshold": 100,
            "enable_merging": True,
            "analyze_merges": False,
            "backup_before_merge": True,
            "enable_undo": True,
            "max_undo_operations": 100
        }
        
        # Load saved advanced config, keeping backward compatibility with
        # older enterprise_config saves.
        saved_config = self.settings.value("advanced_config", None)
        if saved_config is None:
            saved_config = self.settings.value("enterprise_config", None)
        if saved_config:
            try:
                import json
                self.enterprise_config.update(json.loads(saved_config))
            except Exception:
                pass
        
        # Initialize undo manager
        self.undo_manager = UndoManager()
        self.undo_manager.max_operations = self.enterprise_config.get("max_undo_operations", 100)
        
        # Initialize advanced components (if available)
        self.db_manager = None
        self.plugin_manager = None
        self.memory_manager = None
        self.file_merger = None
        self._init_enterprise_components()
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        self.init_ui()
        self._install_no_wheel_behavior()
        self.setup_shortcuts()
        self.apply_theme()
        self.restore_window_state()
    
    def _init_enterprise_components(self):
        """Initialize enterprise backend components if available"""
        if not HAS_ENTERPRISE_MODULES:
            return
            
        try:
            if self.enterprise_config.get("use_database", True):
                self.db_manager = DatabaseManager(self.enterprise_config.get("db_path", "duplicates.db"))
                logger.info("✅ Database manager initialized")
                
            if self.enterprise_config.get("use_plugins", True):
                self.plugin_manager = PluginManager(self.enterprise_config.get("plugin_dir", "plugins"))
                # PluginManager loads plugins in __init__, no separate call needed
                logger.info("✅ Plugin manager initialized")
                
            if self.enterprise_config.get("optimize_memory", True):
                self.memory_manager = MemoryManager()
                logger.info("✅ Memory manager initialized")
                
            if self.enterprise_config.get("enable_merging", True):
                self.file_merger = FileMerger()
                logger.info("✅ File merger initialized")
                
        except Exception as e:
            logger.warning(f"Advanced component initialization error: {e}")
        
    def init_ui(self):
        self.setWindowTitle("Duplicate File Finder")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1000, 600)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.create_status_bar()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Scanner tab
        self.create_scanner_tab()
        
        # Results tab
        self.create_results_tab()
        
        # Settings tab
        self.create_settings_tab()
        
        # Apply saved smart preset after UI is ready
        self._apply_selected_preset(initial=True)
        
        self.directory = None
    
    def create_toolbar(self):
        """Create main toolbar with quick actions"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolBar")  # Required for saveState/restoreState
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setContentsMargins(6, 6, 6, 6)
        self.addToolBar(toolbar)
        
        # Open folder action
        open_action = QAction("Open Folder", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.setToolTip("Select folder to scan (Ctrl+O)")
        open_action.triggered.connect(self.select_directory)
        toolbar.addAction(open_action)
        open_btn = toolbar.widgetForAction(open_action)
        if open_btn:
            open_btn.setMinimumWidth(118)
            open_btn.setMinimumHeight(38)
        
        # Start scan action
        self.scan_action = QAction("Scan", self)
        self.scan_action.setShortcut(QKeySequence("Ctrl+S"))
        self.scan_action.setToolTip("Start scanning for duplicates (Ctrl+S)")
        self.scan_action.triggered.connect(self.scan_duplicates)
        self.scan_action.setEnabled(False)
        toolbar.addAction(self.scan_action)
        scan_btn = toolbar.widgetForAction(self.scan_action)
        if scan_btn:
            scan_btn.setMinimumWidth(82)
            scan_btn.setMinimumHeight(38)
        
        toolbar.addSeparator()
        
        # Pause action
        self.pause_action = QAction("Pause", self)
        self.pause_action.setShortcut(QKeySequence("Ctrl+P"))
        self.pause_action.setToolTip("Pause/Resume scan (Ctrl+P)")
        self.pause_action.triggered.connect(self.toggle_pause)
        self.pause_action.setEnabled(False)
        toolbar.addAction(self.pause_action)
        pause_btn = toolbar.widgetForAction(self.pause_action)
        if pause_btn:
            pause_btn.setMinimumWidth(84)
            pause_btn.setMinimumHeight(38)
        
        # Cancel action
        self.cancel_action = QAction("⏹️ Stop", self)
        self.cancel_action.setShortcut(QKeySequence("Escape"))
        self.cancel_action.setToolTip("Cancel scan (Escape)")
        self.cancel_action.triggered.connect(self.cancel_scan)
        self.cancel_action.setEnabled(False)
        toolbar.addAction(self.cancel_action)
        cancel_btn = toolbar.widgetForAction(self.cancel_action)
        if cancel_btn:
            cancel_btn.setMinimumWidth(76)
            cancel_btn.setMinimumHeight(38)
        
        toolbar.addSeparator()
        
        # Theme toggle
        self.theme_action = QAction("Dark Mode", self)
        self.theme_action.setShortcut(QKeySequence("Ctrl+T"))
        self.theme_action.setToolTip("Toggle dark/light theme (Ctrl+T)")
        self.theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(self.theme_action)
        theme_btn = toolbar.widgetForAction(self.theme_action)
        if theme_btn:
            theme_btn.setMinimumWidth(124)
            theme_btn.setMinimumHeight(38)
        
        toolbar.addSeparator()
        
        # Export menu
        export_btn = QToolButton()
        export_btn.setText("Export")
        export_btn.setToolTip("Export results")
        export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        export_btn.setMinimumWidth(98)
        export_btn.setMinimumHeight(38)
        
        export_menu = QMenu(self)
        export_menu.addAction("Export to JSON", self.export_to_json)
        export_menu.addAction("Export to CSV", self.export_to_csv)
        export_menu.addAction("Export to HTML", self.export_to_html)
        export_btn.setMenu(export_menu)
        toolbar.addWidget(export_btn)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Search in results
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search in results...")
        self.search_edit.setMinimumWidth(240)
        self.search_edit.setMaximumWidth(320)
        self.search_edit.setMinimumHeight(38)
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self.filter_results)
        self.search_edit.setToolTip("Filter results by filename (Ctrl+F)")
        toolbar.addWidget(self.search_edit)
        
        # Help action
        help_action = QAction("Help", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.setToolTip("Show keyboard shortcuts (F1)")
        help_action.triggered.connect(self.show_shortcuts_help)
        toolbar.addAction(help_action)
        help_btn = toolbar.widgetForAction(help_action)
        if help_btn:
            help_btn.setMinimumWidth(78)
            help_btn.setMinimumHeight(38)
        
        # Create menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create menu bar with file, view, tools menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Folder...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.select_directory)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_json = QAction("Export to JSON...", self)
        export_json.triggered.connect(self.export_to_json)
        file_menu.addAction(export_json)
        
        export_csv = QAction("Export to CSV...", self)
        export_csv.triggered.connect(self.export_to_csv)
        file_menu.addAction(export_csv)
        
        export_html = QAction("Export to HTML...", self)
        export_html.triggered.connect(self.export_to_html)
        file_menu.addAction(export_html)
        
        file_menu.addSeparator()
        
        import_session = QAction("Import Session...", self)
        import_session.triggered.connect(self.import_session)
        file_menu.addAction(import_session)
        
        export_session = QAction("Export Session...", self)
        export_session.triggered.connect(self.export_session)
        file_menu.addAction(export_session)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        self.undo_action = QAction("&Undo Last Operation", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo_last_operation)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)
        
        edit_menu.addSeparator()
        
        select_all_action = QAction("Select &All in Group", self)
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self.select_all_in_group)
        edit_menu.addAction(select_all_action)
        
        deselect_all_action = QAction("&Deselect All", self)
        deselect_all_action.setShortcut("Ctrl+Shift+A")
        deselect_all_action.triggered.connect(self.deselect_all_files)
        edit_menu.addAction(deselect_all_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        theme_action = QAction("Toggle &Theme", self)
        theme_action.setShortcut("Ctrl+T")
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        
        view_menu.addSeparator()
        
        refresh_action = QAction("&Refresh Results", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_results)
        view_menu.addAction(refresh_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        scan_action = QAction("🔍 &Start Scan", self)
        scan_action.setShortcut("Ctrl+S")
        scan_action.triggered.connect(self.scan_duplicates)
        tools_menu.addAction(scan_action)
        
        tools_menu.addSeparator()
        
        enterprise_config_action = QAction("&Advanced Configuration...", self)
        enterprise_config_action.triggered.connect(self.show_enterprise_config)
        tools_menu.addAction(enterprise_config_action)
        
        if HAS_ENTERPRISE_MODULES:
            merge_analysis_action = QAction("🔀 &Merge Analysis...", self)
            merge_analysis_action.triggered.connect(self.show_merge_analysis)
            tools_menu.addAction(merge_analysis_action)
        
        tools_menu.addSeparator()
        
        clear_backups_action = QAction("Clear Undo &Backups", self)
        clear_backups_action.triggered.connect(self.clear_undo_backups)
        tools_menu.addAction(clear_backups_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self.show_shortcuts_help)
        help_menu.addAction(shortcuts_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """Create status bar with useful information"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # GPU status
        gpu_label = QLabel()
        if HAS_CUDA:
            gpu_label.setText(f"GPU: {CUDA_BACKEND.upper()}")
            gpu_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        else:
            gpu_label.setText("CPU Only")
            gpu_label.setStyleSheet("color: #ff9800;")
        self.status_bar.addPermanentWidget(gpu_label)
        
        # Worker threads with mode indicator
        mode_str = "GPU mode" if HAS_CUDA else "CPU mode"
        cpu_label = QLabel(f"{OPTIMAL_WORKERS} Workers ({mode_str})")
        self.status_bar.addPermanentWidget(cpu_label)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        self.status_bar.addPermanentWidget(sep)
        
        # Results count
        self.results_status = QLabel("No scan results")
        self.status_bar.addPermanentWidget(self.results_status)
        
        # Main status message
        self.status_bar.showMessage("Ready - Select a folder to begin scanning")
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Focus search
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        search_shortcut.activated.connect(lambda: self.search_edit.setFocus())
        
        # Clear search
        clear_search = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
        clear_search.activated.connect(lambda: self.search_edit.clear())
        
        # Switch tabs
        tab1 = QShortcut(QKeySequence("Ctrl+1"), self)
        tab1.activated.connect(lambda: self.tab_widget.setCurrentIndex(0))
        
        tab2 = QShortcut(QKeySequence("Ctrl+2"), self)
        tab2.activated.connect(lambda: self.tab_widget.setCurrentIndex(1))
        
        tab3 = QShortcut(QKeySequence("Ctrl+3"), self)
        tab3.activated.connect(lambda: self.tab_widget.setCurrentIndex(2))
        
        # Select all in current group
        select_all = QShortcut(QKeySequence("Ctrl+A"), self)
        select_all.activated.connect(self.select_all_in_group)
        
        # Delete selected
        delete_shortcut = QShortcut(QKeySequence("Delete"), self)
        delete_shortcut.activated.connect(self.delete_selected_files)
        
        # Recycle bin
        recycle_shortcut = QShortcut(QKeySequence("Shift+Delete"), self)
        recycle_shortcut.activated.connect(self.recycle_selected_files)
    
    def apply_theme(self):
        """Apply current theme"""
        if self.is_dark_theme:
            self.setStyleSheet(DARK_THEME)
            self.theme_action.setText("Light Mode")
        else:
            self.setStyleSheet(LIGHT_THEME)
            self.theme_action.setText("Dark Mode")
    
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        self.is_dark_theme = not self.is_dark_theme
        self.settings.setValue("dark_theme", self.is_dark_theme)
        self.apply_theme()
        self.status_bar.showMessage(f"Theme changed to {'Dark' if self.is_dark_theme else 'Light'} mode", 2000)
    
    def restore_window_state(self):
        """Restore window position and size"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore last directory
        last_dir = self.settings.value("last_directory")
        if last_dir and os.path.isdir(last_dir):
            self.directory = last_dir
            self.dir_label.setText(f"📁 {last_dir}")
            self.scan_btn.setEnabled(True)
            self.scan_action.setEnabled(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter for folder drop"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if os.path.isdir(path):
                        event.acceptProposedAction()
                        return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle folder drop"""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    self.directory = path
                    self.dir_label.setText(f"📁 {path}")
                    self.scan_btn.setEnabled(True)
                    self.scan_action.setEnabled(True)
                    self.status_bar.showMessage(f"Folder dropped: {path}", 3000)
                    return
    
    def filter_results(self, text):
        """Filter results based on search text"""
        search_text = text.lower()
        
        # Filter duplicate files tree
        for i in range(self.duplicate_files_tree.topLevelItemCount()):
            item = self.duplicate_files_tree.topLevelItem(i)
            if item:
                filename = item.text(1).lower()
                filepath = item.text(2).lower()
                visible = search_text in filename or search_text in filepath or not search_text
                item.setHidden(not visible)
        
        # Filter similar files tree
        for i in range(self.similar_files_tree.topLevelItemCount()):
            item = self.similar_files_tree.topLevelItem(i)
            if item:
                filename = item.text(1).lower()
                filepath = item.text(2).lower()
                visible = search_text in filename or search_text in filepath or not search_text
                item.setHidden(not visible)
    
    def show_shortcuts_help(self):
        """Show keyboard shortcuts help dialog"""
        shortcuts = """
<h2>Keyboard Shortcuts</h2>

<h3>Navigation</h3>
<table>
<tr><td><b>Ctrl+1</b></td><td>Switch to Scanner tab</td></tr>
<tr><td><b>Ctrl+2</b></td><td>Switch to Results tab</td></tr>
<tr><td><b>Ctrl+3</b></td><td>Switch to Settings tab</td></tr>
</table>

<h3>Scanning</h3>
<table>
<tr><td><b>Ctrl+O</b></td><td>Open folder</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Start scan</td></tr>
<tr><td><b>Ctrl+P</b></td><td>Pause/Resume scan</td></tr>
<tr><td><b>Escape</b></td><td>Cancel scan</td></tr>
</table>

<h3>Results</h3>
<table>
<tr><td><b>Ctrl+F</b></td><td>Focus search box</td></tr>
<tr><td><b>Ctrl+Shift+F</b></td><td>Clear search</td></tr>
<tr><td><b>Ctrl+A</b></td><td>Select all in group</td></tr>
<tr><td><b>Delete</b></td><td>Delete selected files</td></tr>
<tr><td><b>Shift+Delete</b></td><td>Move to Recycle Bin</td></tr>
</table>

<h3>Other</h3>
<table>
<tr><td><b>Ctrl+T</b></td><td>Toggle dark/light theme</td></tr>
<tr><td><b>F1</b></td><td>Show this help</td></tr>
</table>

<h3>Tips</h3>
<ul>
<li>Drag and drop a folder to scan it</li>
<li>Use the search box to filter results</li>
<li>GPU acceleration is automatic when available</li>
</ul>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(shortcuts)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
        
    def create_scanner_tab(self):
        """Create the scanner configuration tab"""
        scanner_widget = QWidget()
        outer_layout = QVBoxLayout(scanner_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        label_width = 160

        def make_row_label(text):
            label = QLabel(text)
            label.setFixedWidth(label_width)
            label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            return label

        def make_form_row():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(14)
            row_widget.setMinimumHeight(34)
            return row_widget, row_layout
        
        # Directory selection
        dir_group = QGroupBox("Directory Selection")
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setContentsMargins(12, 14, 12, 12)
        
        dir_select_layout = QHBoxLayout()
        dir_select_layout.setSpacing(12)
        self.select_btn = QPushButton("Select Directory to Scan")
        self.select_btn.clicked.connect(self.select_directory)
        self.select_btn.setMinimumWidth(260)
        self.select_btn.setMinimumHeight(40)
        dir_select_layout.addWidget(self.select_btn)

        self.dir_label = QLabel("No directory selected")
        self.dir_label.setWordWrap(True)
        self.dir_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.dir_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        dir_select_layout.addWidget(self.dir_label)
        dir_layout.addLayout(dir_select_layout)
        
        layout.addWidget(dir_group)
        
        # Move destination selection
        move_group = QGroupBox("Move Destination (Optional)")
        move_layout = QVBoxLayout(move_group)
        move_layout.setContentsMargins(12, 14, 12, 12)
        
        move_select_layout = QHBoxLayout()
        move_select_layout.setSpacing(12)
        self.move_dest_btn = QPushButton("Select Move Destination")
        self.move_dest_btn.clicked.connect(self.select_move_destination)
        self.move_dest_btn.setMinimumWidth(260)
        self.move_dest_btn.setMinimumHeight(40)
        move_select_layout.addWidget(self.move_dest_btn)
        
        self.move_dest_label = QLabel("No destination selected (move actions are disabled)")
        self.move_dest_label.setWordWrap(True)
        self.move_dest_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.move_dest_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        move_select_layout.addWidget(self.move_dest_label)
        move_layout.addLayout(move_select_layout)
        
        layout.addWidget(move_group)
        
        # Smart presets
        preset_group = QGroupBox("⚡ Smart Presets (Category + Accuracy/Performance)")
        preset_layout = QVBoxLayout(preset_group)
        preset_layout.setContentsMargins(14, 10, 14, 14)
        preset_layout.setSpacing(10)

        self.preset_enable_cb = QCheckBox("Enable Smart Presets")
        self.preset_enable_cb.setChecked(self.preset_enabled)
        self.preset_enable_cb.setMinimumHeight(30)
        preset_layout.addWidget(self.preset_enable_cb)

        category_row_widget, category_row = make_form_row()
        category_row.addWidget(make_row_label("Category:"))
        self.category_combo = QComboBox()
        for key, data in CATEGORY_PRESETS.items():
            self.category_combo.addItem(data.get("label", key.capitalize()), key)
        self.category_combo.setMinimumWidth(260)
        category_row.addWidget(self.category_combo, 1)
        preset_layout.addWidget(category_row_widget)

        mode_row_widget, mode_row = make_form_row()
        mode_row.addWidget(make_row_label("Mode:"))
        self.preset_mode_combo = QComboBox()
        self.preset_mode_combo.addItem("Accuracy", "accuracy")
        self.preset_mode_combo.addItem("Performance", "performance")
        self.preset_mode_combo.setMinimumWidth(220)
        mode_row.addWidget(self.preset_mode_combo, 1)
        preset_layout.addWidget(mode_row_widget)

        # Restore last selections
        self._set_combo_by_data(self.category_combo, self.current_category_key)
        self._set_combo_by_data(self.preset_mode_combo, self.current_preset_mode)

        self.preset_desc_label = QLabel("Auto-sets comparison and GPU settings for this category.")
        self.preset_desc_label.setStyleSheet("color: #999; font-style: italic;")
        self.preset_desc_label.setWordWrap(True)
        preset_layout.addWidget(self.preset_desc_label)

        self.preset_enable_cb.stateChanged.connect(self._on_preset_changed)
        self.category_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.preset_mode_combo.currentIndexChanged.connect(self._on_preset_changed)

        layout.addWidget(preset_group)
        
        # Scan filters
        filter_group = QGroupBox("Scan Filters")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setContentsMargins(14, 10, 14, 14)
        filter_layout.setSpacing(10)
        
        # File type filters
        self.type_checkboxes = {}
        file_types_row, file_types_layout = make_form_row()
        file_types_layout.setSpacing(14)
        file_types_layout.addWidget(make_row_label("File Types:"))
        file_types_content = QWidget()
        file_types_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        file_types_content_layout = QHBoxLayout(file_types_content)
        file_types_content_layout.setContentsMargins(0, 0, 0, 0)
        file_types_content_layout.setSpacing(14)
        for type_name in ['images', 'documents', 'text', 'code', 'audio', 'video', 'archives']:
            cb = QCheckBox(type_name.capitalize())
            cb.setChecked(False)  # Unchecked = scan all types
            self.type_checkboxes[type_name] = cb
            file_types_content_layout.addWidget(cb)
        file_types_content_layout.addStretch()
        file_types_layout.addWidget(file_types_content, 1)
        filter_layout.addWidget(file_types_row)
        
        # Custom extensions
        custom_ext_row, custom_ext_layout = make_form_row()
        custom_ext_layout.addWidget(make_row_label("Custom extensions:"))
        self.custom_ext_edit = QLineEdit()
        self.custom_ext_edit.setPlaceholderText(".ext1, .ext2 (leave empty for all)")
        custom_ext_layout.addWidget(self.custom_ext_edit, 1)
        filter_layout.addWidget(custom_ext_row)
        
        # Size filters
        min_size_row, min_size_layout = make_form_row()
        min_size_layout.addWidget(make_row_label("Min size (KB):"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 1000000)
        self.min_size_spin.setValue(0)
        self.min_size_spin.setMinimumWidth(140)
        min_size_layout.addWidget(self.min_size_spin)
        min_size_layout.addStretch()
        filter_layout.addWidget(min_size_row)
        
        max_size_row, max_size_layout = make_form_row()
        max_size_layout.addWidget(make_row_label("Max size (MB):"))
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(0, 100000)
        self.max_size_spin.setValue(0)  # 0 = no limit
        self.max_size_spin.setSpecialValueText("No limit")
        self.max_size_spin.setMinimumWidth(180)
        max_size_layout.addWidget(self.max_size_spin)
        max_size_layout.addStretch()
        filter_layout.addWidget(max_size_row)
        
        # Excluded folders
        exclude_row, exclude_layout = make_form_row()
        exclude_layout.addWidget(make_row_label("Exclude folders:"))
        self.exclude_folders_edit = QLineEdit()
        self.exclude_folders_edit.setText(", ".join(ScanFilters.EXCLUDE_FOLDERS[:5]))
        self.exclude_folders_edit.setPlaceholderText("node_modules, .git, __pycache__")
        exclude_layout.addWidget(self.exclude_folders_edit, 1)
        filter_layout.addWidget(exclude_row)
        
        layout.addWidget(filter_group)
        
        # Scan controls
        scan_group = QGroupBox("Scan Configuration")
        scan_layout = QVBoxLayout(scan_group)
        scan_layout.setContentsMargins(14, 10, 14, 14)
        scan_layout.setSpacing(10)
        
        hash_row, hash_layout = make_form_row()
        hash_layout.addWidget(make_row_label("Hash Type:"))
        self.hash_combo = QComboBox()
        self.hash_combo.addItems(["MD5", "SHA256", "Perceptual", "Audio", "Semantic"])
        self.hash_combo.setMinimumWidth(170)
        self.hash_combo.setToolTip(
            "MD5/SHA256: Exact byte-for-byte matches\n"
            "Perceptual: Similar images/videos (GPU accelerated)\n"
            "Audio: Similar audio files (Chromaprint + GPU MFCC)\n"
            "Semantic: Similar text content (GPU embeddings)"
        )
        self.hash_combo.currentTextChanged.connect(self._on_hash_type_changed)
        hash_layout.addWidget(self.hash_combo, 1)
        scan_layout.addWidget(hash_row)
        
        self.find_similar_cb = QCheckBox("Find similar files (not just exact duplicates)")
        self.find_similar_cb.setChecked(True)
        self.find_similar_cb.setMinimumHeight(30)
        scan_layout.addWidget(self.find_similar_cb)
        
        # Scan buttons
        scan_btn_layout = QHBoxLayout()
        scan_btn_layout.setSpacing(12)
        
        self.scan_btn = QPushButton("Start Scan")
        self.scan_btn.clicked.connect(self.scan_duplicates)
        self.scan_btn.setEnabled(False)
        self.scan_btn.setToolTip("Start scanning for duplicates (Ctrl+S)")
        self.scan_btn.setMinimumWidth(160)
        self.scan_btn.setMinimumHeight(38)
        scan_btn_layout.addWidget(self.scan_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip("Pause or resume the scan (Ctrl+P)")
        self.pause_btn.setMinimumWidth(120)
        self.pause_btn.setMinimumHeight(38)
        scan_btn_layout.addWidget(self.pause_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_scan)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Cancel the current scan (Escape)")
        self.cancel_btn.setMinimumWidth(120)
        self.cancel_btn.setMinimumHeight(38)
        scan_btn_layout.addWidget(self.cancel_btn)
        
        scan_layout.addLayout(scan_btn_layout)
        
        layout.addWidget(scan_group)
        
        # Progress and status
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(12, 10, 12, 12)
        progress_layout.setSpacing(6)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Detailed progress info
        progress_details = QHBoxLayout()
        
        self.status_label = QLabel("Select a directory to start")
        progress_details.addWidget(self.status_label)
        
        self.current_file_label = QLabel("")
        self.current_file_label.setStyleSheet("color: #999;")
        progress_details.addWidget(self.current_file_label)
        
        progress_details.addStretch()
        
        self.files_count_label = QLabel("")
        progress_details.addWidget(self.files_count_label)
        
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        progress_details.addWidget(self.eta_label)
        
        progress_layout.addLayout(progress_details)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        self.tab_widget.addTab(scanner_widget, "Scanner")
    
    def _set_combo_by_data(self, combo, data_value):
        """Set combo box to item matching data value."""
        if not combo:
            return
        for i in range(combo.count()):
            if combo.itemData(i) == data_value:
                combo.setCurrentIndex(i)
                return

    def _on_preset_changed(self):
        """Apply preset settings when user changes category or mode."""
        self._apply_selected_preset()

    def _get_active_preset_config(self):
        preset_bucket = CATEGORY_PRESETS.get(self.current_category_key, CATEGORY_PRESETS["general"])
        return preset_bucket, preset_bucket.get(self.current_preset_mode, preset_bucket["accuracy"])

    def _set_hash_focus_value(self, hash_type):
        if not hasattr(self, "hash_combo"):
            return
        self._suppress_hash_focus_feedback = True
        try:
            self.hash_combo.setCurrentText(hash_type)
        finally:
            self._suppress_hash_focus_feedback = False

    def _sync_primary_focus_from_preset(self):
        if not hasattr(self, "hash_combo"):
            return
        _, preset_config = self._get_active_preset_config()
        self._set_hash_focus_value(preset_config.get("hash_type", "MD5"))
        self.hash_combo.setEnabled(not self.preset_enabled)

    def _install_no_wheel_behavior(self):
        for widget_type in (QComboBox, QSpinBox, QSlider):
            for widget in self.findChildren(widget_type):
                widget.installEventFilter(self._wheel_guard)

    def _apply_selected_preset(self, initial=False):
        """Apply the selected smart preset to UI controls."""
        if hasattr(self, "preset_enable_cb"):
            self.preset_enabled = self.preset_enable_cb.isChecked()
        if hasattr(self, "category_combo"):
            self.current_category_key = self.category_combo.currentData() or self.current_category_key
        if hasattr(self, "preset_mode_combo"):
            self.current_preset_mode = self.preset_mode_combo.currentData() or self.current_preset_mode

        self.settings.setValue("preset_enabled", self.preset_enabled)
        self.settings.setValue("preset_category", self.current_category_key)
        self.settings.setValue("preset_mode", self.current_preset_mode)

        preset_bucket, preset_config = self._get_active_preset_config()

        if self.preset_enabled:
            self._apply_preset_to_controls(preset_config, preset_bucket.get("types", []))
        self._sync_primary_focus_from_preset()

        if hasattr(self, "preset_desc_label"):
            mode_label = "Accuracy" if self.current_preset_mode == "accuracy" else "Performance"
            desc = f"{preset_bucket.get('label', 'Category')} · {mode_label}"
            self.preset_desc_label.setText(f"{desc} — auto-sets filters, similarity, and GPU options.")

    def _apply_preset_to_controls(self, preset_config, category_types):
        """Apply preset config to scan and similarity UI controls."""
        # File type checkboxes
        if hasattr(self, "type_checkboxes"):
            for cb in self.type_checkboxes.values():
                cb.setChecked(False)
            for t in category_types:
                if t in self.type_checkboxes:
                    self.type_checkboxes[t].setChecked(True)

        # Scan settings
        if hasattr(self, "hash_combo"):
            self._set_hash_focus_value(preset_config.get("hash_type", "MD5"))
        if hasattr(self, "find_similar_cb"):
            self.find_similar_cb.setChecked(preset_config.get("find_similar", True))

        # Similarity settings
        if hasattr(self, "compare_by_name_cb"):
            self.compare_by_name_cb.setChecked(preset_config.get("compare_by_name", True))
        if hasattr(self, "name_similarity_slider"):
            self.name_similarity_slider.setValue(int(preset_config.get("name_similarity", 0.8) * 100))
        if hasattr(self, "compare_by_size_cb"):
            self.compare_by_size_cb.setChecked(preset_config.get("compare_by_size", True))
        if hasattr(self, "size_tolerance_spin"):
            self.size_tolerance_spin.setValue(int(preset_config.get("size_tolerance", 1)))
        if hasattr(self, "compare_text_content_cb"):
            self.compare_text_content_cb.setChecked(preset_config.get("compare_text_content", False))
        if hasattr(self, "content_similarity_slider"):
            self.content_similarity_slider.setValue(int(preset_config.get("content_similarity", 0.9) * 100))

        # Perceptual hash preset
        if hasattr(self, "phash_preset_combo"):
            desired = preset_config.get("phash_preset", "balanced")
            for i in range(self.phash_preset_combo.count()):
                if self.phash_preset_combo.itemData(i) == desired:
                    self.phash_preset_combo.setCurrentIndex(i)
                    break

        # GPU toggles (auto-enable when available)
        if HAS_CUDA and hasattr(self, "use_gpu_cb"):
            self.use_gpu_cb.setChecked(bool(preset_config.get("use_gpu", True)))
        if HAS_CUDA and hasattr(self, "use_gpu_hashing_cb"):
            self.use_gpu_hashing_cb.setChecked(bool(preset_config.get("use_gpu", True)))

    def create_results_tab(self):
        """Create the results display tab"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Stepwise workflow controls
        step_group = QGroupBox("Workflow")
        step_layout = QHBoxLayout(step_group)
        step_layout.setSpacing(8)

        self.step_buttons = {}
        self.step_button_group = QButtonGroup()
        self.step_button_group.setExclusive(True)

        step_definitions = [
            ("All", "All Results"),
            ("Exact", "Step 1: Exact Duplicates"),
            ("Content", "Step 2: Same Content"),
            ("Name/Size", "Step 3: Name/Size Similarity"),
            ("SameNameDiffSize", "Step 3b: Same Name ≠ Size")
        ]

        for step_key, label in step_definitions:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda _, s=step_key: self._set_step_filter(s))
            self.step_buttons[step_key] = btn
            self.step_button_group.addButton(btn)
            step_layout.addWidget(btn)

        self.prev_step_btn = QPushButton("Previous")
        self.prev_step_btn.clicked.connect(lambda: self._cycle_step(-1))
        self.prev_step_btn.setMinimumHeight(36)
        step_layout.addWidget(self.prev_step_btn)

        self.next_step_btn = QPushButton("Next")
        self.next_step_btn.clicked.connect(lambda: self._cycle_step(1))
        self.next_step_btn.setMinimumHeight(36)
        step_layout.addWidget(self.next_step_btn)

        step_layout.addStretch()
        self.step_status_label = QLabel("All results")
        self.step_status_label.setStyleSheet("color: #999;")
        step_layout.addWidget(self.step_status_label)

        layout.addWidget(step_group)

        # Results tabs
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)
        
        # Exact duplicates tab
        self.create_duplicates_results()
        
        # Similar files tab
        self.create_similar_results()
        
        # Action buttons
        actions_container = QVBoxLayout()
        actions_container.setSpacing(10)
        actions_row_top = QHBoxLayout()
        actions_row_top.setSpacing(10)
        actions_row_bottom = QHBoxLayout()
        actions_row_bottom.setSpacing(10)
        
        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_files)
        self.delete_selected_btn.setEnabled(False)
        self.delete_selected_btn.setMinimumHeight(38)
        actions_row_top.addWidget(self.delete_selected_btn)
        
        self.recycle_selected_btn = QPushButton("♻️ Send to Recycle Bin")
        self.recycle_selected_btn.clicked.connect(self.recycle_selected_files)
        self.recycle_selected_btn.setEnabled(False)
        self.recycle_selected_btn.setToolTip("Safely move files to Recycle Bin (recoverable)")
        self.recycle_selected_btn.setMinimumHeight(38)
        if not HAS_SEND2TRASH:
            self.recycle_selected_btn.setEnabled(False)
            self.recycle_selected_btn.setToolTip("Install send2trash package: pip install send2trash")
        actions_row_top.addWidget(self.recycle_selected_btn)
        
        self.move_selected_btn = QPushButton("Move Selected")
        self.move_selected_btn.clicked.connect(self.move_selected_files)
        self.move_selected_btn.setEnabled(False)
        self.move_selected_btn.setMinimumHeight(38)
        actions_row_top.addWidget(self.move_selected_btn)

        self.bulk_move_btn = QPushButton("Bulk Move Duplicates")
        self.bulk_move_btn.clicked.connect(self.show_bulk_move_preview)
        self.bulk_move_btn.setEnabled(False)
        self.bulk_move_btn.setToolTip("Move all duplicates while keeping one original per group")
        self.bulk_move_btn.setStyleSheet("background-color: #2e7d32; color: white; padding: 8px 16px;")
        self.bulk_move_btn.setMinimumHeight(38)
        actions_row_top.addWidget(self.bulk_move_btn)

        self.audio_smart_move_btn = QPushButton("Audio Smart Move")
        self.audio_smart_move_btn.clicked.connect(self.show_audio_move_preview)
        self.audio_smart_move_btn.setEnabled(False)
        self.audio_smart_move_btn.setToolTip("Keep the best quality audio in each duplicate group")
        self.audio_smart_move_btn.setStyleSheet("background-color: #6a1b9a; color: white; padding: 8px 16px;")
        self.audio_smart_move_btn.setMinimumHeight(38)
        actions_row_top.addWidget(self.audio_smart_move_btn)
        actions_row_top.addStretch()
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_in_group)
        self.select_all_btn.setMinimumHeight(36)
        actions_row_bottom.addWidget(self.select_all_btn)
        
        self.clear_selection_btn = QPushButton("Clear")
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        self.clear_selection_btn.setMinimumHeight(36)
        actions_row_bottom.addWidget(self.clear_selection_btn)
        
        # Visual separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        actions_row_bottom.addWidget(separator)
        
        # Bulk delete all duplicates button
        self.bulk_delete_btn = QPushButton("Delete All Duplicates...")
        self.bulk_delete_btn.clicked.connect(self.show_bulk_delete_preview)
        self.bulk_delete_btn.setEnabled(False)
        self.bulk_delete_btn.setToolTip("Preview and delete all duplicates at once (keeps one per group)")
        self.bulk_delete_btn.setMinimumHeight(36)
        actions_row_bottom.addWidget(self.bulk_delete_btn)
        
        actions_row_bottom.addStretch()
        
        # Export button
        self.export_btn = QPushButton("📥 Export Results")
        self.export_btn.clicked.connect(self.show_export_menu)
        self.export_btn.setMinimumHeight(36)
        self.export_btn.setMinimumWidth(140)
        actions_row_bottom.addWidget(self.export_btn)
        
        actions_container.addLayout(actions_row_top)
        actions_container.addLayout(actions_row_bottom)
        layout.addLayout(actions_container)
        
        self.tab_widget.addTab(results_widget, "Results")
    
    def _set_step_filter(self, step_name):
        """Set current workflow step and refresh results."""
        self.current_step_filter = step_name

        if step_name == "Exact":
            self.similar_step_filter = None
            self.results_tabs.setCurrentIndex(0)
            self.step_status_label.setText("Showing exact duplicates only")
        elif step_name == "Content":
            self.similar_step_filter = {"content"}
            self.results_tabs.setCurrentIndex(1)
            self.step_status_label.setText("Showing same-content matches")
        elif step_name == "Name/Size":
            self.similar_step_filter = {"name_size", "same_name_diff_size"}
            self.results_tabs.setCurrentIndex(1)
            self.step_status_label.setText("Showing name/size similarity matches")
        elif step_name == "SameNameDiffSize":
            self.similar_step_filter = {"same_name_diff_size"}
            self.results_tabs.setCurrentIndex(1)
            self.step_status_label.setText("Showing same name, different size matches")
        else:
            self.similar_step_filter = None
            self.step_status_label.setText("Showing all results")

        self.populate_duplicate_groups()
        self.populate_similar_groups()

    def _cycle_step(self, direction):
        """Move to previous or next workflow step."""
        steps = ["Exact", "Content", "Name/Size", "SameNameDiffSize"]
        if self.current_step_filter in steps:
            idx = steps.index(self.current_step_filter)
        else:
            idx = 0 if direction > 0 else len(steps) - 1
        idx = max(0, min(len(steps) - 1, idx + direction))
        next_step = steps[idx]
        if next_step in self.step_buttons:
            self.step_buttons[next_step].setChecked(True)
        self._set_step_filter(next_step)

    def _update_step_counts(self):
        """Update step buttons with counts."""
        exact_count = len(self.duplicates) if hasattr(self, "duplicates") else 0
        content_count = 0
        name_size_count = 0
        same_name_diff_size_count = 0
        for group in self.similar_files.values() if hasattr(self, "similar_files") else []:
            match_type = group.get("match_type", "")
            if match_type == "content":
                content_count += 1
            elif match_type == "name_size":
                name_size_count += 1
            elif match_type == "same_name_diff_size":
                same_name_diff_size_count += 1

        if "Exact" in self.step_buttons:
            self.step_buttons["Exact"].setText(f"Step 1: Exact ({exact_count})")
        if "Content" in self.step_buttons:
            self.step_buttons["Content"].setText(f"Step 2: Content ({content_count})")
        if "Name/Size" in self.step_buttons:
            self.step_buttons["Name/Size"].setText(f"Step 3: Name/Size ({name_size_count + same_name_diff_size_count})")
        if "SameNameDiffSize" in self.step_buttons:
            self.step_buttons["SameNameDiffSize"].setText(f"Step 3b: Same Name ≠ Size ({same_name_diff_size_count})")

    def create_duplicates_results(self):
        """Create exact duplicates results view"""
        duplicates_widget = QWidget()
        layout = QHBoxLayout(duplicates_widget)
        
        # Splitter for layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Groups list
        groups_widget = QWidget()
        groups_layout = QVBoxLayout(groups_widget)
        
        # Groups header with sorting
        groups_header = QHBoxLayout()
        groups_header.addWidget(QLabel("🔗 Duplicate Groups:"))
        groups_header.addStretch()
        groups_header.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Size (Largest First)",
            "File Count",
            "Newest First", 
            "Oldest First",
            "Name (A-Z)"
        ])
        self.sort_combo.setMinimumWidth(180)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        self.sort_combo.setToolTip("Sort duplicate groups by different criteria")
        groups_header.addWidget(self.sort_combo)
        
        # File type filter
        groups_header.addWidget(QLabel("Filter:"))
        self.file_type_filter = QComboBox()
        self.file_type_filter.addItems([
            "All Types",
            "Images",
            "Documents", 
            "Videos",
            "Audio",
            "Archives",
            "Code",
            "Text",
            "Other"
        ])
        self.file_type_filter.setMinimumWidth(150)
        self.file_type_filter.currentTextChanged.connect(self._on_file_type_filter_changed)
        self.file_type_filter.setToolTip("Filter results by file type")
        groups_header.addWidget(self.file_type_filter)
        
        groups_layout.addLayout(groups_header)
        
        self.groups_tree = QTreeWidget()
        self.groups_tree.setHeaderLabels(["Group", "Files", "Total Size"])
        self.groups_tree.currentItemChanged.connect(self.show_duplicate_group_files)
        groups_layout.addWidget(self.groups_tree)
        
        splitter.addWidget(groups_widget)
        
        # Files in selected group
        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.addWidget(QLabel("Files in Group (star = suggested keep)"))
        
        self.duplicate_files_tree = QTreeWidget()
        self.duplicate_files_tree.setHeaderLabels(["Select", "File Name", "Path", "Type", "Quality", "Details", "Size", "Modified"])
        self.duplicate_files_tree.itemChanged.connect(self.update_selection_buttons)
        self.duplicate_files_tree.currentItemChanged.connect(self.show_file_preview)
        files_layout.addWidget(self.duplicate_files_tree)
        
        splitter.addWidget(files_widget)
        
        # Preview area
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.addWidget(QLabel("Preview"))
        
        self.duplicate_preview_area = QTextEdit()
        self.duplicate_preview_area.setMaximumWidth(350)
        preview_layout.addWidget(self.duplicate_preview_area)
        
        splitter.addWidget(preview_widget)
        
        splitter.setSizes([300, 500, 350])
        layout.addWidget(splitter)
        
        self.results_tabs.addTab(duplicates_widget, "Exact Duplicates")
    
    def create_similar_results(self):
        """Create similar files results view"""
        similar_widget = QWidget()
        layout = QHBoxLayout(similar_widget)
        
        # Splitter for layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Similar groups list
        groups_widget = QWidget()
        groups_layout = QVBoxLayout(groups_widget)
        groups_layout.addWidget(QLabel("🔗 Similar File Groups:"))
        
        self.similar_groups_tree = QTreeWidget()
        self.similar_groups_tree.setHeaderLabels(["Group", "Similarity", "Files"])
        self.similar_groups_tree.currentItemChanged.connect(self.show_similar_group_files)
        groups_layout.addWidget(self.similar_groups_tree)
        
        splitter.addWidget(groups_widget)
        
        # Files in selected group
        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.addWidget(QLabel("Similar Files (star = suggested keep)"))
        
        self.similar_files_tree = QTreeWidget()
        self.similar_files_tree.setHeaderLabels(["Select", "File Name", "Path", "Type", "Quality", "Details", "Size", "Modified"])
        self.similar_files_tree.itemChanged.connect(self.update_selection_buttons)
        self.similar_files_tree.currentItemChanged.connect(self.show_file_preview)
        files_layout.addWidget(self.similar_files_tree)
        
        splitter.addWidget(files_widget)
        
        # Preview area
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.addWidget(QLabel("Preview"))
        
        self.similar_preview_area = QTextEdit()
        self.similar_preview_area.setMaximumWidth(350)
        preview_layout.addWidget(self.similar_preview_area)
        
        splitter.addWidget(preview_widget)
        
        splitter.setSizes([300, 500, 350])
        layout.addWidget(splitter)
        
        self.results_tabs.addTab(similar_widget, "Similar Files")

        # Default step selection
        if "All" in self.step_buttons:
            self.step_buttons["All"].setChecked(True)
            self._set_step_filter("All")
    
    def create_settings_tab(self):
        """Create settings configuration tab"""
        settings_widget = QWidget()
        outer_layout = QVBoxLayout(settings_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll_area)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)
        
        # GPU & Performance Settings
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QGridLayout(perf_group)
        perf_layout.setContentsMargins(14, 12, 14, 14)
        perf_layout.setHorizontalSpacing(14)
        perf_layout.setVerticalSpacing(10)
        perf_layout.setColumnStretch(1, 1)
        perf_layout.setColumnStretch(3, 1)
        
        # GPU acceleration
        self.use_gpu_cb = QCheckBox("Enable CUDA GPU Acceleration")
        self.use_gpu_cb.setChecked(HAS_CUDA)
        self.use_gpu_cb.setEnabled(HAS_CUDA)
        if not HAS_CUDA:
            self.use_gpu_cb.setToolTip("Install CuPy or PyTorch with CUDA for GPU acceleration")
        else:
            self.use_gpu_cb.setToolTip(f"Using {CUDA_BACKEND.upper()} backend")
        perf_layout.addWidget(self.use_gpu_cb, 0, 0, 1, 2)
        
        # GPU status label
        gpu_status = "✅ Available" if HAS_CUDA else "❌ Not Available"
        gpu_backend = f" ({CUDA_BACKEND.upper()})" if HAS_CUDA else ""
        self.gpu_status_label = QLabel(f"GPU Status: {gpu_status}{gpu_backend}")
        self.gpu_status_label.setStyleSheet(f"color: {'green' if HAS_CUDA else 'red'}; font-weight: bold;")
        perf_layout.addWidget(self.gpu_status_label, 0, 2, 1, 2)
        
        # Parallel processing
        self.use_parallel_cb = QCheckBox("Enable Parallel Processing (Multi-threading)")
        self.use_parallel_cb.setChecked(True)
        mode_str = "GPU mode" if HAS_CUDA else "CPU mode"
        self.use_parallel_cb.setToolTip(f"Uses {OPTIMAL_WORKERS} worker threads ({mode_str})")
        perf_layout.addWidget(self.use_parallel_cb, 1, 0, 1, 2)
        
        # Worker threads with adaptive explanation
        worker_info = f"Worker Threads: {OPTIMAL_WORKERS} ({mode_str})"
        if HAS_CUDA:
            worker_info += f" - reduced from {OPTIMAL_WORKERS_CPU_ONLY} to avoid GPU oversubscription"
        perf_layout.addWidget(QLabel(worker_info), 1, 2, 1, 2)
        
        # Batch size for GPU
        perf_layout.addWidget(QLabel("GPU Batch Size:"), 2, 0)
        self.gpu_batch_spin = QSpinBox()
        self.gpu_batch_spin.setRange(8, 256)
        self.gpu_batch_spin.setValue(64)
        self.gpu_batch_spin.setToolTip("Number of images to process per GPU batch")
        perf_layout.addWidget(self.gpu_batch_spin, 2, 1)
        
        # Memory optimization
        self.memory_opt_cb = QCheckBox("Memory Optimization (for large scans)")
        self.memory_opt_cb.setChecked(True)
        self.memory_opt_cb.setToolTip("Clears GPU memory periodically to prevent OOM errors")
        perf_layout.addWidget(self.memory_opt_cb, 2, 2, 1, 2)
        
        layout.addWidget(perf_group)
        
        # Perceptual Hash Settings (for image/video duplicates)
        phash_group = QGroupBox("🖼️ Perceptual Hash Settings (Image/Video)")
        phash_layout = QGridLayout(phash_group)
        phash_layout.setContentsMargins(14, 12, 14, 14)
        phash_layout.setHorizontalSpacing(14)
        phash_layout.setVerticalSpacing(10)
        phash_layout.setColumnStretch(1, 1)
        phash_layout.setColumnStretch(3, 1)
        
        # Import threshold system
        try:
            from gpu_accelerator import PHashThresholds
            self.phash_thresholds = PHashThresholds
        except ImportError:
            self.phash_thresholds = None
        
        # Preset selector
        phash_layout.addWidget(QLabel("Matching Strictness:"), 0, 0)
        self.phash_preset_combo = QComboBox()
        self.phash_preset_combo.addItem("🔒 Strict", "strict")
        self.phash_preset_combo.addItem("⚖️ Balanced", "balanced")
        self.phash_preset_combo.addItem("🔍 Aggressive", "aggressive")
        self.phash_preset_combo.setCurrentIndex(1)  # Default to balanced
        self.phash_preset_combo.setMinimumWidth(220)
        self.phash_preset_combo.currentIndexChanged.connect(self.on_phash_preset_changed)
        self.phash_preset_combo.setToolTip(
            "Strict: Only very similar files (minimal differences)\n"
            "Balanced: Good balance between precision and recall\n"
            "Aggressive: Catch more potential duplicates (may include false positives)"
        )
        phash_layout.addWidget(self.phash_preset_combo, 0, 1)
        
        # Preset description
        self.phash_preset_desc = QLabel("Good balance between precision and recall")
        self.phash_preset_desc.setStyleSheet("color: #666; font-style: italic;")
        self.phash_preset_desc.setWordWrap(True)
        phash_layout.addWidget(self.phash_preset_desc, 0, 2, 1, 2)
        
        # Current thresholds display
        phash_layout.addWidget(QLabel("Current thresholds (Hamming distance):"), 1, 0, 1, 2)
        self.phash_thresholds_label = QLabel()
        self.update_phash_thresholds_display()
        self.phash_thresholds_label.setStyleSheet("font-family: monospace; color: #4fc3f7;")
        phash_layout.addWidget(self.phash_thresholds_label, 1, 2, 1, 2)
        
        # Hamming distance explanation
        hamming_help = QLabel(
            "ℹ️ Hamming distance = number of differing bits (0=identical, 64=completely different)\n"
            "    0: Identical | 1-5: Very similar | 6-10: Similar | 11-15: Somewhat similar | 16+: Different"
        )
        hamming_help.setStyleSheet("color: #888; font-size: 10px;")
        hamming_help.setWordWrap(True)
        phash_layout.addWidget(hamming_help, 2, 0, 1, 4)
        
        layout.addWidget(phash_group)
        
        # Similarity detection settings
        similarity_group = QGroupBox("🎯 Similarity Detection Settings")
        similarity_layout = QGridLayout(similarity_group)
        similarity_layout.setContentsMargins(14, 12, 14, 14)
        similarity_layout.setHorizontalSpacing(14)
        similarity_layout.setVerticalSpacing(10)
        similarity_layout.setColumnStretch(1, 1)
        
        # Name similarity
        self.compare_by_name_cb = QCheckBox("Compare by filename similarity")
        self.compare_by_name_cb.setChecked(True)
        similarity_layout.addWidget(self.compare_by_name_cb, 0, 0, 1, 2)
        
        similarity_layout.addWidget(QLabel("Name similarity threshold:"), 1, 0)
        self.name_similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.name_similarity_slider.setRange(50, 100)
        self.name_similarity_slider.setValue(80)
        self.name_similarity_slider.valueChanged.connect(self.update_name_similarity_label)
        similarity_layout.addWidget(self.name_similarity_slider, 1, 1)
        
        self.name_similarity_label = QLabel("80%")
        self.name_similarity_label.setMinimumWidth(48)
        similarity_layout.addWidget(self.name_similarity_label, 1, 2)
        
        # Size similarity
        self.compare_by_size_cb = QCheckBox("Compare by file size similarity")
        self.compare_by_size_cb.setChecked(True)
        similarity_layout.addWidget(self.compare_by_size_cb, 2, 0, 1, 2)
        
        similarity_layout.addWidget(QLabel("Size tolerance (MB):"), 3, 0)
        self.size_tolerance_spin = QSpinBox()
        self.size_tolerance_spin.setRange(0, 100)
        self.size_tolerance_spin.setValue(1)
        self.size_tolerance_spin.setMaximumWidth(160)
        similarity_layout.addWidget(self.size_tolerance_spin, 3, 1)
        
        # Content similarity
        self.compare_text_content_cb = QCheckBox("Compare text file content similarity")
        self.compare_text_content_cb.setChecked(True)
        similarity_layout.addWidget(self.compare_text_content_cb, 4, 0, 1, 2)
        
        similarity_layout.addWidget(QLabel("Content similarity threshold:"), 5, 0)
        self.content_similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.content_similarity_slider.setRange(70, 100)
        self.content_similarity_slider.setValue(90)
        self.content_similarity_slider.valueChanged.connect(self.update_content_similarity_label)
        similarity_layout.addWidget(self.content_similarity_slider, 5, 1)
        
        self.content_similarity_label = QLabel("90%")
        self.content_similarity_label.setMinimumWidth(48)
        similarity_layout.addWidget(self.content_similarity_label, 5, 2)
        
        layout.addWidget(similarity_group)
        
        # Help text with adaptive worker info
        mode_str = "GPU mode" if HAS_CUDA else "CPU mode"
        worker_explanation = (
            f"• Uses {OPTIMAL_WORKERS} worker threads ({mode_str})\n"
        )
        if HAS_CUDA:
            worker_explanation += f"• Reduced from {OPTIMAL_WORKERS_CPU_ONLY} threads to avoid GPU oversubscription\n"
        worker_explanation += "• ThreadPoolExecutor for I/O-bound operations"
        
        help_text = QLabel(
            "📋 Feature Guide:\n\n"
            "GPU Acceleration:\n"
            "• CUDA acceleration for perceptual image hashing\n"
            "• Batch processing for faster image comparison\n"
            "• Automatic fallback to CPU if GPU unavailable\n\n"
            "⚡ Parallel Processing:\n"
            f"{worker_explanation}\n\n"
            "🎯 Similarity Detection:\n"
            "• Filename similarity: Compares file name patterns\n"
            "• Size similarity: Finds files with similar sizes\n"
            "• Content similarity: Compares text file content"
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("settingsHelpText")
        layout.addWidget(help_text)
        
        layout.addStretch()
        self.tab_widget.addTab(settings_widget, "Settings")
    
    def on_phash_preset_changed(self, index):
        """Handle perceptual hash preset change"""
        preset_name = self.phash_preset_combo.currentData()
        if self.phash_thresholds and preset_name:
            self.phash_thresholds.set_preset(preset_name)
            # Update description
            desc = self.phash_thresholds.get_preset_description(preset_name)
            self.phash_preset_desc.setText(desc)
            # Update thresholds display
            self.update_phash_thresholds_display()
    
    def update_phash_thresholds_display(self):
        """Update the display of current pHash thresholds"""
        if self.phash_thresholds:
            thresholds = self.phash_thresholds.get_all_thresholds()
            display_items = [f"{k}: ≤{v}" for k, v in thresholds.items()]
            self.phash_thresholds_label.setText(" | ".join(display_items))
        else:
            self.phash_thresholds_label.setText("(threshold system not available)")
    
    def update_name_similarity_label(self, value):
        self.name_similarity_label.setText(f"{value}%")
    
    def update_content_similarity_label(self, value):
        self.content_similarity_label.setText(f"{value}%")

    def get_media_profiles_from_ui(self):
        self.media_profiles = merge_media_profiles(self.media_profiles)
        return self.media_profiles
    
    def get_scan_filters(self):
        """Get current scan filter settings"""
        filters = ScanFilters()
        
        # Get enabled file types
        filters.enabled_types = set()
        for type_name, cb in self.type_checkboxes.items():
            if cb.isChecked():
                filters.enabled_types.add(type_name)
        
        # Get custom extensions
        custom_ext = self.custom_ext_edit.text().strip()
        if custom_ext:
            filters.custom_extensions = set(
                ext.strip().lower() if ext.strip().startswith('.') else f'.{ext.strip().lower()}'
                for ext in custom_ext.split(',')
                if ext.strip()
            )
        
        # Get size limits
        filters.min_size_bytes = self.min_size_spin.value() * 1024  # KB to bytes
        max_mb = self.max_size_spin.value()
        filters.max_size_bytes = max_mb * 1024 * 1024 if max_mb > 0 else float('inf')
        
        # Get excluded folders
        exclude_text = self.exclude_folders_edit.text().strip()
        if exclude_text:
            filters.excluded_folders = set(
                folder.strip() for folder in exclude_text.split(',')
                if folder.strip()
            )
        
        return filters
    
    def select_directory(self):
        """Select directory to scan"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Scan")
        if directory:
            self.directory = directory
            self.dir_label.setText(f"📁 {directory}")
            self.scan_btn.setEnabled(True)
            self.scan_action.setEnabled(True)
            self.status_label.setText(f"Ready to scan: {directory}")
            self.status_bar.showMessage(f"Selected: {directory}", 3000)
    
    def select_move_destination(self):
        """Select destination directory for moving files"""
        destination = QFileDialog.getExistingDirectory(self, "Select Move Destination")
        if destination:
            self.move_destination = destination
            self.move_dest_label.setText(f"📦 Files will be moved to: {destination}")
            self.status_label.setText(f"Move destination set: {destination}")
            self.status_bar.showMessage(f"Move destination: {destination}", 3000)
            self._update_bulk_operation_buttons()
    
    def get_comparison_settings(self):
        """Get current comparison settings"""
        effective_hash_type = self.hash_combo.currentText()
        if self.preset_enabled:
            _, preset_config = self._get_active_preset_config()
            effective_hash_type = preset_config.get("hash_type", effective_hash_type)
        return {
            'hash_type': effective_hash_type,
            'find_similar': self.find_similar_cb.isChecked(),
            'compare_by_name': self.compare_by_name_cb.isChecked(),
            'name_similarity': self.name_similarity_slider.value() / 100.0,
            'compare_by_size': self.compare_by_size_cb.isChecked(),
            'size_tolerance': self.size_tolerance_spin.value(),
            'compare_text_content': self.compare_text_content_cb.isChecked(),
            'content_similarity': self.content_similarity_slider.value() / 100.0,
            'media_profiles': self.get_media_profiles_from_ui(),
        }
    
    def _on_hash_type_changed(self, hash_type: str):
        """Handle hash type selection change - show relevant status info."""
        if self._suppress_hash_focus_feedback:
            return
        status_messages = {
            'MD5': "📝 MD5: Fast exact-match hashing for all file types",
            'SHA256': "🔐 SHA256: Cryptographic hashing for exact duplicates",
            'Perceptual': f"🖼️ Perceptual: GPU-accelerated image/video similarity {'(GPU: ✅)' if HAS_CUDA else '(CPU mode)'}",
            'Audio': f"🎵 Audio: Chromaprint + MFCC fingerprinting {'(GPU MFCC: ✅)' if HAS_AUDIO_FINGERPRINTER else '(Not available)'}",
            'Semantic': f"📚 Semantic: Text similarity via embeddings {'(GPU: ✅)' if HAS_SEMANTIC_SIMILARITY else '(Not available)'}"
        }
        
        msg = status_messages.get(hash_type, "")
        if msg:
            self.status_bar.showMessage(msg, 5000)
            
        # Warn if feature not available
        if hash_type == 'Audio' and not HAS_AUDIO_FINGERPRINTER:
            QMessageBox.warning(self, "Feature Not Available", 
                "Audio fingerprinting requires pyacoustid and librosa.\n"
                "Install with: pip install pyacoustid librosa soundfile")
        elif hash_type == 'Semantic' and not HAS_SEMANTIC_SIMILARITY:
            QMessageBox.warning(self, "Feature Not Available",
                "Semantic similarity requires sentence-transformers.\n"
                "Install with: pip install sentence-transformers")
    
    def scan_duplicates(self):
        """Start scanning for duplicates and similar files"""
        if not self.directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first")
            return
        
        # Enable pause/cancel buttons and update toolbar
        self.scan_btn.setEnabled(False)
        self.scan_action.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_action.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.cancel_action.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get performance settings
        use_parallel = self.use_parallel_cb.isChecked()
        use_gpu = self.use_gpu_cb.isChecked() and HAS_CUDA
        
        mode_str = []
        if use_gpu:
            mode_str.append(f"GPU ({CUDA_BACKEND.upper()})")
        if use_parallel:
            mode_str.append(f"Parallel ({OPTIMAL_WORKERS} threads)")
        mode_display = " + ".join(mode_str) if mode_str else "Sequential"
        
        self.status_label.setText(f"Starting scan with {mode_display}...")
        self.status_bar.showMessage(f"Scanning {self.directory} with {mode_display}...")
        self.scan_start_time = time.time()
        
        # Clear previous results
        self.duplicates = {}
        self.similar_files = {}
        self.groups_tree.clear()
        self.similar_groups_tree.clear()
        self.duplicate_files_tree.clear()
        self.similar_files_tree.clear()
        self._update_bulk_operation_buttons()
        
        # Get comparison settings and filters
        comparison_settings = self.get_comparison_settings()
        scan_filters = self.get_scan_filters()
        
        # Start enhanced scan thread with GPU and parallel processing
        self.scan_thread = EnhancedScanThread(
            self.directory, 
            comparison_settings, 
            scan_filters,
            use_parallel=use_parallel,
            use_gpu=use_gpu
        )
        self.scan_thread.progress.connect(self.progress_bar.setValue)
        self.scan_thread.status_update.connect(self.status_label.setText)
        self.scan_thread.current_file.connect(self.update_current_file)
        self.scan_thread.eta_update.connect(self.eta_label.setText)
        self.scan_thread.files_processed.connect(self.update_files_count)
        self.scan_thread.gpu_status.connect(self.update_gpu_status)
        self.scan_thread.finished.connect(self.scan_completed)
        self.scan_thread.start()
    
    def update_gpu_status(self, status):
        """Update GPU status display"""
        self.status_label.setText(status)
    
    def update_current_file(self, filename):
        """Update current file being processed"""
        self.current_file_label.setText(f"{filename[:50]}..." if len(filename) > 50 else filename)
    
    def update_files_count(self, processed, total):
        """Update files processed count"""
        self.files_count_label.setText(f"{processed:,} / {total:,} files")
    
    def toggle_pause(self):
        """Toggle pause/resume scan"""
        if self.scan_thread and self.scan_thread.isRunning():
            if self.scan_thread.is_paused:
                self.scan_thread.resume()
                self.pause_btn.setText("Pause")
                self.pause_action.setText("Pause")
                self.status_label.setText("Scanning resumed...")
                self.status_bar.showMessage("Scan resumed", 2000)
            else:
                self.scan_thread.pause()
                self.pause_btn.setText("Resume")
                self.pause_action.setText("Resume")
                self.status_label.setText("Scan paused")
                self.status_bar.showMessage("Scan paused - Click Resume to continue", 5000)
    
    def cancel_scan(self):
        """Cancel the current scan"""
        if self.scan_thread and self.scan_thread.isRunning():
            reply = QMessageBox.question(
                self, "Cancel Scan",
                "Are you sure you want to cancel the scan?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.scan_thread.cancel()
                self.scan_btn.setEnabled(True)
                self.scan_action.setEnabled(True)
                self.pause_btn.setEnabled(False)
                self.pause_action.setEnabled(False)
                self.cancel_btn.setEnabled(False)
                self.cancel_action.setEnabled(False)
                self.progress_bar.setVisible(False)
                self.status_label.setText("Scan cancelled")
                self.status_bar.showMessage("Scan cancelled by user")
                self.current_file_label.setText("")
                self.eta_label.setText("")
                self.files_count_label.setText("")
    
    def scan_completed(self, duplicates, similar_files):
        """Handle scan completion"""
        self.duplicates = duplicates
        self.similar_files = similar_files
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        self.scan_action.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_action.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.cancel_action.setEnabled(False)
        self.current_file_label.setText("")
        self.eta_label.setText("")
        
        # Calculate scan duration
        if self.scan_start_time:
            duration = time.time() - self.scan_start_time
            duration_str = str(timedelta(seconds=int(duration)))
        else:
            duration_str = "Unknown"
        
        # Populate results
        self.populate_duplicate_groups()
        self.populate_similar_groups()
        self._update_step_counts()
        
        # Show summary
        total_duplicates = sum(len(files) for files in duplicates.values())
        total_similar = sum(len(group['files']) for group in similar_files.values())
        
        summary = f"Scan complete in {duration_str}! Found {len(duplicates)} duplicate groups ({total_duplicates} files)"
        if similar_files:
            summary += f" and {len(similar_files)} similar groups ({total_similar} files)"
        
        self.status_label.setText(summary)
        self.status_bar.showMessage(summary)
        
        # Update results status in status bar
        self.results_status.setText(f"{len(duplicates)} duplicates, {len(similar_files)} similar")
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(1)
        
        # Enable bulk delete button if duplicates found
        self.bulk_delete_btn.setEnabled(bool(duplicates))
        self._update_bulk_operation_buttons()
        
        if duplicates or similar_files:
            QMessageBox.information(self, "Scan Complete", summary)
        else:
            QMessageBox.information(self, "Scan Complete", "No duplicates or similar files found!")
    
    def _on_sort_changed(self, sort_text):
        """Handle sort option change - re-populate with new sorting"""
        if hasattr(self, 'duplicates') and self.duplicates:
            self.populate_duplicate_groups()
    
    def _on_file_type_filter_changed(self, filter_text):
        """Handle file type filter change - re-populate with filtering"""
        if hasattr(self, 'duplicates') and self.duplicates:
            self.populate_duplicate_groups()
    
    def _get_file_type_category(self, file_path):
        """Get file type category for a file"""
        ext = os.path.splitext(file_path)[1].lower()
        
        type_map = {
            'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico', '.svg'],
            'Documents': ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp'],
            'Videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'],
            'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
            'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
            'Code': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs'],
            'Text': ['.txt', '.md', '.rtf', '.csv', '.json', '.xml', '.html', '.css', '.log', '.yaml', '.yml']
        }
        
        for category, extensions in type_map.items():
            if ext in extensions:
                return category
        return 'Other'
    
    def _passes_file_type_filter(self, files, filter_type):
        """Check if a file group passes the file type filter"""
        if filter_type == "All Types":
            return True
        
        # Check if any file in the group matches the filter
        for file_path in files:
            if self._get_file_type_category(file_path) == filter_type:
                return True
        return False
    
    def _get_group_sort_key(self, hash_val, files, sort_option):
        """Get sort key for a duplicate group based on sort option."""
        try:
            if sort_option == "Size (Largest First)":
                total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
                return -total_size  # Negative for descending
            elif sort_option == "File Count":
                return -len(files)  # Negative for descending
            elif sort_option == "Newest First":
                newest = max((os.path.getmtime(f) for f in files if os.path.exists(f)), default=0)
                return -newest  # Negative for descending
            elif sort_option == "Oldest First":
                oldest = min((os.path.getmtime(f) for f in files if os.path.exists(f)), default=float('inf'))
                return oldest
            elif sort_option == "Name (A-Z)":
                # Sort by first file's name
                return os.path.basename(files[0]).lower() if files else ""
            else:
                return 0
        except Exception:
            return 0
    
    def _suggest_file_to_keep(self, files, quick=False):
        """
        Suggest which file to keep from a group of duplicates.

        Scoring criteria (higher = keep):
        - Higher quality score (resolution/bitrate when available)
        - Larger file size
        - Older file (original)

        quick=True skips _get_quality_score() (which triggers audio/image/video I/O)
        and ranks by size + age only.  Use this during tree population; the full score
        is available when the user actually selects the group (NR-2).
        """
        if not files:
            return None

        scored_files = []
        for f in files:
            if not os.path.exists(f):
                continue
            try:
                stat = os.stat(f)
                # NR-2: skip heavy I/O in quick mode; use 0.0 as quality placeholder
                quality = 0.0 if quick else self._get_quality_score(f)
                size = stat.st_size
                scored_files.append((quality, size, -stat.st_mtime, f))
            except Exception:
                continue

        if not scored_files:
            return files[0] if files else None

        # Return file with highest quality, then size, then oldest
        scored_files.sort(reverse=True)
        return scored_files[0][3]
    
    def populate_duplicate_groups(self):
        """Populate the duplicate groups tree with sorting and filtering"""
        self.groups_tree.clear()

        # Get current sort option
        sort_option = self.sort_combo.currentText() if hasattr(self, 'sort_combo') else "Size (Largest First)"

        # Get current file type filter
        file_type_filter = self.file_type_filter.currentText() if hasattr(self, 'file_type_filter') else "All Types"

        # Convert to list, filter, and sort
        groups_list = []
        for hash_val, files in self.duplicates.items():
            if self._passes_file_type_filter(files, file_type_filter):
                groups_list.append((hash_val, files))

        groups_list.sort(key=lambda x: self._get_group_sort_key(x[0], x[1], sort_option))

        # NR-3: Populate in batches so the event loop can breathe between chunks.
        # Each batch of 100 items is inserted with repaints disabled (NR-1) then the
        # next batch is scheduled via QTimer.singleShot(0) — fired when the event loop
        # is idle — so buttons stay responsive while the tree fills.
        self._populate_dup_batch(groups_list, 0)

        # Status update is independent of the tree fill, so do it immediately
        total_groups = len(self.duplicates)
        shown_groups = len(groups_list)
        if file_type_filter != "All Types":
            self.status_bar.showMessage(
                f"📊 Showing {shown_groups} of {total_groups} groups (filtered by {file_type_filter})",
                3000
            )

    def _populate_dup_batch(self, groups_list, start, batch_size=100):
        """Insert one batch of duplicate-group rows, then schedule the next (NR-3)."""
        end = min(start + batch_size, len(groups_list))
        self.groups_tree.setUpdatesEnabled(False)
        try:
            for i in range(start, end):
                hash_val, files = groups_list[i]
                group_item = QTreeWidgetItem()

                file_type = self._get_file_type_category(files[0]) if files else "Unknown"
                group_item.setText(0, f"📁 Group {i+1} ({file_type})")
                group_item.setText(1, str(len(files)))

                # Calculate total size (stat-only, no audio decode)
                total_size = 0
                for file_path in files:
                    try:
                        total_size += os.path.getsize(file_path)
                    except Exception:
                        pass

                group_item.setText(2, self.format_file_size(total_size))
                group_item.setData(0, Qt.ItemDataRole.UserRole, hash_val)

                # NR-2: quick=True skips _get_quality_score() I/O during population;
                # full quality score is computed lazily when the user selects the group.
                suggested = self._suggest_file_to_keep(files, quick=True)
                group_item.setData(1, Qt.ItemDataRole.UserRole, suggested)
                if suggested:
                    group_item.setToolTip(0, f"Keep: {os.path.basename(suggested)}")

                self.groups_tree.addTopLevelItem(group_item)
        finally:
            self.groups_tree.setUpdatesEnabled(True)

        if end < len(groups_list):
            QTimer.singleShot(0, lambda s=end: self._populate_dup_batch(groups_list, s))

    def populate_similar_groups(self):
        """Populate the similar files groups tree"""
        self.similar_groups_tree.clear()

        # NR-3: Pre-filter into a flat list so we can slice by index for batching
        items_list = []
        for i, (key, group_data) in enumerate(self.similar_files.items()):
            if self.similar_step_filter:
                match_type = group_data.get("match_type")
                if match_type not in self.similar_step_filter:
                    continue
            items_list.append((i, key, group_data))

        self._populate_sim_batch(items_list, 0)

    def _populate_sim_batch(self, items_list, start, batch_size=100):
        """Insert one batch of similar-group rows, then schedule the next (NR-3)."""
        end = min(start + batch_size, len(items_list))
        self.similar_groups_tree.setUpdatesEnabled(False)
        try:
            for idx in range(start, end):
                orig_i, key, group_data = items_list[idx]
                group_item = QTreeWidgetItem()
                match_type = group_data.get("match_type", "")
                if match_type == "content":
                    label = "Content Group"
                elif match_type == "same_name_diff_size":
                    label = "Same Name ≠ Size"
                elif match_type == "name_size":
                    label = "Name/Size Group"
                else:
                    label = "Similar Group"
                group_item.setText(0, f"{label} {orig_i+1}")
                group_item.setText(1, ", ".join(group_data.get('reasons', [])))
                group_item.setText(2, str(len(group_data.get('files', []))))
                group_item.setData(0, Qt.ItemDataRole.UserRole, key)
                # NR-2: quick=True; quality details computed lazily on selection
                suggested = self._suggest_file_to_keep(group_data.get("files", []), quick=True)
                if suggested:
                    group_item.setToolTip(0, f"Keep: {os.path.basename(suggested)}")

                self.similar_groups_tree.addTopLevelItem(group_item)
        finally:
            self.similar_groups_tree.setUpdatesEnabled(True)

        if end < len(items_list):
            QTimer.singleShot(0, lambda s=end: self._populate_sim_batch(items_list, s))
    
    def show_duplicate_group_files(self, current, previous):
        """Show files in selected duplicate group with sorting and keep suggestions"""
        if current is None:
            return
            
        hash_val = current.data(0, Qt.ItemDataRole.UserRole)
        suggested_keep = current.data(1, Qt.ItemDataRole.UserRole)
        files = self.duplicates.get(hash_val, [])
        suggested_keep = self._suggest_file_to_keep(files)
        
        self.duplicate_files_tree.clear()
        
        # Build file info list for sorting
        file_infos = []
        for file_path in files:
            try:
                stat = os.stat(file_path)
                quality_info = self._get_quality_info(file_path)
                file_infos.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'is_suggested': file_path == suggested_keep,
                    'type': quality_info.get("category", "other"),
                    'quality': quality_info.get("score", 0.0),
                    'details': quality_info.get("details", "N/A")
                })
            except Exception:
                quality_info = self._get_quality_info(file_path)
                file_infos.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                'size': 0,
                'mtime': 0,
                'is_suggested': file_path == suggested_keep,
                'type': quality_info.get("category", "other"),
                'quality': quality_info.get("score", 0.0),
                'details': quality_info.get("details", "N/A")
            })
        
        # Sort: suggested keep first, then by modification time (oldest first)
        file_infos.sort(key=lambda x: (not x['is_suggested'], x['mtime']))
        
        for info in file_infos:
            item = QTreeWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            
            # Don't check the suggested keep file by default
            if info['is_suggested']:
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setText(1, f"⭐ {info['name']}")
                # Make suggested file stand out
                font = item.font(1)
                font.setBold(True)
                item.setFont(1, font)
            else:
                item.setCheckState(0, Qt.CheckState.Checked)  # Pre-check duplicates for deletion
                item.setText(1, info['name'])
            
            item.setText(2, info['path'])
            item.setText(3, info['type'].capitalize())
            item.setText(4, f"{info['quality']:.2f}")
            item.setText(5, info['details'])
            item.setText(6, self.format_file_size(info['size']))
            
            # Add modification date
            if info['mtime'] > 0:
                mod_date = datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M')
                item.setText(7, mod_date)
            else:
                item.setText(7, "Unknown")
            
            item.setData(0, Qt.ItemDataRole.UserRole, info['path'])
            self.duplicate_files_tree.addTopLevelItem(item)
        
        # Auto-resize columns
        for i in range(self.duplicate_files_tree.columnCount()):
            self.duplicate_files_tree.resizeColumnToContents(i)
    
    def show_similar_group_files(self, current, previous):
        """Show files in selected similar group with sorting"""
        if current is None:
            return
            
        group_key = current.data(0, Qt.ItemDataRole.UserRole)
        group_data = self.similar_files.get(group_key, {})
        files = group_data.get('files', [])
        suggested_keep = self._suggest_file_to_keep(files)
        
        self.similar_files_tree.clear()
        
        # Build file info list for sorting
        file_infos = []
        for file_path in files:
            try:
                stat = os.stat(file_path)
                quality_info = self._get_quality_info(file_path)
                file_infos.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'is_suggested': file_path == suggested_keep,
                    'type': quality_info.get("category", "other"),
                    'quality': quality_info.get("score", 0.0),
                    'details': quality_info.get("details", "N/A")
                })
            except Exception:
                quality_info = self._get_quality_info(file_path)
                file_infos.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'size': 0,
                    'mtime': 0,
                    'is_suggested': file_path == suggested_keep,
                    'type': quality_info.get("category", "other"),
                    'quality': quality_info.get("score", 0.0),
                    'details': quality_info.get("details", "N/A")
                })
        
        # Sort: suggested keep first, then by size
        file_infos.sort(key=lambda x: (not x['is_suggested'], -x['size'], x['name'].lower()))
        
        for info in file_infos:
            item = QTreeWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            
            # Suggested keep: don't pre-check, mark with ⭐
            if info['is_suggested']:
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setText(1, f"⭐ {info['name']}")
                font = item.font(1)
                font.setBold(True)
                item.setFont(1, font)
            else:
                item.setCheckState(0, Qt.CheckState.Checked)  # Pre-check for deletion
                item.setText(1, info['name'])
            
            item.setText(2, info['path'])
            item.setText(3, info['type'].capitalize())
            item.setText(4, f"{info['quality']:.2f}")
            item.setText(5, info['details'])
            item.setText(6, self.format_file_size(info['size']))
            
            # Add modification date
            if info['mtime'] > 0:
                mod_date = datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M')
                item.setText(7, mod_date)
            else:
                item.setText(7, "Unknown")
            
            item.setData(0, Qt.ItemDataRole.UserRole, info['path'])
            self.similar_files_tree.addTopLevelItem(item)
        
        # Auto-resize columns
        for i in range(self.similar_files_tree.columnCount()):
            self.similar_files_tree.resizeColumnToContents(i)
    
    def show_file_preview(self, current, previous):
        """Show file preview in preview area"""
        if current is None:
            return
            
        file_path = current.data(0, Qt.ItemDataRole.UserRole)
        if not file_path:
            return
        
        # Determine which preview area to use
        sender = self.sender()
        if sender == self.duplicate_files_tree:
            preview_area = self.duplicate_preview_area
        else:
            preview_area = self.similar_preview_area
        
        try:
            stat = os.stat(file_path)
            size_mb = stat.st_size / (1024 * 1024)
            
            preview_text = f"📄 File: {os.path.basename(file_path)}\n"
            preview_text += f"📁 Path: {file_path}\n"
            preview_text += f"📏 Size: {self.format_file_size(stat.st_size)}\n"
            preview_text += f"📅 Modified: {self.format_timestamp(stat.st_mtime)}\n\n"
            
            ext = os.path.splitext(file_path)[1].lower()
            
            # Text file preview
            if ext in ['.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.xml']:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1000)  # First 1000 characters
                        preview_text += f"📝 Content preview:\n{'-'*40}\n{content}"
                        if len(content) == 1000:
                            preview_text += "\n\n... (truncated)"
                except Exception:
                    preview_text += "❌ Cannot preview text content"
                    
            # Image file preview
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                preview_text += "🖼️ Image file - Open in image viewer for preview"
                if HAS_PILLOW:
                    try:
                        img = Image.open(file_path)
                        preview_text += f"\n📐 Dimensions: {img.size[0]} x {img.size[1]}"
                        preview_text += f"\n🎨 Mode: {img.mode}"
                    except Exception:
                        pass
            
            # Video file info
            elif ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']:
                preview_text += "🎬 Video file"
            
            # Audio file info
            elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                preview_text += "🎵 Audio file"
            
            else:
                preview_text += f"📄 File type: {ext or 'Unknown'}"
                
            preview_area.setText(preview_text)
            
        except Exception as e:
            preview_area.setText(f"❌ Error reading file: {e}")
    
    def update_selection_buttons(self):
        """Update the state of selection-based buttons"""
        has_selection = self.has_selected_files()
        self.delete_selected_btn.setEnabled(has_selection)
        self.recycle_selected_btn.setEnabled(has_selection and HAS_SEND2TRASH)
        self.move_selected_btn.setEnabled(has_selection and self.move_destination is not None)
        self._update_bulk_operation_buttons()

    def _update_bulk_operation_buttons(self):
        """Update bulk operation button states based on current context."""
        has_duplicates = bool(getattr(self, 'duplicates', {}))
        has_destination = self.move_destination is not None
        self.bulk_move_btn.setEnabled(has_duplicates and has_destination)
        self.audio_smart_move_btn.setEnabled(has_duplicates and has_destination and self._has_audio_duplicates())

    def _is_audio_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.audio_extensions

    def _has_audio_duplicates(self):
        """Check if any duplicate group contains at least two audio files."""
        for files in self.duplicates.values():
            audio_files = [f for f in files if self._is_audio_file(f)]
            if len(audio_files) >= 2:
                return True
        return False

    def _get_quality_score(self, file_path):
        """Compute a quality score for keep-best decisions."""
        info = self._get_quality_info(file_path)
        return info.get("score", 0.0)

    def _get_quality_info(self, file_path):
        """Get quality score and display details for a file."""
        cached = self._quality_cache.get(file_path)
        if isinstance(cached, dict):
            return cached

        score = 0.0
        details = "N/A"
        category = self._get_file_type_category(file_path).lower()

        try:
            if category == "audio":
                meta = self.audio_quality_analyzer.analyze_audio_quality(file_path)
                if meta and isinstance(meta, dict):
                    score = float(meta.get("quality_score", 0.0))
                    fmt = meta.get("format", "unknown").upper()
                    bitrate = meta.get("bitrate_kbps", 0)
                    sample_rate = meta.get("sample_rate_hz", 0)
                    details = f"{fmt} • {bitrate:.0f} kbps • {sample_rate / 1000:.1f} kHz"
            elif category == "images" and HAS_PILLOW:
                with Image.open(file_path) as img:
                    width, height = img.size
                    score = (width * height) / 1_000_000.0
                    details = f"{width}x{height}"
            elif category == "videos" and HAS_CV2:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    score = (width * height) / 1_000_000.0
                    details = f"{width}x{height}"
                cap.release()
        except Exception:
            score = 0.0
            details = "N/A"

        info = {
            "score": score,
            "details": details,
            "category": category
        }
        self._quality_cache[file_path] = info
        return info

    def has_selected_files(self):
        """Check if any files are selected in either tree"""
        # Check duplicate files tree
        for i in range(self.duplicate_files_tree.topLevelItemCount()):
            item = self.duplicate_files_tree.topLevelItem(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                return True
        
        # Check similar files tree
        for i in range(self.similar_files_tree.topLevelItemCount()):
            item = self.similar_files_tree.topLevelItem(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                return True
        
        return False
    
    def get_selected_files(self):
        """Get list of selected file paths"""
        selected_files = []
        
        # Get from duplicate files tree
        for i in range(self.duplicate_files_tree.topLevelItemCount()):
            item = self.duplicate_files_tree.topLevelItem(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                file_path = item.data(0, Qt.ItemDataRole.UserRole)
                if file_path:
                    selected_files.append(file_path)
        
        # Get from similar files tree
        for i in range(self.similar_files_tree.topLevelItemCount()):
            item = self.similar_files_tree.topLevelItem(i)
            if item.checkState(0) == Qt.CheckState.Checked:
                file_path = item.data(0, Qt.ItemDataRole.UserRole)
                if file_path:
                    selected_files.append(file_path)
        
        return selected_files
    
    def select_all_in_group(self):
        """Select all files in the currently active group"""
        current_tab = self.results_tabs.currentIndex()
        
        if current_tab == 0:  # Duplicates tab
            tree = self.duplicate_files_tree
        else:  # Similar files tab
            tree = self.similar_files_tree
        
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            item.setCheckState(0, Qt.CheckState.Checked)
        
        self.update_selection_buttons()
    
    def clear_selection(self):
        """Clear all selections"""
        for tree in [self.duplicate_files_tree, self.similar_files_tree]:
            for i in range(tree.topLevelItemCount()):
                item = tree.topLevelItem(i)
                item.setCheckState(0, Qt.CheckState.Unchecked)
        
        self.update_selection_buttons()
    
    def delete_selected_files(self):
        """Delete selected files with undo support"""
        selected_files = self.get_selected_files()
        
        if not selected_files:
            QMessageBox.warning(self, "Warning", "No files selected for deletion")
            return
        
        # Check if undo is enabled
        undo_enabled = self.enterprise_config.get("enable_undo", True)
        
        warning_text = f"Are you sure you want to delete {len(selected_files)} selected files?"
        if undo_enabled:
            warning_text += "\n\n✅ Undo is enabled - files will be backed up first."
        else:
            warning_text += "\n\n⚠️ This action cannot be undone!"
        
        reply = QMessageBox.question(
            self, 
            "Confirm Delete",
            warning_text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            backed_up_count = 0
            failed_files = []
            
            for file_path in selected_files:
                try:
                    # Backup file if undo is enabled
                    if undo_enabled:
                        if self.undo_manager.backup_file(file_path):
                            backed_up_count += 1
                    
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    failed_files.append(f"{file_path}: {e}")
            
            # Show results
            if deleted_count > 0:
                undo_msg = f"\n📦 {backed_up_count} files backed up for undo." if backed_up_count > 0 else ""
                QMessageBox.information(
                    self, 
                    "Delete Complete", 
                    f"✅ Successfully deleted {deleted_count} files.{undo_msg}"
                )
            
            if failed_files:
                QMessageBox.warning(
                    self,
                    "Some Deletions Failed",
                    f"❌ Failed to delete {len(failed_files)} files:\n\n" + 
                    "\n".join(failed_files[:5]) +  # Show first 5 errors
                    (f"\n... and {len(failed_files)-5} more" if len(failed_files) > 5 else "")
                )
            
            # Update undo state
            self._update_undo_state()
            
            # Refresh the view
            self.refresh_current_view()
    
    def show_bulk_delete_preview(self):
        """Show preview dialog for bulk deleting all duplicates (keeps one per group)"""
        if not self.duplicates:
            QMessageBox.warning(self, "No Duplicates", "No duplicate files found to process.")
            return
        
        # Calculate what will be kept vs deleted
        files_to_keep = []
        files_to_delete = []
        total_space_freed = 0
        
        for hash_val, files in self.duplicates.items():
            if len(files) < 2:
                continue
            
            # Get the suggested file to keep
            suggested = self._suggest_file_to_keep(files)
            
            for f in files:
                if f == suggested:
                    files_to_keep.append(f)
                else:
                    files_to_delete.append(f)
                    try:
                        total_space_freed += os.path.getsize(f)
                    except OSError:
                        pass
        
        if not files_to_delete:
            QMessageBox.information(self, "Nothing to Delete", "No duplicate files to delete.")
            return
        
        # Create preview dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("🗑️ Bulk Delete Duplicates - Preview")
        dialog.resize(1000, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Summary section
        summary_group = QGroupBox("📊 Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(QLabel(f"<b>Groups:</b> {len(self.duplicates)}"))
        summary_layout.addWidget(QLabel(f"<b>Files to KEEP:</b> {len(files_to_keep)} (marked with ⭐)"))
        summary_layout.addWidget(QLabel(f"<b>Files to DELETE:</b> {len(files_to_delete)}"))
        summary_layout.addWidget(QLabel(f"<b>Space to free:</b> {self.format_file_size(total_space_freed)}"))
        layout.addWidget(summary_group)
        
        # Preview tree - show what will be kept and deleted
        preview_tree = QTreeWidget()
        preview_tree.setHeaderLabels(["File", "Action", "Type", "Quality", "Details", "Size", "Path"])
        preview_tree.setAlternatingRowColors(True)
        
        for hash_val, files in self.duplicates.items():
            if len(files) < 2:
                continue
            
            suggested = self._suggest_file_to_keep(files)
            group_item = QTreeWidgetItem()
            group_item.setText(0, f"📁 Group ({len(files)} files)")
            preview_tree.addTopLevelItem(group_item)
            
            for f in files:
                item = QTreeWidgetItem(group_item)
                item.setText(0, os.path.basename(f))
                
                if f == suggested:
                    item.setText(1, "⭐ KEEP")
                    item.setForeground(1, QColor(0, 150, 0))  # Green
                else:
                    item.setText(1, "🗑️ DELETE")
                    item.setForeground(1, QColor(200, 0, 0))  # Red
                
                info = self._get_quality_info(f)
                item.setText(2, info.get("category", "other").capitalize())
                item.setText(3, f"{info.get('score', 0.0):.2f}")
                item.setText(4, info.get("details", "N/A"))

                try:
                    item.setText(5, self.format_file_size(os.path.getsize(f)))
                except OSError:
                    item.setText(5, "Unknown")
                
                item.setText(6, os.path.dirname(f))
        
        preview_tree.expandAll()
        for i in range(preview_tree.columnCount()):
            preview_tree.resizeColumnToContents(i)
        layout.addWidget(preview_tree)
        
        # Options
        options_layout = QHBoxLayout()
        
        use_recycle_cb = QCheckBox("♻️ Send to Recycle Bin (safer)")
        use_recycle_cb.setChecked(HAS_SEND2TRASH)
        use_recycle_cb.setEnabled(HAS_SEND2TRASH)
        if not HAS_SEND2TRASH:
            use_recycle_cb.setToolTip("Install send2trash: pip install send2trash")
        options_layout.addWidget(use_recycle_cb)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        
        buttons_layout.addStretch()
        
        delete_btn = QPushButton(f"🗑️ Delete {len(files_to_delete)} Files")
        delete_btn.setStyleSheet("background-color: #d32f2f; color: white; padding: 8px 16px;")
        delete_btn.clicked.connect(lambda: self._execute_bulk_delete(files_to_delete, use_recycle_cb.isChecked(), dialog))
        buttons_layout.addWidget(delete_btn)
        
        layout.addLayout(buttons_layout)
        
        dialog.exec()
    
    def _execute_bulk_delete(self, files_to_delete, use_recycle, dialog):
        """Execute bulk deletion of files"""
        reply = QMessageBox.question(
            self,
            "⚠️ Final Confirmation",
            f"This will permanently {'send to recycle bin' if use_recycle else 'delete'} {len(files_to_delete)} files.\n\n"
            "This action cannot be easily undone.\n\n"
            "Are you absolutely sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        dialog.accept()
        
        # Clear previews and force GC to release file handles
        self.duplicate_preview_area.clear()
        self.similar_preview_area.clear()
        import gc
        gc.collect()
        
        deleted_count = 0
        failed_files = []
        
        # Use undo manager if enabled
        undo_enabled = self.enterprise_config.get("enable_undo", True)
        backed_up_count = 0
        
        for file_path in files_to_delete:
            try:
                # Backup if undo is enabled
                if undo_enabled and not use_recycle:
                    if self.undo_manager.backup_file(file_path):
                        backed_up_count += 1
                
                # Check if file exists before attempting deletion
                # (Use the original path which likely has the prefix and is most reliable for existence check)
                if not os.path.exists(file_path):
                    continue

                # Prepare path variants
                # send2trash typically prefers standard paths (no prefix)
                # os.remove for long paths REQUIRED the prefix
                
                normalized_path = file_path.replace("\\\\?\\", "") if file_path.startswith("\\\\?\\") else file_path
                prefixed_path = file_path if file_path.startswith("\\\\?\\") else f"\\\\?\\{os.path.abspath(file_path)}"
                
                if use_recycle:
                    # For Recycle Bin: Try normalized first, then prefixed
                    paths_to_try = [normalized_path, prefixed_path]
                    # Dedup if they are the same (short path)
                    if normalized_path == prefixed_path:
                        paths_to_try = [normalized_path]
                else:
                    # For Permanent Delete: Try prefixed first (better for long paths), then normalized
                    paths_to_try = [prefixed_path, normalized_path]
                    if normalized_path == prefixed_path:
                        paths_to_try = [prefixed_path]

                success = False
                last_error = None
                
                for path_variant in paths_to_try:
                    try:
                        if use_recycle and HAS_SEND2TRASH:
                            # Use function from dependencies if available, or direct import
                            if 'send2trash_func' in globals() and send2trash_func:
                                send2trash_func(path_variant)
                            else:
                                send2trash(path_variant)
                        else:
                            os.chmod(path_variant, 0o777)  # Try to remove read-only attribute
                            os.remove(path_variant)
                        
                        success = True
                        break
                    except Exception as e:
                        last_error = e
                
                if success:
                    deleted_count += 1
                elif last_error:
                    # Reraise the last error to be caught by the outer try/except
                    raise last_error
                
            except Exception as e:
                failed_files.append(f"{os.path.basename(file_path)}: {e}")
        
        # Show results
        result_msg = f"✅ Successfully {'recycled' if use_recycle else 'deleted'} {deleted_count} files."
        if backed_up_count > 0:
            result_msg += f"\n📦 {backed_up_count} files backed up for undo."
        
        QMessageBox.information(self, "Bulk Delete Complete", result_msg)
        
        if failed_files:
            QMessageBox.warning(
                self,
                "Some Deletions Failed",
                f"❌ Failed to delete {len(failed_files)} files:\n\n" +
                "\n".join(failed_files[:10]) +
                (f"\n... and {len(failed_files)-10} more" if len(failed_files) > 10 else "")
            )
        
        # Update undo state
        self._update_undo_state()
        
        # Refresh the view
        self.refresh_current_view()

    
    def show_bulk_move_preview(self):
        """Show preview dialog for bulk moving all duplicates (keeps one per group)"""
        if not self.duplicates:
            QMessageBox.warning(self, "No Duplicates", "No duplicate files found to process.")
            return
        if not self.move_destination:
            QMessageBox.warning(self, "Warning", "Please select a move destination first")
            return

        files_to_keep = []
        files_to_move = []
        total_space_freed = 0
        group_plans = []

        for hash_val, files in self.duplicates.items():
            if len(files) < 2:
                continue
            keep_file = self._suggest_file_to_keep(files)
            if keep_file:
                files_to_keep.append((keep_file, "Best quality/size/oldest"))

            for f in files:
                if f == keep_file:
                    continue
                files_to_move.append(f)
                try:
                    total_space_freed += os.path.getsize(f)
                except Exception:
                    pass

            group_plans.append({
                "files": files,
                "keep": keep_file,
                "reason": "Best quality/size/oldest"
            })

        if not files_to_move:
            QMessageBox.information(self, "Nothing to Move", "No duplicate files to move.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("🚀 Bulk Move Duplicates - Preview")
        dialog.resize(1050, 650)

        layout = QVBoxLayout(dialog)

        summary_group = QGroupBox("📊 Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(QLabel(f"<b>Groups to process:</b> {len(group_plans)}"))
        summary_layout.addWidget(QLabel(f"<b>Files to KEEP:</b> {len(files_to_keep)} (kept in place)"))
        summary_layout.addWidget(QLabel(f"<b>Files to MOVE:</b> {len(files_to_move)}"))
        summary_layout.addWidget(QLabel(f"<b>Space to free:</b> {self.format_file_size(total_space_freed)}"))
        summary_layout.addWidget(QLabel(f"<b>Destination:</b> {self.move_destination}"))
        layout.addWidget(summary_group)

        preview_tree = QTreeWidget()
        preview_tree.setHeaderLabels(["File", "Action", "Reason", "Type", "Quality", "Details", "Size", "Path"])
        preview_tree.setAlternatingRowColors(True)

        for group in group_plans:
            files = group["files"]
            keep_file = group["keep"]
            group_item = QTreeWidgetItem()
            group_item.setText(0, f"📁 Group ({len(files)} files)")
            preview_tree.addTopLevelItem(group_item)

            for f in files:
                item = QTreeWidgetItem(group_item)
                item.setText(0, os.path.basename(f))

                if f == keep_file:
                    item.setText(1, "⭐ KEEP")
                    item.setText(2, group["reason"])
                    item.setForeground(1, QColor(0, 150, 0))
                    font = item.font(0)
                    font.setBold(True)
                    item.setFont(0, font)
                else:
                    item.setText(1, "📦 MOVE")
                    item.setText(2, "Duplicate")
                    item.setForeground(1, QColor(200, 120, 0))

                info = self._get_quality_info(f)
                item.setText(3, info.get("category", "other").capitalize())
                item.setText(4, f"{info.get('score', 0.0):.2f}")
                item.setText(5, info.get("details", "N/A"))

                try:
                    item.setText(6, self.format_file_size(os.path.getsize(f)))
                except OSError:
                    item.setText(6, "Unknown")
                item.setText(7, os.path.dirname(f))

        preview_tree.expandAll()
        for i in range(preview_tree.columnCount()):
            preview_tree.resizeColumnToContents(i)
        layout.addWidget(preview_tree)

        buttons_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)

        buttons_layout.addStretch()

        proceed_btn = QPushButton(f"🚀 Move {len(files_to_move)} Files")
        proceed_btn.setStyleSheet("background-color: #2e7d32; color: white; padding: 8px 16px;")
        proceed_btn.clicked.connect(lambda: self._start_bulk_move_duplicates(files_to_move, dialog))
        buttons_layout.addWidget(proceed_btn)
        layout.addLayout(buttons_layout)

        dialog.exec()

    def _start_bulk_move_duplicates(self, files_to_move, preview_dialog):
        """Start bulk move operation with progress dialog."""
        # MV-1: preview_dialog may be None when called from move_selected_files
        if preview_dialog is not None:
            preview_dialog.accept()

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("🚀 Bulk Move Duplicates")
        progress_dialog.resize(500, 180)
        progress_layout = QVBoxLayout(progress_dialog)

        status_label = QLabel("Preparing move operation...")
        progress_layout.addWidget(status_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, len(files_to_move))
        progress_layout.addWidget(progress_bar)

        buttons_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        buttons_layout.addStretch()
        buttons_layout.addWidget(cancel_btn)
        progress_layout.addLayout(buttons_layout)

        thread = BulkMoveThread(files_to_move, self.move_destination)
        self._active_move_thread = thread

        def on_progress(done, total):
            progress_bar.setValue(done)
            progress_bar.setMaximum(total)

        def on_status(text):
            status_label.setText(text)

        def on_cancel():
            status_label.setText("Cancelling...")
            cancel_btn.setEnabled(False)
            thread.cancel()

        def on_finished(result):
            progress_dialog.accept()
            self._handle_bulk_move_results(result, operation_name="Bulk Move Duplicates")

        thread.progress.connect(on_progress)
        thread.status.connect(on_status)
        thread.result.connect(on_finished)
        cancel_btn.clicked.connect(on_cancel)

        thread.start()
        progress_dialog.exec()

    def show_audio_move_preview(self):
        """Show preview dialog for audio smart move (keep best quality)."""
        if not self.duplicates:
            QMessageBox.warning(self, "No Duplicates", "No duplicate files found to process.")
            return
        if not self.move_destination:
            QMessageBox.warning(self, "Warning", "Please select a move destination first")
            return
        if not self._has_audio_duplicates():
            QMessageBox.information(self, "No Audio Duplicates", "No audio duplicate groups found.")
            return

        audio_groups = []
        warnings = []
        total_analyzed = 0
        total_to_move = 0
        total_kept = 0
        total_skipped = 0
        improvement_scores = []
        kept_log = []

        for hash_val, files in self.duplicates.items():
            audio_files = [f for f in files if self._is_audio_file(f)]
            if len(audio_files) < 2:
                continue

            meta_map = {}
            valid_files = []
            skipped_files = []

            for f in audio_files:
                meta = self.audio_quality_analyzer.analyze_audio_quality(f)
                if meta is None:
                    warnings.append(f"{f}: Unable to analyze audio metadata")
                    skipped_files.append(f)
                    continue
                total_analyzed += 1
                if meta.get('analysis_error'):
                    warnings.append(f"{f}: Corrupted or unreadable audio metadata")
                    skipped_files.append(f)
                else:
                    valid_files.append(f)
                meta_map[f] = meta

            if len(valid_files) < 2:
                total_skipped += len(skipped_files)
                continue

            max_score = max(meta_map[f]['quality_score'] for f in valid_files)
            best_file = None
            for f in audio_files:
                if f in valid_files and meta_map[f]['quality_score'] == max_score:
                    best_file = f
                    break
            if best_file is None:
                best_file = valid_files[0]

            other_scores = [meta_map[f]['quality_score'] for f in valid_files if f != best_file]
            if other_scores:
                improvement_scores.append(max_score - (sum(other_scores) / len(other_scores)))
            else:
                improvement_scores.append(0.0)

            move_files = [f for f in valid_files if f != best_file]
            total_to_move += len(move_files)
            total_kept += 1
            total_skipped += len(skipped_files)

            best_meta = meta_map.get(best_file, {})
            kept_log.append(
                f"KEEP: {best_file} | {best_meta.get('format', 'unknown').upper()} "
                f"| {best_meta.get('bitrate_kbps', 0):.0f} kbps | "
                f"{best_meta.get('sample_rate_hz', 0) / 1000:.1f} kHz | "
                f"Score {best_meta.get('quality_score', 0):.1f}"
            )

            audio_groups.append({
                "files": audio_files,
                "valid_files": valid_files,
                "skipped_files": skipped_files,
                "meta": meta_map,
                "best_file": best_file,
                "move_files": move_files,
                "max_score": max_score
            })

        if not audio_groups:
            QMessageBox.information(self, "No Audio Groups", "No eligible audio duplicate groups found.")
            return

        avg_improvement = sum(improvement_scores) / len(improvement_scores) if improvement_scores else 0.0

        dialog = QDialog(self)
        dialog.setWindowTitle("🎵 Audio Smart Move - Preview")
        dialog.resize(980, 680)

        layout = QVBoxLayout(dialog)

        summary_group = QGroupBox("📊 Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(QLabel(f"<b>Audio groups:</b> {len(audio_groups)}"))
        summary_layout.addWidget(QLabel(f"<b>Audio files analyzed:</b> {total_analyzed}"))
        summary_layout.addWidget(QLabel(f"<b>Files to KEEP:</b> {total_kept}"))
        summary_layout.addWidget(QLabel(f"<b>Files to MOVE:</b> {total_to_move}"))
        summary_layout.addWidget(QLabel(f"<b>Skipped (corrupted/unreadable):</b> {total_skipped}"))
        summary_layout.addWidget(QLabel(f"<b>Avg quality improvement:</b> {avg_improvement:.1f}"))
        summary_layout.addWidget(QLabel(f"<b>Destination:</b> {self.move_destination}"))
        layout.addWidget(summary_group)

        preview_tree = QTreeWidget()
        preview_tree.setHeaderLabels(["File", "Format", "Bitrate", "Sample Rate", "Size", "Quality", "Action"])
        preview_tree.setAlternatingRowColors(True)

        for group in audio_groups:
            group_item = QTreeWidgetItem()
            group_item.setText(0, f"🎵 Audio Group ({len(group['files'])} files)")
            preview_tree.addTopLevelItem(group_item)

            max_score = group["max_score"]
            for f in group["files"]:
                item = QTreeWidgetItem(group_item)
                item.setText(0, os.path.basename(f))
                meta = group["meta"].get(f, {})

                fmt = (meta.get('format') or 'unknown').upper()
                item.setText(1, fmt)
                item.setText(2, f"{meta.get('bitrate_kbps', 0):.0f} kbps")
                item.setText(3, self.format_sample_rate(meta.get('sample_rate_hz', 0)))
                item.setText(4, self.format_file_size(meta.get('file_size_bytes', 0)))

                if f in group["skipped_files"]:
                    item.setText(5, "N/A")
                    item.setText(6, "⚠️ SKIP")
                    item.setForeground(6, QColor(200, 120, 0))
                    item.setToolTip(0, "Corrupted or unreadable audio metadata")
                    continue

                score = meta.get('quality_score', 0.0)
                quality_pct = (score / max_score) * 100 if max_score > 0 else 0.0
                item.setText(5, f"{quality_pct:.0f}%")
                item.setToolTip(5, f"Quality score: {score:.1f}")

                if f == group["best_file"]:
                    item.setText(6, "⭐ KEEP")
                    item.setForeground(6, QColor(0, 150, 0))
                    font = item.font(0)
                    font.setBold(True)
                    item.setFont(0, font)
                else:
                    item.setText(6, "📦 MOVE")
                    item.setForeground(6, QColor(200, 0, 0))

        preview_tree.expandAll()
        for i in range(preview_tree.columnCount()):
            preview_tree.resizeColumnToContents(i)
        layout.addWidget(preview_tree)

        buttons_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(cancel_btn)
        buttons_layout.addStretch()

        proceed_btn = QPushButton(f"🎵 Move {total_to_move} Audio Files")
        proceed_btn.setStyleSheet("background-color: #6a1b9a; color: white; padding: 8px 16px;")
        proceed_btn.clicked.connect(lambda: self._start_audio_smart_move(audio_groups, avg_improvement, kept_log, warnings, dialog))
        buttons_layout.addWidget(proceed_btn)
        layout.addLayout(buttons_layout)

        dialog.exec()

    def _start_audio_smart_move(self, audio_groups, avg_improvement, kept_log, warnings, preview_dialog):
        """Start audio smart move operation."""
        preview_dialog.accept()

        files_to_move = []
        total_analyzed = 0
        total_kept = 0

        for group in audio_groups:
            files_to_move.extend(group["move_files"])
            total_analyzed += len(group["valid_files"]) + len(group["skipped_files"])
            total_kept += 1

        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("🎵 Audio Smart Move")
        progress_dialog.resize(500, 180)
        progress_layout = QVBoxLayout(progress_dialog)

        status_label = QLabel("Preparing audio move operation...")
        progress_layout.addWidget(status_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, len(files_to_move))
        progress_layout.addWidget(progress_bar)

        buttons_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        buttons_layout.addStretch()
        buttons_layout.addWidget(cancel_btn)
        progress_layout.addLayout(buttons_layout)

        thread = BulkMoveThread(files_to_move, self.move_destination)
        self._active_move_thread = thread

        def on_progress(done, total):
            progress_bar.setValue(done)
            progress_bar.setMaximum(total)

        def on_status(text):
            status_label.setText(text)

        def on_cancel():
            status_label.setText("Cancelling...")
            cancel_btn.setEnabled(False)
            thread.cancel()

        def on_finished(result):
            progress_dialog.accept()
            self._handle_audio_move_results(
                result,
                total_analyzed=total_analyzed,
                total_kept=total_kept,
                avg_improvement=avg_improvement,
                kept_log=kept_log,
                warnings=warnings
            )

        thread.progress.connect(on_progress)
        thread.status.connect(on_status)
        thread.result.connect(on_finished)
        cancel_btn.clicked.connect(on_cancel)

        thread.start()
        progress_dialog.exec()

    def _handle_bulk_move_results(self, result, operation_name="Bulk Move"):
        """Handle results of bulk move operations."""
        moved_count = result.get("moved_count", 0)
        failed_files = result.get("failed_files", [])
        moved_pairs = result.get("moved_pairs", [])
        space_freed = result.get("space_freed", 0)
        cancelled = result.get("cancelled", False)

        undo_enabled = self.enterprise_config.get("enable_undo", True)
        if undo_enabled and moved_pairs:
            for original_path, new_path in moved_pairs:
                self.undo_manager.record_move(original_path, new_path)
            self._update_undo_state()

        if cancelled:
            QMessageBox.information(self, operation_name, "⚠️ Operation cancelled by user.")

        if moved_count > 0:
            QMessageBox.information(
                self,
                operation_name,
                f"✅ Successfully moved {moved_count} files.\n"
                f"📦 Space freed: {self.format_file_size(space_freed)}"
            )

        if failed_files:
            self._show_error_dialog("Some Moves Failed", failed_files)

        self.refresh_current_view()

    def _handle_audio_move_results(self, result, total_analyzed, total_kept, avg_improvement, kept_log, warnings):
        """Handle results of audio smart move operation."""
        moved_count = result.get("moved_count", 0)
        failed_files = result.get("failed_files", [])
        moved_pairs = result.get("moved_pairs", [])
        cancelled = result.get("cancelled", False)

        undo_enabled = self.enterprise_config.get("enable_undo", True)
        if undo_enabled and moved_pairs:
            for original_path, new_path in moved_pairs:
                self.undo_manager.record_move(original_path, new_path)
            self._update_undo_state()

        if cancelled:
            QMessageBox.information(self, "Audio Smart Move", "⚠️ Operation cancelled by user.")

        QMessageBox.information(
            self,
            "Audio Smart Move Complete",
            f"✅ Audio files analyzed: {total_analyzed}\n"
            f"📦 Files moved: {moved_count}\n"
            f"⭐ Files kept: {total_kept}\n"
            f"📈 Avg quality improvement: {avg_improvement:.1f}"
        )

        if warnings:
            self._show_error_dialog("Audio Analysis Warnings", warnings)
        if failed_files:
            self._show_error_dialog("Some Moves Failed", failed_files)

        if kept_log:
            self._show_log_dialog("Audio Smart Move - Kept Files", kept_log)

        self.refresh_current_view()

    def _show_error_dialog(self, title, errors):
        """Show a scrollable error dialog with details."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(700, 400)
        layout = QVBoxLayout(dialog)

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setText("\n".join(errors))
        layout.addWidget(text_area)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def _show_log_dialog(self, title, lines):
        """Show a scrollable dialog with log details."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(700, 400)
        layout = QVBoxLayout(dialog)

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setText("\n".join(lines))
        layout.addWidget(text_area)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def move_selected_files(self):
        """Move selected files to destination folder"""
        if not self.move_destination:
            QMessageBox.warning(self, "Warning", "Please select a move destination first")
            return

        selected_files = self.get_selected_files()

        if not selected_files:
            QMessageBox.warning(self, "Warning", "No files selected for moving")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Move",
            f"Move {len(selected_files)} selected files to:\n{self.move_destination}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # MV-1: Delegate to the background-threaded infrastructure instead of
            # looping with shutil.move() on the main thread (caused "Not Responding").
            # _start_bulk_move_duplicates handles progress dialog, undo recording,
            # and calls refresh_current_view() via _handle_bulk_move_results().
            os.makedirs(self.move_destination, exist_ok=True)
            self._start_bulk_move_duplicates(selected_files, preview_dialog=None)
    
    def refresh_current_view(self):
        """Refresh the current view after file operations"""
        # Clear selections
        self.clear_selection()
        self._prune_result_sets()
        
        # Re-populate trees (files may have been moved/deleted)
        self.populate_duplicate_groups()
        self.populate_similar_groups()
        
        # Clear file trees
        self.duplicate_files_tree.clear()
        self.similar_files_tree.clear()
        self._update_bulk_operation_buttons()
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    def format_sample_rate(self, sample_rate_hz):
        """Format sample rate in user-friendly format."""
        try:
            rate = float(sample_rate_hz)
            if rate <= 0:
                return "Unknown"
            return f"{rate / 1000:.1f} kHz"
        except Exception:
            return "Unknown"
    
    def format_timestamp(self, timestamp):
        """Format timestamp in readable format"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "Unknown"
    
    def show_export_menu(self):
        """Show export menu with format options"""
        menu = QMenu(self)
        
        json_action = QAction("📄 Export to JSON", self)
        json_action.triggered.connect(self.export_to_json)
        menu.addAction(json_action)
        
        csv_action = QAction("📊 Export to CSV", self)
        csv_action.triggered.connect(self.export_to_csv)
        menu.addAction(csv_action)
        
        html_action = QAction("🌐 Export to HTML", self)
        html_action.triggered.connect(self.export_to_html)
        menu.addAction(html_action)
        
        menu.exec(self.export_btn.mapToGlobal(self.export_btn.rect().bottomLeft()))
    
    def delete_to_recycle_bin(self):
        """Delete selected files to recycle bin (safe delete)"""
        selected_files = self.get_selected_files()
        
        if not selected_files:
            QMessageBox.warning(self, "Warning", "No files selected for deletion")
            return
        
        reply = QMessageBox.question(
            self, 
            "Confirm Delete to Recycle Bin",
            f"Move {len(selected_files)} selected files to Recycle Bin?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            failed_files = []
            
            for file_path in selected_files:
                try:
                    if HAS_SEND2TRASH:
                        send2trash(file_path)
                    else:
                        os.remove(file_path)  # Fallback to permanent delete
                    deleted_count += 1
                except Exception as e:
                    failed_files.append(f"{file_path}: {e}")
            
            # Show results
            if deleted_count > 0:
                method = "Recycle Bin" if HAS_SEND2TRASH else "Permanently deleted"
                QMessageBox.information(
                    self, 
                    "Delete Complete", 
                    f"✅ Successfully moved {deleted_count} files to {method}."
                )
            
            if failed_files:
                QMessageBox.warning(
                    self,
                    "Some Deletions Failed",
                    f"❌ Failed to delete {len(failed_files)} files:\n\n" + 
                    "\n".join(failed_files[:5]) +
                    (f"\n... and {len(failed_files)-5} more" if len(failed_files) > 5 else "")
                )
            
            # Refresh the view
            self.refresh_current_view()
    
    def recycle_selected_files(self):
        """Alias for delete_to_recycle_bin"""
        self.delete_to_recycle_bin()
    
    def export_to_json(self):
        """Export results to JSON file"""
        if not self.duplicates and not self.similar_files:
            QMessageBox.warning(self, "Warning", "No results to export. Run a scan first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Report", 
            f"duplicate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                export_data = {
                    "scan_date": datetime.now().isoformat(),
                    "scan_folder": self.directory or "",
                    "duplicates": {},
                    "similar_files": {}
                }
                
                # Add duplicate groups
                for i, (hash_val, files) in enumerate(self.duplicates.items()):
                    group_data = {
                        "files": [],
                        "total_size": 0
                    }
                    for file_path_item in files:
                        try:
                            size = os.path.getsize(file_path_item)
                            group_data["files"].append({
                                "path": file_path_item,
                                "name": os.path.basename(file_path_item),
                                "size": size
                            })
                            group_data["total_size"] += size
                        except Exception:
                            group_data["files"].append({
                                "path": file_path_item,
                                "name": os.path.basename(file_path_item),
                                "size": 0
                            })
                    export_data["duplicates"][f"group_{i+1}"] = group_data
                
                # Add similar file groups
                for i, (key, group_data) in enumerate(self.similar_files.items()):
                    export_data["similar_files"][f"group_{i+1}"] = {
                        "reasons": group_data.get("reasons", []),
                        "files": [
                            {
                                "path": f,
                                "name": os.path.basename(f),
                                "size": os.path.getsize(f) if os.path.exists(f) else 0
                            } for f in group_data.get("files", [])
                        ]
                    }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Export Complete", f"✅ Report saved to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"❌ Failed to export:\n{e}")
    
    def export_to_csv(self):
        """Export results to CSV file"""
        if not self.duplicates and not self.similar_files:
            QMessageBox.warning(self, "Warning", "No results to export. Run a scan first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Report", 
            f"duplicate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['Type', 'Group', 'File Name', 'File Path', 'Size (bytes)', 'Reason'])
                    
                    # Write duplicates
                    for i, (hash_val, files) in enumerate(self.duplicates.items()):
                        for file_path_item in files:
                            try:
                                size = os.path.getsize(file_path_item)
                            except Exception:
                                size = 0
                            writer.writerow([
                                'Duplicate',
                                f'Group {i+1}',
                                os.path.basename(file_path_item),
                                file_path_item,
                                size,
                                'Exact hash match'
                            ])
                    
                    # Write similar files
                    for i, (key, group_data) in enumerate(self.similar_files.items()):
                        reasons = ", ".join(group_data.get("reasons", []))
                        for file_path_item in group_data.get("files", []):
                            try:
                                size = os.path.getsize(file_path_item)
                            except Exception:
                                size = 0
                            writer.writerow([
                                'Similar',
                                f'Group {i+1}',
                                os.path.basename(file_path_item),
                                file_path_item,
                                size,
                                reasons
                            ])
                
                QMessageBox.information(self, "Export Complete", f"✅ Report saved to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"❌ Failed to export:\n{e}")
    
    def export_to_html(self):
        """Export results to HTML report"""
        if not self.duplicates and not self.similar_files:
            QMessageBox.warning(self, "Warning", "No results to export. Run a scan first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML Report", 
            f"duplicate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html)"
        )
        
        if file_path:
            try:
                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Duplicate File Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #4CAF50; margin-top: 30px; }}
        h3 {{ color: #666; margin-top: 20px; }}
        .summary {{ background: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .group {{ background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .file {{ padding: 8px; margin: 5px 0; background: #f9f9f9; border-left: 3px solid #4CAF50; }}
        .file-path {{ color: #666; font-size: 0.9em; }}
        .file-size {{ color: #888; font-size: 0.85em; }}
        .similar {{ border-left-color: #2196F3; }}
        .reasons {{ color: #2196F3; font-style: italic; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>🔍 Duplicate File Report</h1>
    <div class="summary">
        <p><strong>📅 Scan Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>📁 Scanned Folder:</strong> {self.directory or "N/A"}</p>
        <p><strong>📊 Duplicate Groups:</strong> {len(self.duplicates)}</p>
        <p><strong>🔗 Similar File Groups:</strong> {len(self.similar_files)}</p>
    </div>
"""
                
                # Add duplicates section
                if self.duplicates:
                    html_content += "    <h2>📋 Duplicate Files</h2>\n"
                    for i, (hash_val, files) in enumerate(self.duplicates.items()):
                        total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
                        html_content += f'    <div class="group">\n'
                        html_content += f'        <h3>Group {i+1} ({len(files)} files, {self.format_file_size(total_size)} total)</h3>\n'
                        for file_path_item in files:
                            try:
                                size = self.format_file_size(os.path.getsize(file_path_item))
                            except Exception:
                                size = "Unknown"
                            html_content += f'        <div class="file">\n'
                            html_content += f'            <strong>{os.path.basename(file_path_item)}</strong>\n'
                            html_content += f'            <div class="file-path">{file_path_item}</div>\n'
                            html_content += f'            <div class="file-size">Size: {size}</div>\n'
                            html_content += f'        </div>\n'
                        html_content += '    </div>\n'
                
                # Add similar files section
                if self.similar_files:
                    html_content += "    <h2>🔗 Similar Files</h2>\n"
                    for i, (key, group_data) in enumerate(self.similar_files.items()):
                        reasons = ", ".join(group_data.get("reasons", []))
                        files = group_data.get("files", [])
                        html_content += f'    <div class="group">\n'
                        html_content += f'        <h3>Group {i+1} ({len(files)} files)</h3>\n'
                        html_content += f'        <p class="reasons">Similarity: {reasons}</p>\n'
                        for file_path_item in files:
                            try:
                                size = self.format_file_size(os.path.getsize(file_path_item))
                            except Exception:
                                size = "Unknown"
                            html_content += f'        <div class="file similar">\n'
                            html_content += f'            <strong>{os.path.basename(file_path_item)}</strong>\n'
                            html_content += f'            <div class="file-path">{file_path_item}</div>\n'
                            html_content += f'            <div class="file-size">Size: {size}</div>\n'
                            html_content += f'        </div>\n'
                        html_content += '    </div>\n'
                
                html_content += """
</body>
</html>"""
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                QMessageBox.information(self, "Export Complete", f"✅ Report saved to:\n{file_path}")
                
                # Offer to open in browser
                reply = QMessageBox.question(
                    self, "Open Report",
                    "Would you like to open the report in your browser?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open(f'file:///{file_path}')
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"❌ Failed to export:\n{e}")

    # ==================== Advanced Config & Undo Methods ====================
    
    def show_enterprise_config(self):
        """Show enterprise configuration dialog"""
        dialog = EnterpriseConfigDialog(self, self.enterprise_config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.enterprise_config.update(dialog.get_config())
            
            # Save to settings
            import json
            self.settings.setValue("advanced_config", json.dumps(self.enterprise_config))
            self.settings.setValue("enterprise_config", json.dumps(self.enterprise_config))
            
            # Update undo manager
            self.undo_manager.max_operations = self.enterprise_config.get("max_undo_operations", 100)
            
            # Reinitialize advanced components
            self._init_enterprise_components()
            
            self.status_bar.showMessage("Advanced configuration updated", 3000)
    
    def show_merge_analysis(self):
        """Show merge analysis dialog for selected files"""
        if not HAS_ENTERPRISE_MODULES or not self.file_merger:
            QMessageBox.warning(self, "Feature Unavailable", 
                              "Merge analysis requires enterprise modules.\n\n"
                              "Install with: pip install -r requirements_enterprise.txt")
            return
        
        # Get selected files from current view
        current_tab = self.tab_widget.currentIndex()
        files_to_analyze = []
        
        if current_tab == 1:  # Results tab
            active_tree = self.duplicate_files_tree if self.results_tabs.currentIndex() == 0 else self.similar_files_tree
            for i in range(active_tree.topLevelItemCount()):
                item = active_tree.topLevelItem(i)
                if item.checkState(0) == Qt.CheckState.Checked:
                    file_path = item.data(0, Qt.ItemDataRole.UserRole)
                    if file_path:
                        files_to_analyze.append(file_path)
        
        if len(files_to_analyze) < 2:
            QMessageBox.warning(self, "Selection Required",
                              "Please select at least 2 files to analyze for merging.")
            return
        
        try:
            # Perform merge analysis
            analysis = self.file_merger.analyze_mergeable_files(files_to_analyze)
            
            # Build result message
            result = f"""📊 Merge Analysis Results

Files analyzed: {len(files_to_analyze)}
File type: {analysis.get('file_type', 'Unknown')}
Mergeable: {'✅ Yes' if analysis.get('mergeable', False) else '❌ No'}
Confidence: {analysis.get('confidence', 0):.1%}

"""
            if analysis.get('merge_suggestions'):
                result += "📋 Merge Suggestions:\n"
                for i, suggestion in enumerate(analysis['merge_suggestions'], 1):
                    result += f"\n{i}. {suggestion.get('description', 'No description')}\n"
                    result += f"   Strategy: {suggestion.get('merge_strategy', 'Unknown')}\n"
                    result += f"   Confidence: {suggestion.get('confidence', 0):.1%}\n"
            else:
                result += "ℹ️ No specific merge suggestions available.\n"
            
            QMessageBox.information(self, "Merge Analysis", result)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"❌ Merge analysis failed:\n{e}")
    
    def import_session(self):
        """Import a previous scan session from JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Session", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Restore duplicates
                self.duplicates = session_data.get('duplicates', {})
                
                # Restore similar files
                self.similar_files = session_data.get('similar_files', {})
                
                # Restore directory if available
                if 'directory' in session_data:
                    self.directory = session_data['directory']
                    self.dir_label.setText(f"📁 {self.directory}")
                    self.scan_btn.setEnabled(True)
                    self.scan_action.setEnabled(True)
                
                # Update UI
                self._prune_result_sets()
                self.populate_duplicate_groups()
                self.populate_similar_groups()
                
                # Update status
                total_duplicates = sum(len(files) for files in self.duplicates.values())
                self.results_status.setText(
                    f"📊 {len(self.duplicates)} groups, {total_duplicates} duplicates (imported)"
                )
                
                # Switch to results tab
                self.tab_widget.setCurrentIndex(1)
                
                QMessageBox.information(self, "Import Complete",
                    f"✅ Session imported successfully!\n\n"
                    f"Duplicate groups: {len(self.duplicates)}\n"
                    f"Similar file groups: {len(self.similar_files)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"❌ Failed to import session:\n{e}")
    
    def export_session(self):
        """Export current scan session to JSON"""
        if not self.duplicates and not self.similar_files:
            QMessageBox.warning(self, "Warning", "No results to export. Run a scan first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Session",
            f"scan_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                session_data = {
                    'version': '2.0',
                    'export_date': datetime.now().isoformat(),
                    'directory': getattr(self, 'directory', ''),
                    'duplicates': self.duplicates,
                    'similar_files': self.similar_files,
                    'statistics': {
                        'duplicate_groups': len(self.duplicates),
                        'total_duplicates': sum(len(files) for files in self.duplicates.values()),
                        'similar_groups': len(self.similar_files)
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2)
                
                QMessageBox.information(self, "Export Complete",
                    f"✅ Session exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"❌ Failed to export session:\n{e}")
    
    def undo_last_operation(self):
        """Undo the last file operation"""
        if not self.enterprise_config.get("enable_undo", True):
            QMessageBox.information(self, "Undo Disabled",
                                  "Undo is disabled. Enable it in Advanced Configuration.")
            return
        
        success, message = self.undo_manager.undo_last_operation()
        
        if success:
            QMessageBox.information(self, "Undo Successful", f"✅ {message}")
            # Refresh results
            self.refresh_results()
        else:
            QMessageBox.warning(self, "Undo Failed", f"⚠️ {message}")
        
        # Update undo action state
        self._update_undo_state()
    
    def _update_undo_state(self):
        """Update the undo action enabled state"""
        undo_count = self.undo_manager.get_undo_count()
        self.undo_action.setEnabled(undo_count > 0)
        if undo_count > 0:
            self.undo_action.setText(f"↩️ &Undo Last Operation ({undo_count})")
        else:
            self.undo_action.setText("↩️ &Undo Last Operation")
    
    def clear_undo_backups(self):
        """Clear all undo backups"""
        undo_count = self.undo_manager.get_undo_count()
        if undo_count == 0:
            QMessageBox.information(self, "No Backups", "There are no undo backups to clear.")
            return
        
        reply = QMessageBox.question(
            self, "Clear Backups",
            f"Are you sure you want to clear {undo_count} backup(s)?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.undo_manager.clear_backups()
            self._update_undo_state()
            QMessageBox.information(self, "Backups Cleared", "✅ All undo backups have been cleared.")
    
    def refresh_results(self):
        """Refresh the results display"""
        self._prune_result_sets()
        self.populate_duplicate_groups()
        self.populate_similar_groups()
        self.status_bar.showMessage("Results refreshed", 2000)
    
    def deselect_all_files(self):
        """Deselect all files in the current tree"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 1:  # Results tab
            # Get current sub-tab
            results_tab_index = self.results_tabs.currentIndex()
            
            if results_tab_index == 0:  # Duplicates tab
                for i in range(self.duplicate_files_tree.topLevelItemCount()):
                    item = self.duplicate_files_tree.topLevelItem(i)
                    item.setCheckState(0, Qt.CheckState.Unchecked)
            elif results_tab_index == 1:  # Similar files tab
                for i in range(self.similar_files_tree.topLevelItemCount()):
                    item = self.similar_files_tree.topLevelItem(i)
                    item.setCheckState(0, Qt.CheckState.Unchecked)
    
    def show_about(self):
        """Show about dialog"""
        about_text = f"""<h2>Duplicate File Finder</h2>
<p><b>Edition:</b> Unified App</p>
<p><b>Features:</b></p>
<ul>
    <li>GPU Acceleration: {'Available (' + CUDA_BACKEND + ')' if HAS_CUDA else 'CPU Only'}</li>
    <li>Advanced Modules: {'Available' if HAS_ENTERPRISE_MODULES else 'Not Installed'}</li>
    <li>Audio Fingerprinting: {'Available' if HAS_AUDIO_FINGERPRINTER else 'Not Available'}</li>
    <li>Video Fingerprinting: {'Available' if HAS_VIDEO_FINGERPRINTER else 'Not Available'}</li>
    <li>Semantic Similarity: {'Available' if HAS_SEMANTIC_SIMILARITY else 'Not Available'}</li>
    <li>Recycle Bin Support: {'Available' if HAS_SEND2TRASH else 'Not Available'}</li>
</ul>
<p><b>Capabilities:</b></p>
<ul>
    <li>Multiple hash algorithms (MD5, SHA256, Perceptual, Audio, Semantic)</li>
    <li>Parallel processing with {OPTIMAL_WORKERS} workers</li>
    <li>Smart sorting and keep suggestions</li>
    <li>Undo last file operation</li>
    <li>Session import/export</li>
    <li>Dark/Light theme support</li>
</ul>
"""
        QMessageBox.about(self, "About", about_text)

    def _prune_result_sets(self):
        """Remove result entries for files that no longer exist."""
        pruned_duplicates = {}
        for hash_val, files in self.duplicates.items():
            existing_files = [file_path for file_path in files if os.path.exists(file_path)]
            if len(existing_files) > 1:
                pruned_duplicates[hash_val] = existing_files
        self.duplicates = pruned_duplicates

        pruned_similar = {}
        for key, group_data in self.similar_files.items():
            existing_files = [file_path for file_path in group_data.get("files", []) if os.path.exists(file_path)]
            if len(existing_files) > 1:
                updated_group = dict(group_data)
                updated_group["files"] = existing_files
                pruned_similar[key] = updated_group
        self.similar_files = pruned_similar
    
    def closeEvent(self, event):
        """Handle application close - save state and cleanup"""
        # Save window state
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        if self.directory:
            self.settings.setValue("last_directory", self.directory)
        
        # Save enterprise config
        import json
        self.settings.setValue("advanced_config", json.dumps(self.enterprise_config))
        self.settings.setValue("enterprise_config", json.dumps(self.enterprise_config))
        self.settings.setValue("media_profiles", json.dumps(self.get_media_profiles_from_ui()))
        
        # Check for pending undo operations
        undo_count = self.undo_manager.get_undo_count()
        if undo_count > 0:
            reply = QMessageBox.question(
                self, "Pending Backups",
                f"You have {undo_count} file backup(s) that can be undone.\n\n"
                "Do you want to keep them for next session?\n\n"
                "• Yes - Keep backups (can restore later)\n"
                "• No - Clear backups and exit\n"
                "• Cancel - Don't exit",
                QMessageBox.StandardButton.Yes | 
                QMessageBox.StandardButton.No | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            elif reply == QMessageBox.StandardButton.No:
                self.undo_manager.clear_backups()
        
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EnhancedDuplicateFinder()
    window.show()
    sys.exit(app.exec())
