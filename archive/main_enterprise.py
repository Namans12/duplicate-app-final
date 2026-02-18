#!/usr/bin/env python3
"""
Enterprise Duplicate File Finder
Complete application with all advanced features integrated
"""
import sys
import os
import hashlib
import time
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass
import json

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not available. Please install: pip install PyQt6")

# Import our enterprise modules
try:
    from database_manager import DatabaseManager
    from gpu_accelerator import GPUAccelerator
    from plugin_system import PluginManager
    from memory_optimizer import MemoryManager, StreamingHasher
    from duplicate_merger import FileMerger
    HAS_ENTERPRISE_MODULES = True
except ImportError as e:
    print(f"Enterprise modules not available: {e}")
    HAS_ENTERPRISE_MODULES = False

@dataclass
class ScanProgress:
    current_file: str = ""
    files_processed: int = 0
    total_files: int = 0
    current_phase: str = "Initializing"
    start_time: float = 0.0
    duplicates_found: int = 0
    bytes_saved: int = 0

class EnterpriseWorker(QThread):
    """Enhanced worker thread with enterprise features"""
    progress_updated = pyqtSignal(ScanProgress)
    duplicates_found = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, folders: List[str], config: Dict[str, Any]):
        super().__init__()
        self.folders = folders
        self.config = config
        self.should_stop = False
        
        # Initialize enterprise components
        self.db_manager = None
        self.gpu_accelerator = None
        self.plugin_manager = None
        self.memory_manager = None
        self.file_merger = None
        
        if HAS_ENTERPRISE_MODULES:
            self._init_enterprise_components()
            
    def _init_enterprise_components(self):
        """Initialize enterprise backend components"""
        try:
            # Database manager for persistence
            if self.config.get("use_database", True):
                self.db_manager = DatabaseManager()
                
            # GPU acceleration for image processing
            if self.config.get("use_gpu", False):
                self.gpu_accelerator = GPUAccelerator()
                
            # Plugin system for extensibility
            if self.config.get("use_plugins", True):
                self.plugin_manager = PluginManager()
                self.plugin_manager.load_plugins()
                
            # Memory optimization
            if self.config.get("optimize_memory", True):
                self.memory_manager = MemoryManager()
                
            # File merging system
            if self.config.get("enable_merging", True):
                self.file_merger = FileMerger()
                
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize enterprise components: {e}")
    
    def run(self):
        """Enhanced scan with enterprise features"""
        progress = ScanProgress()
        progress.start_time = time.time()
        progress.current_phase = "Scanning files"
        
        try:
            # Collect all files
            all_files = []
            for folder in self.folders:
                if self.should_stop:
                    return
                    
                progress.current_phase = f"Collecting files from {folder}"
                self.progress_updated.emit(progress)
                
                for root, dirs, files in os.walk(folder):
                    if self.should_stop:
                        return
                        
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            all_files.append(file_path)
                            
            progress.total_files = len(all_files)
            progress.current_phase = "Processing files"
            self.progress_updated.emit(progress)
            
            # Check for incremental scan capability
            if self.db_manager and self.config.get("incremental_scan", True):
                files_to_process = self.db_manager.get_new_or_modified_files(all_files)
                progress.current_phase = f"Incremental scan: {len(files_to_process)} new/modified files"
            else:
                files_to_process = all_files
                progress.current_phase = "Full scan"
                
            self.progress_updated.emit(progress)
            
            # Process files with enterprise features
            duplicates = self._process_files_enterprise(files_to_process, progress)
            
            # Analyze for potential merges
            if self.file_merger and self.config.get("analyze_merges", True):
                progress.current_phase = "Analyzing merge opportunities"
                self.progress_updated.emit(progress)
                
                for group in duplicates.values():
                    if len(group) > 1:
                        merge_analysis = self.file_merger.analyze_mergeable_files(group)
                        if merge_analysis.get("mergeable", False):
                            # Add merge suggestions to the duplicate info
                            for file_path in group:
                                if file_path not in duplicates:
                                    duplicates[file_path] = []
                                duplicates[file_path].append({
                                    "type": "merge_suggestion",
                                    "analysis": merge_analysis
                                })
            
            # Calculate statistics
            progress.current_phase = "Finalizing results"
            total_duplicates = sum(len(group) - 1 for group in duplicates.values() if len(group) > 1)
            total_bytes_saved = 0
            
            for group in duplicates.values():
                if len(group) > 1:
                    try:
                        file_size = os.path.getsize(group[0])
                        total_bytes_saved += file_size * (len(group) - 1)
                    except OSError:
                        pass
                        
            progress.duplicates_found = total_duplicates
            progress.bytes_saved = total_bytes_saved
            progress.current_phase = "Complete"
            self.progress_updated.emit(progress)
            
            # Store results in database
            if self.db_manager:
                self.db_manager.store_scan_results(duplicates, progress.__dict__)
            
            self.duplicates_found.emit(duplicates)
            
        except Exception as e:
            self.error_occurred.emit(f"Scan error: {str(e)}")
    
    def _process_files_enterprise(self, files: List[str], progress: ScanProgress) -> Dict[str, List[str]]:
        """Process files using enterprise features"""
        file_hashes = defaultdict(list)
        
        # Use streaming hasher for large files
        if self.memory_manager:
            hasher = StreamingHasher()
        else:
            hasher = None
            
        # Process files in batches for GPU acceleration
        if self.gpu_accelerator and self.config.get("use_gpu_hashing", False):
            # Separate image files for GPU processing
            image_files = []
            other_files = []
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            
            for file_path in files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in image_extensions:
                    image_files.append(file_path)
                else:
                    other_files.append(file_path)
                    
            # Process images with GPU acceleration
            if image_files:
                progress.current_phase = "GPU processing images"
                self.progress_updated.emit(progress)
                
                try:
                    gpu_hashes = self.gpu_accelerator.batch_perceptual_hash(image_files)
                    for file_path, hash_value in gpu_hashes.items():
                        file_hashes[hash_value].append(file_path)
                        progress.files_processed += 1
                        progress.current_file = os.path.basename(file_path)
                        self.progress_updated.emit(progress)
                except Exception as e:
                    print(f"GPU processing failed, falling back to CPU: {e}")
                    other_files.extend(image_files)
                    
            files_to_hash = other_files
        else:
            files_to_hash = files
            
        # Process remaining files with CPU (and plugins)
        progress.current_phase = "Hashing files"
        self.progress_updated.emit(progress)
        
        # Use thread pool for parallel processing
        max_workers = min(os.cpu_count() or 4, 8)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all hashing tasks
            future_to_file = {}
            
            for file_path in files_to_hash:
                if self.should_stop:
                    break
                    
                future = executor.submit(self._hash_file_enterprise, file_path, hasher)
                future_to_file[future] = file_path
                
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_file):
                if self.should_stop:
                    break
                    
                file_path = future_to_file[future]
                try:
                    file_hash = future.result()
                    if file_hash:
                        file_hashes[file_hash].append(file_path)
                        
                    progress.files_processed += 1
                    progress.current_file = os.path.basename(file_path)
                    
                    # Update ETA
                    elapsed = time.time() - progress.start_time
                    if progress.files_processed > 0:
                        rate = progress.files_processed / elapsed
                        remaining = progress.total_files - progress.files_processed
                        eta = remaining / rate if rate > 0 else 0
                        progress.current_phase = f"Processing files (ETA: {eta:.0f}s)"
                        
                    self.progress_updated.emit(progress)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        # Filter out non-duplicates and return groups
        duplicates = {hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1}
        return duplicates
    
    def _hash_file_enterprise(self, file_path: str, streaming_hasher) -> Optional[str]:
        """Hash file using enterprise features"""
        try:
            # Check if we have a custom plugin for this file type
            if self.plugin_manager:
                custom_hash = self.plugin_manager.get_custom_hash(file_path)
                if custom_hash:
                    return custom_hash
                    
            # Use streaming hasher for large files
            if streaming_hasher and os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                return streaming_hasher.hash_file(file_path)
            else:
                # Standard hashing for smaller files
                return self._hash_file_standard(file_path)
                
        except Exception as e:
            print(f"Error hashing {file_path}: {e}")
            return None
    
    def _hash_file_standard(self, file_path: str) -> Optional[str]:
        """Standard file hashing"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def stop(self):
        """Stop the scanning process"""
        self.should_stop = True

class EnterpriseConfigDialog(QDialog):
    """Configuration dialog for enterprise features"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enterprise Configuration")
        self.setModal(True)
        self.resize(500, 600)
        
        self.config = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tab widget for different configuration sections
        tabs = QTabWidget()
        
        # Database settings
        db_tab = QWidget()
        db_layout = QVBoxLayout(db_tab)
        
        self.use_database_cb = QCheckBox("Enable database persistence")
        self.use_database_cb.setChecked(True)
        db_layout.addWidget(self.use_database_cb)
        
        self.incremental_scan_cb = QCheckBox("Enable incremental scanning")
        self.incremental_scan_cb.setChecked(True)
        db_layout.addWidget(self.incremental_scan_cb)
        
        db_layout.addWidget(QLabel("Database path:"))
        self.db_path_edit = QLineEdit("duplicates.db")
        db_layout.addWidget(self.db_path_edit)
        
        tabs.addTab(db_tab, "Database")
        
        # GPU settings
        gpu_tab = QWidget()
        gpu_layout = QVBoxLayout(gpu_tab)
        
        self.use_gpu_cb = QCheckBox("Enable GPU acceleration")
        self.use_gpu_cb.setChecked(False)
        gpu_layout.addWidget(self.use_gpu_cb)
        
        self.use_gpu_hashing_cb = QCheckBox("Use GPU for image hashing")
        self.use_gpu_hashing_cb.setChecked(False)
        gpu_layout.addWidget(self.use_gpu_hashing_cb)
        
        gpu_layout.addWidget(QLabel("Batch size for GPU processing:"))
        self.gpu_batch_size = QSpinBox()
        self.gpu_batch_size.setRange(1, 1000)
        self.gpu_batch_size.setValue(32)
        gpu_layout.addWidget(self.gpu_batch_size)
        
        tabs.addTab(gpu_tab, "GPU")
        
        # Plugin settings
        plugin_tab = QWidget()
        plugin_layout = QVBoxLayout(plugin_tab)
        
        self.use_plugins_cb = QCheckBox("Enable plugin system")
        self.use_plugins_cb.setChecked(True)
        plugin_layout.addWidget(self.use_plugins_cb)
        
        plugin_layout.addWidget(QLabel("Plugin directory:"))
        self.plugin_dir_edit = QLineEdit("plugins")
        plugin_layout.addWidget(self.plugin_dir_edit)
        
        tabs.addTab(plugin_tab, "Plugins")
        
        # Memory optimization
        memory_tab = QWidget()
        memory_layout = QVBoxLayout(memory_tab)
        
        self.optimize_memory_cb = QCheckBox("Enable memory optimization")
        self.optimize_memory_cb.setChecked(True)
        memory_layout.addWidget(self.optimize_memory_cb)
        
        memory_layout.addWidget(QLabel("Large file threshold (MB):"))
        self.large_file_threshold = QSpinBox()
        self.large_file_threshold.setRange(1, 10000)
        self.large_file_threshold.setValue(100)
        memory_layout.addWidget(self.large_file_threshold)
        
        tabs.addTab(memory_tab, "Memory")
        
        # Merge settings
        merge_tab = QWidget()
        merge_layout = QVBoxLayout(merge_tab)
        
        self.enable_merging_cb = QCheckBox("Enable file merging analysis")
        self.enable_merging_cb.setChecked(True)
        merge_layout.addWidget(self.enable_merging_cb)
        
        self.analyze_merges_cb = QCheckBox("Auto-analyze merge opportunities")
        self.analyze_merges_cb.setChecked(True)
        merge_layout.addWidget(self.analyze_merges_cb)
        
        tabs.addTab(merge_tab, "Merging")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
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
            "analyze_merges": self.analyze_merges_cb.isChecked()
        }

class EnterpriseDuplicateFinderApp(QMainWindow):
    """Enhanced duplicate finder with enterprise features"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enterprise Duplicate File Finder")
        self.setGeometry(100, 100, 1200, 800)
        
        # Configuration
        self.config = {
            "use_database": True,
            "incremental_scan": True,
            "use_gpu": False,
            "use_plugins": True,
            "optimize_memory": True,
            "enable_merging": True,
            "analyze_merges": True
        }
        
        # State
        self.worker = None
        self.duplicates_data = {}
        self.current_theme = "light"
        
        self.init_ui()
        self.apply_theme()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Toolbar with enterprise features
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # Main content area with enhanced features
        main_content = self.create_main_content()
        layout.addWidget(main_content)
        
        # Enhanced status bar
        self.create_status_bar()
        
        # Create menu bar
        self.create_menu_bar()
    
    def create_toolbar(self):
        """Create enhanced toolbar with enterprise features"""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)

        # Folder selection
        folder_label = QLabel("Select folders to scan (Ctrl+O):")
        toolbar_layout.addWidget(folder_label)
        self.folder_label = folder_label

        # Use a larger editor to occupy the space instead of a Browse button
        folder_list = QLineEdit()
        folder_list.setPlaceholderText("Paste folder paths separated by ';' or use Ctrl+O to browse")
        folder_list.setMinimumWidth(400)
        toolbar_layout.addWidget(folder_list, 1)
        self.folder_list = folder_list

        # Enterprise configuration
        self.config_button = QPushButton("Enterprise Config")
        self.config_button.clicked.connect(self.show_enterprise_config)
        toolbar_layout.addWidget(self.config_button)

        # Scan controls
        self.scan_button = QPushButton("Start Scan")
        self.scan_button.clicked.connect(self.start_scan)
        toolbar_layout.addWidget(self.scan_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)
        toolbar_layout.addWidget(self.stop_button)

        return toolbar_widget
    
    def create_main_content(self):
        """Create main content area with enhanced features"""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Enhanced results
        left_panel = self.create_results_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Details and merge options
        right_panel = self.create_details_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([600, 400])
        return splitter
    
    def create_results_panel(self):
        """Create enhanced results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Search and filter controls
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search duplicates...")
        self.search_input.textChanged.connect(self.filter_results)
        search_layout.addWidget(self.search_input)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Images", "Documents", "Videos", "Audio", "Archives"])
        self.filter_combo.currentTextChanged.connect(self.filter_results)
        search_layout.addWidget(self.filter_combo)
        
        layout.addLayout(search_layout)
        
        # Results tree with enhanced information
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels([
            "File Groups", "Count", "Size", "Type", "Merge Status"
        ])
        self.results_tree.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.results_tree)
        
        # Progress bar with detailed information
        self.progress_frame = QFrame()
        self.progress_frame.setFrameStyle(QFrame.Shape.Box)
        progress_layout = QVBoxLayout(self.progress_frame)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_details = QLabel("Ready to scan...")
        progress_layout.addWidget(self.progress_details)
        
        self.eta_label = QLabel("")
        progress_layout.addWidget(self.eta_label)
        
        layout.addWidget(self.progress_frame)
        
        return panel
    
    def create_details_panel(self):
        """Create details panel with merge capabilities"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File details
        details_group = QGroupBox("File Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # Merge analysis
        merge_group = QGroupBox("Merge Analysis")
        merge_layout = QVBoxLayout(merge_group)
        
        self.merge_text = QTextEdit()
        self.merge_text.setReadOnly(True)
        merge_layout.addWidget(self.merge_text)
        
        merge_buttons = QHBoxLayout()
        
        self.analyze_merge_button = QPushButton("Analyze Mergeability")
        self.analyze_merge_button.clicked.connect(self.analyze_merge)
        merge_buttons.addWidget(self.analyze_merge_button)
        
        self.execute_merge_button = QPushButton("Execute Merge")
        self.execute_merge_button.clicked.connect(self.execute_merge)
        self.execute_merge_button.setEnabled(False)
        merge_buttons.addWidget(self.execute_merge_button)
        
        merge_layout.addLayout(merge_buttons)
        
        layout.addWidget(merge_group)
        
        return panel
    
    def create_menu_bar(self):
        """Create enhanced menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        import_action = QAction("Import Session", self)
        import_action.triggered.connect(self.import_session)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        theme_action = QAction("Toggle Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        config_action = QAction("Enterprise Configuration", self)
        config_action.triggered.connect(self.show_enterprise_config)
        tools_menu.addAction(config_action)
        
        plugin_action = QAction("Plugin Manager", self)
        plugin_action.triggered.connect(self.show_plugin_manager)
        tools_menu.addAction(plugin_action)
    
    def create_status_bar(self):
        """Create enhanced status bar"""
        status = self.statusBar()
        
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label)
        
        # Statistics labels
        self.stats_label = QLabel("Files: 0 | Duplicates: 0 | Space saved: 0 B")
        status.addPermanentWidget(self.stats_label)
        
        # Theme indicator
        self.theme_label = QLabel("Theme: Light")
        status.addPermanentWidget(self.theme_label)
    
    def show_enterprise_config(self):
        """Show enterprise configuration dialog"""
        dialog = EnterpriseConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.config.update(dialog.get_config())
            self.status_label.setText("Enterprise configuration updated")
    
    def show_plugin_manager(self):
        """Show plugin manager (placeholder)"""
        QMessageBox.information(self, "Plugin Manager", 
                              "Plugin Manager coming soon!\n\n"
                              "This will allow you to:\n"
                              "- View loaded plugins\n"
                              "- Enable/disable plugins\n"
                              "- Install new plugins\n"
                              "- Configure plugin settings")
    
    def browse_folders(self):
        """Browse for folders to scan"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Scan")
        if folder:
            current_folders = self.folder_list.text()
            if current_folders:
                self.folder_list.setText(f"{current_folders};{folder}")
            else:
                self.folder_list.setText(folder)
    
    def start_scan(self):
        """Start enterprise scan"""
        folders_text = self.folder_list.text().strip()
        if not folders_text:
            QMessageBox.warning(self, "Warning", "Please select at least one folder to scan.")
            return
            
        folders = [folder.strip() for folder in folders_text.split(';') if folder.strip()]
        
        self.scan_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Clear previous results
        self.results_tree.clear()
        self.details_text.clear()
        self.merge_text.clear()
        
        # Start worker with enterprise features
        self.worker = EnterpriseWorker(folders, self.config)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.duplicates_found.connect(self.display_results)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.scan_finished)
        self.worker.start()
    
    def stop_scan(self):
        """Stop the scan"""
        if self.worker:
            self.worker.stop()
            
    def scan_finished(self):
        """Handle scan completion"""
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Scan completed")
    
    def update_progress(self, progress: ScanProgress):
        """Update progress with enhanced information"""
        if progress.total_files > 0:
            percentage = int((progress.files_processed / progress.total_files) * 100)
            self.progress_bar.setValue(percentage)
        else:
            self.progress_bar.setValue(0)
            
        # Update detailed progress information
        self.progress_details.setText(
            f"Phase: {progress.current_phase}\n"
            f"File: {progress.current_file}\n"
            f"Progress: {progress.files_processed}/{progress.total_files}"
        )
        
        # Update ETA
        if progress.start_time > 0:
            elapsed = time.time() - progress.start_time
            if progress.files_processed > 0:
                rate = progress.files_processed / elapsed
                remaining = progress.total_files - progress.files_processed
                eta = remaining / rate if rate > 0 else 0
                self.eta_label.setText(f"ETA: {eta:.0f} seconds")
                
        # Update statistics
        self.stats_label.setText(
            f"Files: {progress.files_processed} | "
            f"Duplicates: {progress.duplicates_found} | "
            f"Space saved: {self.format_size(progress.bytes_saved)}"
        )
    
    def display_results(self, duplicates: Dict[str, List[str]]):
        """Display enhanced results with merge information"""
        self.duplicates_data = duplicates
        self.populate_results_tree()
        
    def populate_results_tree(self):
        """Populate results tree with enhanced information"""
        self.results_tree.clear()
        
        for hash_value, files in self.duplicates_data.items():
            if len(files) < 2:
                continue
                
            # Create group item
            group_item = QTreeWidgetItem()
            
            # Determine file type
            file_type = self.get_file_type(files[0])
            
            # Check for merge suggestions
            merge_status = "Not analyzed"
            if HAS_ENTERPRISE_MODULES:
                # Check if any file has merge suggestions
                for file_path in files:
                    if isinstance(file_path, dict) and file_path.get("type") == "merge_suggestion":
                        merge_status = "Mergeable"
                        break
                        
            # Calculate total size
            total_size = 0
            try:
                file_size = os.path.getsize(files[0])
                total_size = file_size * len(files)
            except OSError:
                pass
                
            group_item.setText(0, f"Group {hash_value[:8]}...")
            group_item.setText(1, str(len(files)))
            group_item.setText(2, self.format_size(total_size))
            group_item.setText(3, file_type)
            group_item.setText(4, merge_status)
            
            # Add individual files
            for file_path in files:
                if isinstance(file_path, str):  # Skip merge suggestion objects
                    file_item = QTreeWidgetItem()
                    file_item.setText(0, os.path.basename(file_path))
                    file_item.setText(1, "")
                    try:
                        file_size = os.path.getsize(file_path)
                        file_item.setText(2, self.format_size(file_size))
                    except OSError:
                        file_item.setText(2, "Unknown")
                    file_item.setText(3, file_type)
                    file_item.setData(0, Qt.ItemDataRole.UserRole, file_path)
                    group_item.addChild(file_item)
                    
            self.results_tree.addTopLevelItem(group_item)
            
        self.results_tree.expandAll()
    
    def on_selection_changed(self):
        """Handle selection changes with enhanced details"""
        selected_items = self.results_tree.selectedItems()
        if not selected_items:
            return
            
        item = selected_items[0]
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        
        if file_path:
            # Display file details
            self.show_file_details(file_path)
        else:
            # Show group details
            self.show_group_details(item)
    
    def show_file_details(self, file_path: str):
        """Show detailed file information"""
        try:
            stat = os.stat(file_path)
            details = f"""
File: {file_path}
Size: {self.format_size(stat.st_size)}
Modified: {time.ctime(stat.st_mtime)}
Created: {time.ctime(stat.st_ctime)}
Type: {self.get_file_type(file_path)}
            """
            self.details_text.setText(details.strip())
        except Exception as e:
            self.details_text.setText(f"Error getting file details: {e}")
    
    def show_group_details(self, group_item):
        """Show group details"""
        file_count = group_item.childCount()
        files = []
        
        for i in range(file_count):
            child = group_item.child(i)
            file_path = child.data(0, Qt.ItemDataRole.UserRole)
            if file_path:
                files.append(file_path)
                
        details = f"""
Duplicate Group Details:
Files in group: {len(files)}
First file: {files[0] if files else 'N/A'}

Files:
"""
        for i, file_path in enumerate(files, 1):
            details += f"{i}. {file_path}\n"
            
        self.details_text.setText(details.strip())
    
    def analyze_merge(self):
        """Analyze merge possibilities for selected files"""
        if not HAS_ENTERPRISE_MODULES:
            QMessageBox.information(self, "Feature Unavailable", 
                                  "Merge analysis requires enterprise modules.")
            return
            
        selected_items = self.results_tree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a file group to analyze.")
            return
            
        # Get files from selected group
        item = selected_items[0]
        files = []
        
        if item.parent():  # It's a file item
            parent_item = item.parent()
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                file_path = child.data(0, Qt.ItemDataRole.UserRole)
                if file_path:
                    files.append(file_path)
        else:  # It's a group item
            for i in range(item.childCount()):
                child = item.child(i)
                file_path = child.data(0, Qt.ItemDataRole.UserRole)
                if file_path:
                    files.append(file_path)
                    
        if len(files) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 files to analyze merging.")
            return
            
        # Perform merge analysis
        try:
            from duplicate_merger import file_merger
            analysis = file_merger.analyze_mergeable_files(files)
            
            # Display results
            result_text = f"""
Merge Analysis Results:
Files analyzed: {len(files)}
File type: {analysis.get('file_type', 'Unknown')}
Mergeable: {analysis.get('mergeable', False)}
Confidence: {analysis.get('confidence', 0):.2%}

"""
            
            if analysis.get('merge_suggestions'):
                result_text += "Merge Suggestions:\n"
                for suggestion in analysis['merge_suggestions']:
                    result_text += f"- {suggestion.get('description', 'No description')}\n"
                    result_text += f"  Strategy: {suggestion.get('merge_strategy', 'Unknown')}\n"
                    result_text += f"  Confidence: {suggestion.get('confidence', 0):.2%}\n\n"
                    
                self.execute_merge_button.setEnabled(True)
            else:
                result_text += "No merge suggestions available."
                self.execute_merge_button.setEnabled(False)
                
            self.merge_text.setText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Merge analysis failed: {e}")
    
    def execute_merge(self):
        """Execute merge operation"""
        QMessageBox.information(self, "Merge Execution", 
                              "Merge execution will be implemented in the next version.\n\n"
                              "This feature will:\n"
                              "- Create backups of original files\n"
                              "- Execute the suggested merge strategy\n"
                              "- Show a preview before applying changes\n"
                              "- Allow rollback if needed")
    
    def filter_results(self):
        """Filter results based on search and filter criteria"""
        search_text = self.search_input.text().lower()
        file_type_filter = self.filter_combo.currentText()
        
        for i in range(self.results_tree.topLevelItemCount()):
            item = self.results_tree.topLevelItem(i)
            should_show = True
            
            if file_type_filter != "All":
                item_type = item.text(3)
                if file_type_filter.lower() not in item_type.lower():
                    should_show = False
                    
            if search_text and should_show:
                # Search in group and file names
                group_text = item.text(0).lower()
                if search_text not in group_text:
                    # Check child items
                    found_in_children = False
                    for j in range(item.childCount()):
                        child = item.child(j)
                        child_text = child.text(0).lower()
                        if search_text in child_text:
                            found_in_children = True
                            break
                    should_show = found_in_children
                    
            item.setHidden(not should_show)
    
    def get_file_type(self, file_path: str) -> str:
        """Get file type category"""
        ext = os.path.splitext(file_path)[1].lower()
        
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        audio_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg'}
        doc_exts = {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'}
        archive_exts = {'.zip', '.rar', '.7z', '.tar', '.gz'}
        
        if ext in image_exts:
            return "Image"
        elif ext in video_exts:
            return "Video"
        elif ext in audio_exts:
            return "Audio"
        elif ext in doc_exts:
            return "Document"
        elif ext in archive_exts:
            return "Archive"
        else:
            return "Other"
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
            
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme()
        self.theme_label.setText(f"Theme: {self.current_theme.title()}")
    
    def apply_theme(self):
        """Apply the current theme"""
        if self.current_theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTreeWidget {
                    background-color: #3b3b3b;
                    alternate-background-color: #404040;
                }
                QLineEdit {
                    background-color: #404040;
                    border: 1px solid #555555;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #0078d4;
                    border: none;
                    padding: 8px 16px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
        else:
            self.setStyleSheet("")  # Default light theme
    
    def export_results(self):
        """Export scan results"""
        if not self.duplicates_data:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "duplicate_results.json", "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.duplicates_data, f, indent=2)
                QMessageBox.information(self, "Success", "Results exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def import_session(self):
        """Import previous session"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Session", "", "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.duplicates_data = json.load(f)
                self.populate_results_tree()
                QMessageBox.information(self, "Success", "Session imported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Import failed: {e}")
    
    def handle_error(self, error_message: str):
        """Handle errors from worker thread"""
        QMessageBox.critical(self, "Error", error_message)
        self.scan_finished()

def main():
    """Main application entry point"""
    if not PYQT_AVAILABLE:
        print("PyQt6 is required to run this application.")
        print("Please install it with: pip install PyQt6")
        return
        
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Enterprise Duplicate File Finder")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Enterprise Solutions")
    
    # Create and show main window
    window = EnterpriseDuplicateFinderApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
