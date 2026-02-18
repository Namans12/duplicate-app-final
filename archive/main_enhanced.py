#!/usr/bin/env python3
"""
Enhanced Duplicate File Finder
Clean implementation with improved UX features
"""
import sys
import os
import hashlib
import time
from collections import defaultdict
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox, QProgressBar, QLabel, QHBoxLayout,
    QTextEdit, QTabWidget, QFrame, QLineEdit, QStatusBar,
    QMenuBar, QToolBar, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import QThread, pyqtSignal, QSettings, Qt
from PyQt6.QtGui import QFont, QPalette, QColor, QKeySequence, QAction, QShortcut


class FileFilters:
    """Simple file filter class"""
    def passes_filter(self, filepath):
        return True


class UndoManager:
    """Simple undo manager placeholder"""
    pass


class AdvancedSimilarity:
    """Advanced similarity detector placeholder"""
    pass


class ScanThread(QThread):
    """Thread for scanning files and finding duplicates"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    current_file = pyqtSignal(str)
    time_estimate = pyqtSignal(str)

    def __init__(self, directory, filters):
        super().__init__()
        self.directory = directory
        self.filters = filters
        self.start_time = None

    def run(self):
        """Main scanning logic"""
        self.start_time = datetime.now()
        self.status_update.emit("Collecting files...")
        
        # Collect all files
        files = []
        for root, _, filenames in os.walk(self.directory):
            for fn in filenames:
                fp = os.path.join(root, fn)
                if self.filters.passes_filter(fp):
                    files.append(fp)
        
        total = len(files)
        self.status_update.emit(f"Found {total} files. Computing hashes...")
        
        # Hash files and find duplicates
        hashes = defaultdict(list)
        for i, fp in enumerate(files):
            self.current_file.emit(os.path.basename(fp))
            
            # Calculate progress and ETA
            progress_percent = int((i / total) * 100) if total > 0 else 0
            self.progress.emit(progress_percent)
            
            if i > 0 and self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                rate = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                self.time_estimate.emit(f"ETA: {remaining:.0f}s")
            
            try:
                file_hash = self.compute_hash(fp)
                if file_hash:
                    hashes[file_hash].append(fp)
            except Exception as e:
                print(f"Error processing {fp}: {e}")
        
        # Filter to only duplicates
        duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
        self.current_file.emit("🎉 Scan completed!")
        self.status_update.emit(f"Found {len(duplicates)} duplicate groups")
        self.finished.emit(duplicates)

    def compute_hash(self, filepath):
        """Compute MD5 hash of file"""
        try:
            hasher = hashlib.md5()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None


class ModernDuplicateFinder(QMainWindow):
    """Enhanced duplicate finder with modern UI"""
    
    def __init__(self):
        super().__init__()
        self.duplicates = {}
        self.selected_files = set()
        self.filters = FileFilters()
        self.undo_manager = UndoManager()
        self.similarity_detector = AdvancedSimilarity()
        self.settings = QSettings("DuplicateFinder", "Enhanced")
        self.dark_mode = self.settings.value("dark_mode", False, type=bool)
        self.original_duplicates = {}
        self.init_ui()
        self.setup_shortcuts()
        self.apply_theme()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🔍 Smart Duplicate Finder Pro")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to scan for duplicates")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create header
        self.create_header(main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_scanner_tab()
        self.create_results_tab()
        
        # Create status bar
        self.create_status_bar(main_layout)
        
        self.directory = None

    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        scan_action = QAction('&Scan Directory', self)
        scan_action.setShortcut('Ctrl+O')
        scan_action.triggered.connect(self.browse_directory)
        file_menu.addAction(scan_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        theme_action = QAction('🌙 Dark Mode' if not self.dark_mode else '☀️ Light Mode', self)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        self.theme_action = theme_action
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = self.addToolBar('Quick Actions')
        toolbar.setMovable(False)
        
        # Theme toggle button
        theme_action = QAction('🌙' if not self.dark_mode else '☀️', self)
        theme_action.setToolTip('Toggle Dark/Light Mode (Ctrl+T)')
        theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(theme_action)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Delete shortcut
        self.delete_shortcut = QShortcut(QKeySequence("Del"), self)
        self.delete_shortcut.activated.connect(self.delete_selected_shortcut)
        
        # Preview shortcut
        self.preview_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        
        # Theme toggle shortcut
        self.theme_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        self.theme_shortcut.activated.connect(self.toggle_theme)

    def delete_selected_shortcut(self):
        """Handle delete key press"""
        pass  # Placeholder

    def toggle_theme(self):
        """Toggle between dark and light themes"""
        self.dark_mode = not self.dark_mode
        self.settings.setValue('dark_mode', self.dark_mode)
        self.apply_theme()
        
        # Update menu text
        if hasattr(self, 'theme_action'):
            self.theme_action.setText('🌙 Dark Mode' if not self.dark_mode else '☀️ Light Mode')

    def apply_theme(self):
        """Apply the current theme"""
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        
        if self.dark_mode:
            # Dark theme
            pal = QPalette()
            pal.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
            pal.setColor(QPalette.ColorRole.WindowText, QColor(240, 240, 240))
            pal.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            pal.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
            pal.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
            pal.setColor(QPalette.ColorRole.Text, QColor(240, 240, 240))
            pal.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            pal.setColor(QPalette.ColorRole.ButtonText, QColor(240, 240, 240))
            pal.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            pal.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
            pal.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
            self.setPalette(pal)
            self.setStyleSheet("""
                QLabel, QLineEdit, QTextEdit { 
                    color: #f0f0f0; 
                    background-color: #2a2a2a;
                }
                QTabWidget::pane {
                    background-color: #2a2a2a;
                }
                QTreeWidget {
                    background-color: #2a2a2a;
                    color: #f0f0f0;
                }
            """)
        else:
            # Light theme
            self.setPalette(QApplication.style().standardPalette())
            self.setStyleSheet("""
                QLabel, QLineEdit, QTextEdit { 
                    color: #111111; 
                    background-color: #ffffff;
                }
                QTabWidget::pane {
                    background-color: #ffffff;
                }
                QTreeWidget {
                    background-color: #ffffff;
                    color: #111111;
                }
            """)

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, 'About', 
                         'Smart Duplicate Finder Pro\n'
                         'Enhanced UI Version\n\n'
                         'Features:\n'
                         '• Dark/Light theme toggle (Ctrl+T)\n'
                         '• Search within results\n'
                         '• Progress with ETA\n'
                         '• Keyboard shortcuts')

    def create_header(self, layout):
        """Create the header section"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.Box)
        header_layout = QVBoxLayout(header_frame)
        
        # Title
        title_label = QLabel("🔍 Smart Duplicate Finder Pro")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px;
                padding: 10px;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("🔍 Search within results:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by filename or path...")
        self.search_box.textChanged.connect(self.filter_results)
        
        # Clear search button
        clear_search_btn = QPushButton("✖ Clear")
        clear_search_btn.clicked.connect(self.clear_search)
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box, 1)
        search_layout.addWidget(clear_search_btn)
        
        header_layout.addLayout(search_layout)
        
        # Stats
        self.stats_label = QLabel("Ready to scan...")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.stats_label)
        
        layout.addWidget(header_frame)

    def filter_results(self):
        """Filter results based on search text"""
        # Placeholder for filtering logic
        pass

    def clear_search(self):
        """Clear search box"""
        self.search_box.clear()

    def create_scanner_tab(self):
        """Create the scanner tab"""
        scanner_widget = QWidget()
        layout = QVBoxLayout(scanner_widget)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        browse_btn = QPushButton("📁 Browse Directory (Ctrl+O)")
        browse_btn.clicked.connect(self.browse_directory)
        
        dir_layout.addWidget(QLabel("Directory:"))
        dir_layout.addWidget(self.dir_label, 1)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)
        
        # Scan button
        self.scan_btn = QPushButton("🔍 Start Scan")
        self.scan_btn.clicked.connect(self.start_scan)
        self.scan_btn.setEnabled(False)
        layout.addWidget(self.scan_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status
        self.scan_status = QLabel("Ready to scan")
        layout.addWidget(self.scan_status)
        
        # Current file
        self.current_file_label = QLabel("")
        layout.addWidget(self.current_file_label)
        
        # ETA
        self.eta_label = QLabel("")
        layout.addWidget(self.eta_label)
        
        layout.addStretch()
        self.tab_widget.addTab(scanner_widget, "🔍 Scanner")

    def create_results_tab(self):
        """Create the results tab"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Results tree
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["Duplicate Groups", "Size", "Count"])
        layout.addWidget(self.results_tree)
        
        # Action buttons
        button_layout = QHBoxLayout()
        delete_btn = QPushButton("🗑️ Delete Selected")
        move_btn = QPushButton("📁 Move Selected")
        
        button_layout.addWidget(delete_btn)
        button_layout.addWidget(move_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        self.tab_widget.addTab(results_widget, "📋 Results")

    def create_status_bar(self, layout):
        """Create status bar section"""
        pass  # Using QMainWindow's built-in status bar

    def browse_directory(self):
        """Browse for directory to scan"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Scan")
        if directory:
            self.directory = directory
            self.dir_label.setText(directory)
            self.scan_btn.setEnabled(True)
            self.status_bar.showMessage(f"Selected: {directory}")

    def start_scan(self):
        """Start scanning for duplicates"""
        if not self.directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first.")
            return
        
        self.scan_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.scan_status.setText("Scanning...")
        
        # Start scan thread
        self.scan_thread = ScanThread(self.directory, self.filters)
        self.scan_thread.progress.connect(self.progress_bar.setValue)
        self.scan_thread.status_update.connect(self.scan_status.setText)
        self.scan_thread.current_file.connect(self.current_file_label.setText)
        self.scan_thread.time_estimate.connect(self.eta_label.setText)
        self.scan_thread.finished.connect(self.scan_finished)
        self.scan_thread.start()

    def scan_finished(self, duplicates):
        """Handle scan completion"""
        self.duplicates = duplicates
        self.populate_results()
        self.scan_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.scan_status.setText("Scan completed!")
        self.current_file_label.setText("")
        self.eta_label.setText("")
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(1)
        
        # Update stats
        total_files = sum(len(files) for files in duplicates.values())
        total_groups = len(duplicates)
        self.stats_label.setText(f"Found {total_files} duplicate files in {total_groups} groups")
        self.status_bar.showMessage(f"Scan complete: {total_files} duplicates in {total_groups} groups")

    def populate_results(self):
        """Populate the results tree"""
        self.results_tree.clear()
        
        for i, (hash_val, files) in enumerate(self.duplicates.items()):
            # Create group item
            group_item = QTreeWidgetItem()
            group_item.setText(0, f"Group {i+1}")
            group_item.setText(1, f"{self.get_file_size(files[0])}")
            group_item.setText(2, str(len(files)))
            
            # Add file items
            for file_path in files:
                file_item = QTreeWidgetItem()
                file_item.setText(0, os.path.basename(file_path))
                file_item.setText(1, file_path)
                file_item.setText(2, "")
                group_item.addChild(file_item)
            
            self.results_tree.addTopLevelItem(group_item)
        
        self.results_tree.expandAll()

    def get_file_size(self, filepath):
        """Get formatted file size"""
        try:
            size = os.path.getsize(filepath)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        except:
            return "Unknown"


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ModernDuplicateFinder()
    w.show()
    sys.exit(app.exec())
