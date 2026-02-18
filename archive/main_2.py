#!/usr/bin/env python3
"""
Enhanced Duplicate File Finder with Modern UI
"""
import sys
import os
import hashlib
import shutil
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QListWidget, QVBoxLayout, QWidget,
    QFileDialog, QMessageBox, QProgressBar, QLabel, QHBoxLayout, QComboBox,
    QTextEdit, QSplitter, QCheckBox, QGroupBox, QGridLayout, QSpinBox,
    QDateEdit, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QScrollArea, QSlider, QListWidgetItem, QAbstractItemView,
    QLineEdit, QToolBar, QMenu, QStatusBar
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QDate, QTimer, QTime, QSettings
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor, QAction, QKeySequence, QShortcut

# Optional imports with fallbacks
try:
    from PIL import Image
    import imagehash
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from pydub import AudioSegment
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    import chromaprint  # type: ignore
    HAS_CHROMAPRINT = True
except ImportError:
    HAS_CHROMAPRINT = False

try:
    import librosa  # type: ignore
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

class AdvancedSimilarity:
    """Advanced similarity detection for audio, video, and text"""
    
    def __init__(self):
        self.text_model = None
        self.similarity_threshold = 0.8
        self._load_text_model()
        
    def _load_text_model(self):
        """Load sentence transformer model for text similarity"""
        if HAS_TRANSFORMERS:
            try:
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')  # type: ignore
            except Exception as e:
                print(f"Failed to load text model: {e}")
                self.text_model = None
                
    def set_similarity_threshold(self, threshold):
        """Set the similarity threshold for text comparison"""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        
    def get_audio_fingerprint(self, file_path):
        """Get audio fingerprint using chromaprint or librosa"""
        try:
            if HAS_CHROMAPRINT and HAS_AUDIO:
                # Use chromaprint for robust audio fingerprinting
                audio = AudioSegment.from_file(file_path)  # type: ignore
                raw_audio = audio.raw_data
                sample_rate = audio.frame_rate
                
                # Convert to numpy array for chromaprint
                audio_array = np.frombuffer(raw_audio, dtype=np.int16)  # type: ignore
                if audio.channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)  # type: ignore
                    
                fingerprint = chromaprint.fingerprint(audio_array, sample_rate)[1]  # type: ignore
                return f"chromaprint:{fingerprint}"
                
            elif HAS_LIBROSA:
                # Fallback to librosa for audio features
                y, sr = librosa.load(file_path, duration=30)  # type: ignore
                # Extract spectral features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # type: ignore
                chroma = librosa.feature.chroma(y=y, sr=sr)  # type: ignore
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # type: ignore
                
                # Combine features and hash
                features = np.concatenate([  # type: ignore
                    mfcc.mean(axis=1),
                    chroma.mean(axis=1),
                    spectral_contrast.mean(axis=1)
                ])
                
                feature_hash = hashlib.md5(features.tobytes()).hexdigest()
                return f"librosa:{feature_hash}"
                
            elif HAS_AUDIO:
                # Basic pydub fingerprinting as last resort
                audio = AudioSegment.from_file(file_path)  # type: ignore
                samples = np.array(audio.get_array_of_samples())  # type: ignore
                
                # Simple spectral features
                fft = np.fft.fft(samples[:min(len(samples), 44100)])  # type: ignore
                magnitude = np.abs(fft)[:1000]  # type: ignore
                
                feature_hash = hashlib.md5(magnitude.tobytes()).hexdigest()
                return f"pydub:{feature_hash}"
                
        except Exception as e:
            print(f"Audio fingerprinting failed for {file_path}: {e}")
            
        return None
        
    def get_video_scene_hash(self, file_path, num_scenes=5):
        """Extract multiple scene hashes from video for better duplicate detection"""
        if not HAS_CV2:
            return None
            
        try:
            cap = cv2.VideoCapture(file_path)  # type: ignore
            if not cap.isOpened():
                return None
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore
            if total_frames == 0:
                cap.release()
                return None
                
            scene_hashes = []
            
            # Extract frames from different parts of the video
            for i in range(num_scenes):
                frame_pos = int(total_frames * i / (num_scenes - 1)) if num_scenes > 1 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)  # type: ignore
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Convert to RGB and resize for consistent processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
                    frame_resized = cv2.resize(frame_rgb, (64, 64))  # type: ignore
                    
                    if HAS_PILLOW:
                        # Use perceptual hashing for better scene comparison
                        img = Image.fromarray(frame_resized)  # type: ignore
                        scene_hash = str(imagehash.phash(img, hash_size=8))  # type: ignore
                        scene_hashes.append(scene_hash)
                    else:
                        # Fallback to simple hash
                        frame_bytes = frame_resized.tobytes()
                        scene_hash = hashlib.md5(frame_bytes).hexdigest()[:16]
                        scene_hashes.append(scene_hash)
                        
            cap.release()
            
            # Combine scene hashes
            if scene_hashes:
                combined_hash = "|".join(scene_hashes)
                return f"scenes:{combined_hash}"
                
        except Exception as e:
            print(f"Video scene extraction failed for {file_path}: {e}")
            
        return None
        
    def get_text_similarity_hash(self, file_path):
        """Get semantic embedding for text similarity"""
        if not self.text_model:
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read(5000)  # Limit text length
                
            if len(text.strip()) < 50:  # Skip very short texts
                return None
                
            # Get embedding
            embedding = self.text_model.encode([text])
            
            # Convert to hash for grouping (we'll do similarity comparison later)
            embedding_bytes = embedding.tobytes()
            embedding_hash = hashlib.md5(embedding_bytes).hexdigest()
            
            # Store both hash and embedding data
            return f"semantic:{embedding_hash}:{len(text)}"
            
        except Exception as e:
            print(f"Text similarity failed for {file_path}: {e}")
            
        return None
        
    def calculate_text_similarity(self, text1_path, text2_path):
        """Calculate cosine similarity between two text files"""
        if not self.text_model:
            return 0.0
            
        try:
            texts = []
            for path in [text1_path, text2_path]:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read(5000))
                    
            if any(len(text.strip()) < 50 for text in texts):
                return 0.0
                
            embeddings = self.text_model.encode(texts)
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]  # type: ignore
            
            return float(similarity)
            
        except Exception as e:
            print(f"Text similarity calculation failed: {e}")
            return 0.0
            
    def is_similar_enough(self, similarity_score):
        """Check if similarity score meets threshold"""
        return similarity_score >= self.similarity_threshold

class FileFilters:
    def __init__(self):
        self.min_size = 0  # bytes
        self.max_size = float('inf')  # bytes
        self.file_types = []  # list of extensions
        self.date_from = None
        self.date_to = None
        
    def passes_filter(self, file_path):
        try:
            stat = os.stat(file_path)
            
            # Size filter
            if stat.st_size < self.min_size or stat.st_size > self.max_size:
                return False
                
            # File type filter
            if self.file_types:
                ext = os.path.splitext(file_path)[1].lower()
                if ext not in self.file_types:
                    return False
                    
            # Date filter
            if self.date_from or self.date_to:
                file_date = datetime.fromtimestamp(stat.st_mtime).date()
                if self.date_from and file_date < self.date_from:
                    return False
                if self.date_to and file_date > self.date_to:
                    return False
                    
            return True
        except:
            return False

class UndoManager:
    def __init__(self):
        self.backup_dir = os.path.join(os.getcwd(), ".duplicate_finder_backup")
        self.operations = []
        self.max_operations = 100
        self._ensure_backup_dir()
        
    def _ensure_backup_dir(self):
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
                    os.remove(old_op['backup_path'])
                    
            return True
        except Exception as e:
            print(f"Backup failed: {e}")
            return False
            
    def undo_last_operation(self):
        """Restore the last deleted file"""
        if not self.operations:
            return False, "No operations to undo"
            
        try:
            operation = self.operations.pop()
            
            if operation['type'] == 'delete':
                # Restore file from backup
                shutil.copy2(operation['backup_path'], operation['original_path'])
                os.remove(operation['backup_path'])
                return True, f"Restored: {os.path.basename(operation['original_path'])}"
                
        except Exception as e:
            return False, f"Undo failed: {e}"
            
        return False, "Unknown operation type"
        
    def get_recent_operations(self, limit=10):
        """Get recent operations for display"""
        return self.operations[-limit:] if self.operations else []

class ScanThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    current_file = pyqtSignal(str)  # New signal for current file
    time_estimate = pyqtSignal(str)  # New signal for time estimate
    
    def __init__(self, directory, hash_type, filters, similarity_detector=None):
        super().__init__()
        self.directory = directory
        self.hash_type = hash_type
        self.filters = filters
        self.similarity_detector = similarity_detector or AdvancedSimilarity()
        self.start_time = None
        
    def run(self):
        self.start_time = datetime.now()
        file_hashes = defaultdict(list)
        files = []
        
        self.status_update.emit("🔍 Collecting files...")
        
        # Collect all files
        for root, dirs, filenames in os.walk(self.directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if self.filters.passes_filter(file_path):
                    files.append(file_path)
        
        total_files = len(files)
        self.status_update.emit(f"🔍 Processing {total_files} files...")
        
        for i, file_path in enumerate(files):
            try:
                # Emit current file being processed
                filename = os.path.basename(file_path)
                self.current_file.emit(f"📁 {filename}")
                
                # Calculate time estimate
                if i > 0 and self.start_time:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    avg_time_per_file = elapsed / i
                    remaining_files = total_files - i
                    estimated_remaining = remaining_files * avg_time_per_file
                    
                    if estimated_remaining > 60:
                        minutes = int(estimated_remaining // 60)
                        seconds = int(estimated_remaining % 60)
                        self.time_estimate.emit(f"⏱️ {minutes}m {seconds}s remaining")
                    else:
                        self.time_estimate.emit(f"⏱️ {int(estimated_remaining)}s remaining")
                
                file_hash = self.compute_hash(file_path)
                if file_hash:
                    file_hashes[file_hash].append(file_path)
                
                progress = int((i + 1) / total_files * 100)
                self.progress.emit(progress)
                self.status_update.emit(f"✅ Processed {i+1}/{total_files} files")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Filter for duplicates only
        duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}
        self.current_file.emit("🎉 Scan completed!")
        self.time_estimate.emit("⏱️ Finished")
        self.finished.emit(duplicates)
    
    def compute_hash(self, file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if self.hash_type == "MD5":
                return self.md5_hash(file_path)
            elif self.hash_type == "SHA256":
                return self.sha256_hash(file_path)
            elif self.hash_type == "Perceptual" and HAS_PILLOW:
                return self.perceptual_hash(file_path)
            elif self.hash_type == "Audio Fingerprint":
                if ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']:
                    return self.similarity_detector.get_audio_fingerprint(file_path)
                else:
                    return self.md5_hash(file_path)  # fallback for non-audio
            elif self.hash_type == "Video Scenes":
                if ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
                    return self.similarity_detector.get_video_scene_hash(file_path)
                else:
                    return self.md5_hash(file_path)  # fallback for non-video
            elif self.hash_type == "Text Similarity":
                if ext in ['.txt', '.py', '.js', '.html', '.css', '.md', '.doc', '.docx']:
                    return self.similarity_detector.get_text_similarity_hash(file_path)
                else:
                    return self.md5_hash(file_path)  # fallback for non-text
            else:
                return self.md5_hash(file_path)  # fallback
        except:
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
                img = Image.open(file_path)  # type: ignore
                return str(imagehash.phash(img))  # type: ignore
            except:
                return self.md5_hash(file_path)
        else:
            return self.md5_hash(file_path)

class ModernDuplicateFinder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.duplicates = {}
        self.selected_files = set()
        self.filters = FileFilters()
        self.undo_manager = UndoManager()
        self.similarity_detector = AdvancedSimilarity()
        
        # Settings and themes
        self.settings = QSettings("DuplicateFinder", "Enhanced")
        self.dark_mode = self.settings.value("dark_mode", False, type=bool)
        self.original_duplicates = {}  # Store original for search filtering
        
        self.init_ui()
        self.setup_shortcuts()
        self.apply_theme()
        
    def init_ui(self):
        self.setWindowTitle("🔍 Smart Duplicate Finder Pro")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)
        
        # Create menu bar and toolbar
        self.create_menu_bar()
        self.create_toolbar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to scan for duplicates")
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header section with search
        self.create_header(main_layout)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Scanner tab
        self.create_scanner_tab()
        
        # Results tab
        self.create_results_tab()
        
        # Settings tab
        self.create_settings_tab()
        
        # Status bar
        self.create_status_bar(main_layout)
        
        self.directory = None
        
    def create_menu_bar(self):
        """Create menu bar with theme toggle"""
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
        """Create toolbar with quick actions"""
        toolbar = self.addToolBar('Quick Actions')
        toolbar.setMovable(False)
        
        # Browse action
        browse_action = QAction('📁 Browse', self)
        browse_action.setShortcut('Ctrl+O')
        browse_action.triggered.connect(self.browse_directory)
        toolbar.addAction(browse_action)
        
        toolbar.addSeparator()
        
        # Theme toggle
        theme_action = QAction('🌙' if not self.dark_mode else '☀️', self)
        theme_action.setToolTip('Toggle Dark/Light Mode')
        theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(theme_action)
        self.toolbar_theme_action = theme_action
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Delete shortcut
        self.delete_shortcut = QShortcut(QKeySequence("Del"), self)
        self.delete_shortcut.activated.connect(self.delete_selected_shortcut)
        
        # Preview shortcut
        self.preview_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        self.preview_shortcut.activated.connect(self.preview_selected_shortcut)
        
        # Select all shortcut
        self.select_all_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        self.select_all_shortcut.activated.connect(self.select_all_duplicates)
        
        # Undo shortcut
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_action)
        
        # Export shortcut
        self.export_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        self.export_shortcut.activated.connect(self.export_results)
        
    def toggle_theme(self):
        """Toggle between dark and light themes"""
        self.dark_mode = not self.dark_mode
        self.settings.setValue("dark_mode", self.dark_mode)
        self.apply_theme()
        
        # Update menu and toolbar
        self.theme_action.setText('🌙 Dark Mode' if not self.dark_mode else '☀️ Light Mode')
        self.toolbar_theme_action.setText('🌙' if not self.dark_mode else '☀️')
        
    def apply_theme(self):
        """Apply current theme to the application"""
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()
            
    def apply_dark_theme(self):
        """Apply dark theme styling"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        
        self.setPalette(dark_palette)
        
    def apply_light_theme(self):
        """Apply light theme styling"""
        self.setPalette(QApplication.style().standardPalette())
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Smart Duplicate Finder Pro", 
                         "🔍 Smart Duplicate Finder Pro v2.0\n\n"
                         "Advanced AI-powered duplicate file detection with:\n"
                         "• Audio fingerprinting\n"
                         "• Video scene analysis\n" 
                         "• Text similarity scoring\n"
                         "• Perceptual image hashing\n\n"
                         "Built with PyQt6 and modern AI technologies")

    def create_header(self, layout):
        """Create modern header with title and search"""
        header_frame = QFrame()
        header_frame.setFixedHeight(120)  # Increased height for search bar
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 10px;
                margin: 5px;
            }
        """)
        
        header_layout = QVBoxLayout(header_frame)
        
        # Top row with title and stats
        top_row = QHBoxLayout()
        
        # Title
        title_label = QLabel("🔍 Smart Duplicate Finder Pro")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                background: transparent;
                padding: 10px;
            }
        """)
        top_row.addWidget(title_label)
        
        top_row.addStretch()
        
        # Quick stats
        self.stats_label = QLabel("Ready to scan")
        self.stats_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                background: transparent;
                padding: 10px;
            }
        """)
        top_row.addWidget(self.stats_label)
        
        header_layout.addLayout(top_row)
        
        # Search bar row
        search_row = QHBoxLayout()
        search_row.setContentsMargins(20, 0, 20, 10)
        
        search_label = QLabel("🔍 Search duplicates:")
        search_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                background: transparent;
                font-weight: bold;
            }
        """)
        search_row.addWidget(search_label)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by filename or path...")
        self.search_box.textChanged.connect(self.filter_duplicates)
        self.search_box.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.9);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 20px;
                padding: 8px 15px;
                font-size: 14px;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid white;
                background: white;
            }
        """)
        search_row.addWidget(self.search_box, 1)
        
        clear_search_btn = QPushButton("✖")
        clear_search_btn.setFixedSize(40, 40)
        clear_search_btn.clicked.connect(self.clear_search)
        clear_search_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 20px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.3);
            }
        """)
        search_row.addWidget(clear_search_btn)
        
        header_layout.addLayout(search_row)
        
        layout.addWidget(header_frame)
        self.stats_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                background: transparent;
                padding: 10px;
            }
        """)
        header_layout.addWidget(self.stats_label)
        
        layout.addWidget(header_frame)
        
    def create_scanner_tab(self):
        """Create the main scanning interface"""
        scanner_widget = QWidget()
        layout = QVBoxLayout(scanner_widget)
        layout.setSpacing(15)
        
        # Directory selection
        dir_group = QGroupBox("📁 Directory Selection")
        dir_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        dir_layout = QHBoxLayout(dir_group)
        
        self.select_btn = QPushButton("📂 Select Directory")
        self.select_btn.clicked.connect(self.select_directory)
        self.select_btn.setFixedHeight(40)
        dir_layout.addWidget(self.select_btn)
        
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("QLabel { padding: 10px; background: #f5f5f5; border-radius: 5px; }")
        dir_layout.addWidget(self.dir_label, 2)
        
        layout.addWidget(dir_group)
        
        # Filters section
        self.create_filters_section(layout)
        
        # Scan options
        scan_group = QGroupBox("⚙️ Advanced Scan Options")
        scan_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        scan_layout = QGridLayout(scan_group)
        
        scan_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.hash_combo = QComboBox()
        self.hash_combo.addItems([
            "MD5", "SHA256", "Perceptual", 
            "Audio Fingerprint", "Video Scenes", "Text Similarity"
        ])
        self.hash_combo.setFixedHeight(35)
        self.hash_combo.currentTextChanged.connect(self.on_algorithm_changed)
        scan_layout.addWidget(self.hash_combo, 0, 1)
        
        # Similarity threshold slider (for text similarity)
        scan_layout.addWidget(QLabel("Text Similarity Threshold:"), 1, 0)
        self.similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_slider.setRange(50, 95)  # 0.5 to 0.95
        self.similarity_slider.setValue(80)  # 0.8 default
        self.similarity_slider.setEnabled(False)  # Only enabled for text similarity
        self.similarity_slider.valueChanged.connect(self.on_similarity_changed)
        scan_layout.addWidget(self.similarity_slider, 1, 1)
        
        self.similarity_label = QLabel("0.80")
        self.similarity_label.setFixedWidth(40)
        scan_layout.addWidget(self.similarity_label, 1, 2)
        
        # Algorithm info
        self.algorithm_info = QLabel("MD5: Fast hashing for exact duplicates")
        self.algorithm_info.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 5px; }")
        scan_layout.addWidget(self.algorithm_info, 2, 0, 1, 3)
        
        self.scan_btn = QPushButton("🚀 Start Advanced Scan")
        self.scan_btn.clicked.connect(self.scan_duplicates)
        self.scan_btn.setEnabled(False)
        self.scan_btn.setFixedHeight(40)
        self.scan_btn.setFixedWidth(200)
        scan_layout.addWidget(self.scan_btn, 3, 0, 1, 3)
        
        layout.addWidget(scan_group)
        
        # Progress section
        progress_group = QGroupBox("📊 Progress")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(25)
        progress_layout.addWidget(self.progress_bar)
        
        # Progress details row
        progress_details_layout = QHBoxLayout()
        
        self.status_label = QLabel("Select a directory to start")
        self.status_label.setStyleSheet("QLabel { padding: 5px; }")
        progress_details_layout.addWidget(self.status_label)
        
        progress_details_layout.addStretch()
        
        self.progress_label = QLabel("⏱️ Ready")
        self.progress_label.setStyleSheet("QLabel { padding: 5px; color: #666; }")
        progress_details_layout.addWidget(self.progress_label)
        
        progress_layout.addLayout(progress_details_layout)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        self.tab_widget.addTab(scanner_widget, "🔍 Scanner")
        
    def create_filters_section(self, layout):
        """Create file filtering options"""
        filters_group = QGroupBox("🔧 File Filters")
        filters_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        filters_layout = QGridLayout(filters_group)
        
        # File size filters
        filters_layout.addWidget(QLabel("Min Size (MB):"), 0, 0)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 10000)
        self.min_size_spin.setSuffix(" MB")
        filters_layout.addWidget(self.min_size_spin, 0, 1)
        
        filters_layout.addWidget(QLabel("Max Size (MB):"), 0, 2)
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(0, 10000)
        self.max_size_spin.setValue(1000)
        self.max_size_spin.setSuffix(" MB")
        filters_layout.addWidget(self.max_size_spin, 0, 3)
        
        # File type filters
        filters_layout.addWidget(QLabel("File Types:"), 1, 0)
        self.file_types_combo = QComboBox()
        self.file_types_combo.addItems([
            "All Files", "Images", "Videos", "Audio", "Documents", "Archives"
        ])
        filters_layout.addWidget(self.file_types_combo, 1, 1)
        
        # Date filters
        filters_layout.addWidget(QLabel("Modified After:"), 1, 2)
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-365))
        self.date_from.setCalendarPopup(True)
        filters_layout.addWidget(self.date_from, 1, 3)
        
        layout.addWidget(filters_group)
        
    def create_results_tab(self):
        """Create results viewing interface"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("✅ Select All")
        self.select_all_btn.clicked.connect(self.select_all_files)
        self.select_all_btn.setEnabled(False)
        toolbar_layout.addWidget(self.select_all_btn)
        
        self.clear_selection_btn = QPushButton("❌ Clear Selection")
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        self.clear_selection_btn.setEnabled(False)
        toolbar_layout.addWidget(self.clear_selection_btn)
        
        toolbar_layout.addStretch()
        
        self.delete_selected_btn = QPushButton("🗑️ Delete Selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_files)
        self.delete_selected_btn.setEnabled(False)
        self.delete_selected_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; }")
        toolbar_layout.addWidget(self.delete_selected_btn)
        
        self.undo_btn = QPushButton("↶ Undo Last")
        self.undo_btn.clicked.connect(self.undo_last_operation)
        self.undo_btn.setEnabled(False)
        toolbar_layout.addWidget(self.undo_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Groups list
        groups_widget = QWidget()
        groups_layout = QVBoxLayout(groups_widget)
        groups_layout.addWidget(QLabel("📁 Duplicate Groups"))
        
        self.groups_list = QListWidget()
        self.groups_list.currentItemChanged.connect(self.show_group_files)
        groups_layout.addWidget(self.groups_list)
        
        content_splitter.addWidget(groups_widget)
        
        # Files table
        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.addWidget(QLabel("📄 Files in Group"))
        
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(5)
        self.files_table.setHorizontalHeaderLabels(["Select", "File Name", "Path", "Size", "Similarity"])
        header = self.files_table.horizontalHeader()
        if header:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)  # type: ignore
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        files_layout.addWidget(self.files_table)
        
        content_splitter.addWidget(files_widget)
        
        # Preview area
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.addWidget(QLabel("👁️ File Preview"))
        
        self.preview_area = QTextEdit()
        self.preview_area.setMaximumWidth(350)
        preview_layout.addWidget(self.preview_area)
        
        content_splitter.addWidget(preview_widget)
        
        content_splitter.setSizes([300, 500, 350])
        layout.addWidget(content_splitter)
        
        self.tab_widget.addTab(results_widget, "📊 Results")
        
    def create_settings_tab(self):
        """Create settings and configuration tab"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Undo history
        undo_group = QGroupBox("↶ Undo History")
        undo_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        undo_layout = QVBoxLayout(undo_group)
        
        self.undo_list = QListWidget()
        self.undo_list.setMaximumHeight(200)
        undo_layout.addWidget(self.undo_list)
        
        self.refresh_undo_history()
        
        layout.addWidget(undo_group)
        
        # Statistics
        stats_group = QGroupBox("📈 Statistics")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        self.tab_widget.addTab(settings_widget, "⚙️ Settings")
        
    def create_status_bar(self, layout):
        """Create status bar at bottom"""
        status_frame = QFrame()
        status_frame.setFixedHeight(40)
        status_frame.setStyleSheet("""
            QFrame {
                background: #f8f9fa;
                border-top: 1px solid #dee2e6;
                border-radius: 5px;
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        
        self.selection_label = QLabel("No files selected")
        status_layout.addWidget(self.selection_label)
        
        status_layout.addStretch()
        
        self.total_size_label = QLabel("Total size: 0 MB")
        status_layout.addWidget(self.total_size_label)
        
        layout.addWidget(status_frame)
        
    def apply_modern_style(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                border-radius: 10px;
                background: white;
            }
            
            QTabBar::tab {
                background: #f0f0f0;
                border: 1px solid #c0c0c0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            
            QTabBar::tab:selected {
                background: white;
                border-bottom-color: white;
            }
            
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background-color: #45a049;
            }
            
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            
            QComboBox, QSpinBox, QDateEdit {
                border: 2px solid #ddd;
                border-radius: 6px;
                padding: 5px;
                background: white;
            }
            
            QComboBox:focus, QSpinBox:focus, QDateEdit:focus {
                border-color: #4CAF50;
            }
            
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 6px;
                text-align: center;
                background: #f0f0f0;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 4px;
            }
            
            QListWidget, QTableWidget, QTextEdit {
                border: 1px solid #ddd;
                border-radius: 6px;
                background: white;
                alternate-background-color: #f9f9f9;
            }
            
            QListWidget::item:selected, QTableWidget::item:selected {
                background: #e3f2fd;
                color: #1976d2;
            }
        """)
        
    def update_filters(self):
        """Update filters based on UI inputs"""
        self.filters.min_size = self.min_size_spin.value() * 1024 * 1024  # Convert MB to bytes
        max_size = self.max_size_spin.value() * 1024 * 1024
        self.filters.max_size = max_size if max_size > 0 else float('inf')
        
        # File type mapping
        file_type_map = {
            "Images": ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
            "Videos": ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'],
            "Audio": ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
            "Documents": ['.pdf', '.doc', '.docx', '.txt', '.rtf'],
            "Archives": ['.zip', '.rar', '.7z', '.tar', '.gz']
        }
        
        selected_type = self.file_types_combo.currentText()
        if selected_type == "All Files":
            self.filters.file_types = []
        else:
            self.filters.file_types = file_type_map.get(selected_type, [])
            
        # Date filters
        try:
            date_value = self.date_from.date()
            if hasattr(date_value, 'toPython'):
                self.filters.date_from = date_value.toPython()  # type: ignore
            else:
                self.filters.date_from = date_value.toPyDate()  # type: ignore
        except:
            self.filters.date_from = None
        
    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory = directory
            self.dir_label.setText(f"📁 {directory}")
            self.scan_btn.setEnabled(True)
            
    def scan_duplicates(self):
        if not self.directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first")
            return
            
        self.update_filters()
        
        self.scan_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting scan...")
        
        hash_type = self.hash_combo.currentText()
        
        self.scan_thread = ScanThread(self.directory, hash_type, self.filters, self.similarity_detector)
        self.scan_thread.progress.connect(self.progress_bar.setValue)
        self.scan_thread.status_update.connect(self.status_label.setText)
        self.scan_thread.finished.connect(self.scan_completed)
        
        # Connect enhanced progress signals
        self.scan_thread.current_file.connect(self.update_current_file)
        self.scan_thread.time_estimate.connect(self.update_time_estimate)
        
        self.scan_thread.start()
        
    def on_algorithm_changed(self, algorithm):
        """Handle algorithm selection change"""
        # Enable similarity slider only for text similarity
        is_text_similarity = algorithm == "Text Similarity"
        self.similarity_slider.setEnabled(is_text_similarity)
        
        # Update algorithm info
        info_map = {
            "MD5": "MD5: Fast hashing for exact duplicates",
            "SHA256": "SHA256: Secure hashing for exact duplicates", 
            "Perceptual": "Perceptual: AI-powered image similarity detection",
            "Audio Fingerprint": "Audio: Advanced audio fingerprinting with Chromaprint/Librosa",
            "Video Scenes": "Video: Multi-scene analysis for better video duplicate detection",
            "Text Similarity": "Text: Semantic similarity using AI embeddings"
        }
        
        self.algorithm_info.setText(info_map.get(algorithm, "Unknown algorithm"))
        
        # Update threshold if needed
        if is_text_similarity:
            threshold = self.similarity_slider.value() / 100.0
            self.similarity_detector.set_similarity_threshold(threshold)
            
    def on_similarity_changed(self, value):
        """Handle similarity threshold change"""
        threshold = value / 100.0
        self.similarity_label.setText(f"{threshold:.2f}")
        self.similarity_detector.set_similarity_threshold(threshold)
        
    def scan_completed(self, duplicates):
        self.duplicates = duplicates
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        
        if duplicates:
            total_groups = len(duplicates)
            total_files = sum(len(files) for files in duplicates.values())
            self.status_label.setText(f"Scan complete! Found {total_groups} groups with {total_files} duplicate files")
            self.stats_label.setText(f"📊 {total_groups} groups • {total_files} duplicates")
            self.populate_groups()
            self.tab_widget.setCurrentIndex(1)  # Switch to results tab
            self.update_statistics()
        else:
            self.status_label.setText("Scan complete - No duplicates found")
            self.stats_label.setText("✅ No duplicates found")
            self.groups_list.clear()
            self.files_table.setRowCount(0)
            self.preview_area.clear()
            
    def populate_groups(self):
        self.groups_list.clear()
        for i, (hash_val, files) in enumerate(self.duplicates.items()):
            total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
            size_mb = total_size / (1024 * 1024)
            group_name = f"Group {i+1} ({len(files)} files, {size_mb:.1f} MB)"
            self.groups_list.addItem(group_name)
            
        # Enable toolbar buttons
        self.select_all_btn.setEnabled(True)
        self.clear_selection_btn.setEnabled(True)
            
    def show_group_files(self, current, previous):
        if current is None:
            return
            
        group_index = self.groups_list.row(current)
        hash_val = list(self.duplicates.keys())[group_index]
        files = self.duplicates[hash_val]
        
        # Populate table
        self.files_table.setRowCount(len(files))
        
        for row, file_path in enumerate(files):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(self.update_selection)
            self.files_table.setCellWidget(row, 0, checkbox)
            
            # File name
            filename_item = QTableWidgetItem(os.path.basename(file_path))
            self.files_table.setItem(row, 1, filename_item)
            
            # Full path
            path_item = QTableWidgetItem(file_path)
            self.files_table.setItem(row, 2, path_item)
            
            # Size
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                size_item = QTableWidgetItem(f"{size_mb:.2f} MB")
                self.files_table.setItem(row, 3, size_item)
            except:
                size_item = QTableWidgetItem("Unknown")
                self.files_table.setItem(row, 3, size_item)
                
            # Similarity score (for text files when using text similarity)
            similarity_text = "-"
            if self.hash_combo.currentText() == "Text Similarity" and len(files) > 1:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.txt', '.py', '.js', '.html', '.css', '.md']:
                    # Calculate similarity with first file in group
                    if row > 0:
                        first_file = files[0]
                        similarity = self.similarity_detector.calculate_text_similarity(first_file, file_path)
                        similarity_text = f"{similarity:.3f}"
                        
            similarity_item = QTableWidgetItem(similarity_text)
            self.files_table.setItem(row, 4, similarity_item)
                
        # Connect row selection to preview
        self.files_table.cellClicked.connect(self.show_file_preview)
            
    def show_file_preview(self, row, column):
        file_item = self.files_table.item(row, 2)
        if not file_item:
            return
            
        file_path = file_item.text()
        
        try:
            stat = os.stat(file_path)
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            preview_text = f"📄 File: {os.path.basename(file_path)}\n"
            preview_text += f"📁 Path: {file_path}\n"
            preview_text += f"📏 Size: {size_mb:.2f} MB\n"
            preview_text += f"📅 Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.txt', '.py', '.js', '.html', '.css', '.md']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(500)
                        preview_text += f"📖 Content preview:\n{content}"
                except:
                    preview_text += "❌ Cannot preview text content"
                    
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] and HAS_PILLOW:
                preview_text += "🖼️ Image file - Right-click to view in system viewer"
            else:
                preview_text += f"📎 File type: {ext or 'Unknown'}"
                
            self.preview_area.setText(preview_text)
            
        except Exception as e:
            self.preview_area.setText(f"❌ Error reading file: {e}")
            
    def update_selection(self):
        """Update selection tracking and UI"""
        self.selected_files.clear()
        
        for row in range(self.files_table.rowCount()):
            checkbox_widget = self.files_table.cellWidget(row, 0)
            file_item = self.files_table.item(row, 2)
            if checkbox_widget and hasattr(checkbox_widget, 'isChecked') and hasattr(checkbox_widget, 'isChecked') and file_item:
                if checkbox_widget.isChecked():  # type: ignore
                    file_path = file_item.text()
                    self.selected_files.add(file_path)
                
        # Update UI
        count = len(self.selected_files)
        self.selection_label.setText(f"{count} files selected")
        self.delete_selected_btn.setEnabled(count > 0)
        
        # Calculate total size
        total_size = 0
        for file_path in self.selected_files:
            try:
                total_size += os.path.getsize(file_path)
            except:
                pass
                
        size_mb = total_size / (1024 * 1024)
        self.total_size_label.setText(f"Total size: {size_mb:.1f} MB")
        
    def select_all_files(self):
        """Select all files in current view"""
        for row in range(self.files_table.rowCount()):
            checkbox_widget = self.files_table.cellWidget(row, 0)
            if checkbox_widget and hasattr(checkbox_widget, 'setChecked'):
                checkbox_widget.setChecked(True)  # type: ignore
                
    def clear_selection(self):
        """Clear all selections"""
        for row in range(self.files_table.rowCount()):
            checkbox_widget = self.files_table.cellWidget(row, 0)
            if checkbox_widget and hasattr(checkbox_widget, 'setChecked'):
                checkbox_widget.setChecked(False)  # type: ignore
                
    def delete_selected_files(self):
        """Delete selected files with undo support"""
        if not self.selected_files:
            QMessageBox.warning(self, "Warning", "No files selected")
            return
            
        reply = QMessageBox.question(
            self, 
            "Confirm Batch Delete",
            f"Are you sure you want to delete {len(self.selected_files)} selected files?\n\n"
            "These files will be backed up and can be restored using 'Undo Last'.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success_count = 0
            failed_files = []
            
            for file_path in self.selected_files.copy():
                try:
                    # Backup first
                    if self.undo_manager.backup_file(file_path):
                        os.remove(file_path)
                        success_count += 1
                        self.selected_files.remove(file_path)
                    else:
                        failed_files.append(file_path)
                except Exception as e:
                    failed_files.append(file_path)
                    print(f"Failed to delete {file_path}: {e}")
            
            # Update UI
            self.refresh_current_group()
            self.update_selection()
            self.undo_btn.setEnabled(True)
            self.refresh_undo_history()
            
            # Show result
            if failed_files:
                QMessageBox.warning(
                    self, 
                    "Partial Success", 
                    f"Successfully deleted {success_count} files.\n"
                    f"Failed to delete {len(failed_files)} files."
                )
            else:
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Successfully deleted {success_count} files.\n"
                    "Use 'Undo Last' to restore if needed."
                )
                
    def undo_last_operation(self):
        """Undo the last delete operation"""
        success, message = self.undo_manager.undo_last_operation()
        
        if success:
            QMessageBox.information(self, "Undo Successful", message)
            self.refresh_current_group()
            self.refresh_undo_history()
        else:
            QMessageBox.warning(self, "Undo Failed", message)
            
        # Disable undo button if no more operations
        if not self.undo_manager.operations:
            self.undo_btn.setEnabled(False)
            
    def refresh_current_group(self):
        """Refresh the currently displayed group"""
        current_item = self.groups_list.currentItem()
        if current_item:
            self.show_group_files(current_item, None)
            
    def refresh_undo_history(self):
        """Refresh the undo history display"""
        self.undo_list.clear()
        operations = self.undo_manager.get_recent_operations()
        
        for op in reversed(operations):  # Show most recent first
            if op['type'] == 'delete':
                filename = os.path.basename(op['original_path'])
                timestamp = op['timestamp']
                item_text = f"🗑️ Deleted: {filename} ({timestamp})"
                self.undo_list.addItem(item_text)
                
    def update_statistics(self):
        """Update statistics display"""
        if not self.duplicates:
            return
            
        total_groups = len(self.duplicates)
        total_files = sum(len(files) for files in self.duplicates.values())
        
        total_size = 0
        wasted_space = 0
        
        for files in self.duplicates.values():
            if files:
                file_size = 0
                for file_path in files:
                    try:
                        size = os.path.getsize(file_path)
                        total_size += size
                        file_size = size
                    except:
                        pass
                # Wasted space = size of duplicates (keep one)
                wasted_space += file_size * (len(files) - 1)
                
        total_size_mb = total_size / (1024 * 1024)
        wasted_space_mb = wasted_space / (1024 * 1024)
        
        stats_text = f"""
📊 Scan Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Total duplicate groups: {total_groups}
📄 Total duplicate files: {total_files}
📏 Total size of duplicates: {total_size_mb:.1f} MB
💾 Potential space savings: {wasted_space_mb:.1f} MB

🔍 Scan settings:
• Algorithm: {self.hash_combo.currentText()}
• Min size filter: {self.min_size_spin.value()} MB
• Max size filter: {self.max_size_spin.value()} MB
• File type filter: {self.file_types_combo.currentText()}

💡 Tip: Select duplicates and use batch delete to free up space!
        """
        
        self.stats_text.setText(stats_text.strip())
        
    # ============ NEW UX ENHANCEMENT METHODS ============
    
    def browse_directory(self):
        """Browse directory method for menu/toolbar shortcuts"""
        self.select_directory()
        
    def delete_selected_shortcut(self):
        """Delete selected files using Del key"""
        selected_files = []
        for row in range(self.files_table.rowCount()):
            checkbox_item = self.files_table.item(row, 0)
            path_item = self.files_table.item(row, 1)
            if checkbox_item and path_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                selected_files.append(path_item.text())
        
        if selected_files:
            self.delete_selected_files()
        else:
            QMessageBox.information(self, "No Selection", "Please select files to delete by checking the boxes.")
            
    def preview_selected_shortcut(self):
        """Preview selected file using Ctrl+P"""
        current_row = self.files_table.currentRow()
        if current_row >= 0:
            path_item = self.files_table.item(current_row, 1)
            if path_item:
                file_path = path_item.text()
                self.show_file_preview(current_row, 1)
            else:
                QMessageBox.information(self, "No Selection", "Please select a valid file to preview.")
        else:
            QMessageBox.information(self, "No Selection", "Please select a file to preview.")
            
    def select_all_duplicates(self):
        """Select all duplicates using Ctrl+A"""
        self.select_all_files()
                    
    def undo_last_action(self):
        """Undo last action using Ctrl+Z"""
        self.undo_last_operation()
            
    def export_results(self):
        """Export results using Ctrl+E"""
        if not self.duplicates:
            QMessageBox.information(self, "No Data", "No duplicate results to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "duplicates_report.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Group', 'File Path', 'Size (MB)', 'Modified Date'])
                    
                    for i, (hash_val, files) in enumerate(self.duplicates.items()):
                        for file_path in files:
                            if os.path.exists(file_path):
                                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                                writer.writerow([f"Group {i+1}", file_path, f"{size_mb:.2f}", mod_time.strftime("%Y-%m-%d %H:%M:%S")])
                                
                QMessageBox.information(self, "Export Complete", f"Results exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
        
    def filter_duplicates(self, search_text):
        """Filter displayed duplicates based on search text"""
        if not hasattr(self, 'original_duplicates'):
            self.original_duplicates = self.duplicates.copy()
            
        if not search_text.strip():
            # Show all duplicates if search is empty
            self.duplicates = self.original_duplicates.copy()
        else:
            # Filter duplicates based on search text
            filtered_duplicates = {}
            search_lower = search_text.lower()
            
            for hash_val, files in self.original_duplicates.items():
                matching_files = []
                for file_path in files:
                    filename = os.path.basename(file_path).lower()
                    full_path = file_path.lower()
                    if search_lower in filename or search_lower in full_path:
                        matching_files.append(file_path)
                
                if len(matching_files) > 1:  # Only show if still duplicates after filtering
                    filtered_duplicates[hash_val] = matching_files
                    
            self.duplicates = filtered_duplicates
            
        # Update the groups display
        self.populate_groups()
        
        # Update stats
        if self.duplicates:
            total_groups = len(self.duplicates)
            total_files = sum(len(files) for files in self.duplicates.values())
            search_info = f" (filtered)" if search_text.strip() else ""
            self.stats_label.setText(f"📊 {total_groups} groups • {total_files} duplicates{search_info}")
        else:
            self.stats_label.setText("🔍 No matches found" if search_text.strip() else "✅ No duplicates found")
            
    def clear_search(self):
        """Clear search box and show all duplicates"""
        self.search_box.clear()
        
    def connect_progress_signals(self):
        """Connect the enhanced progress signals"""
        if hasattr(self, 'scan_thread'):
            # Connect new progress signals
            self.scan_thread.current_file.connect(self.update_current_file)
            self.scan_thread.time_estimate.connect(self.update_time_estimate)
            
    def update_current_file(self, filename):
        """Update status bar with current file being processed"""
        self.status_bar.showMessage(filename)
        
    def update_time_estimate(self, time_text):
        """Update progress label with time estimate"""
        if hasattr(self, 'progress_label'):
            self.progress_label.setText(time_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernDuplicateFinder()
    window.show()
    sys.exit(app.exec())
