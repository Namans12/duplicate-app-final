# 🔍 Smart Duplicate Finder Pro

A modern, feature-rich GUI application for finding and managing duplicate files with AI-powered detection and advanced filtering capabilities.

## ✨ New Features

### 🎯 Core Enhancements
- **🔄 Undo Functionality**: Files are automatically backed up before deletion with one-click restore
- **✅ Batch Operations**: Select multiple files across groups for bulk deletion or operations
- **🔧 Advanced File Filtering**: Filter by file size, date modified, and file types before scanning
- **📊 Modern Tabbed Interface**: Organized workflow with Scanner, Results, and Settings tabs

### 🎨 Enhanced UI/UX
- **Modern Design**: Gradient headers, rounded corners, and professional styling
- **Smart Layout**: Tabbed interface with dedicated sections for different tasks
- **Progress Tracking**: Real-time status updates and detailed progress information
- **Visual Statistics**: Comprehensive stats showing space usage and potential savings
- **Responsive Design**: Resizable interface that adapts to different screen sizes

### 🚀 Improved Functionality
- **Enhanced File Table**: Checkbox selection, sortable columns, and detailed file information
- **Better Preview**: Rich file previews with metadata (size, modification date, content)
- **Smart Status Bar**: Live selection count and total size calculations
- **Undo History**: View and manage recent delete operations
- **Error Handling**: Graceful handling of file access issues and edge cases

## 🛠️ Technical Features

### Hash Algorithms
- **MD5**: Fast hashing for general duplicate detection
- **SHA256**: Secure hashing for critical applications  
- **Perceptual**: AI-powered image similarity detection (requires Pillow)

### File Type Support
- **Images**: JPG, PNG, GIF, BMP, TIFF with perceptual hashing
- **Videos**: MP4, AVI, MOV, MKV, WMV, FLV
- **Audio**: MP3, WAV, FLAC, AAC, OGG
- **Documents**: PDF, DOC, DOCX, TXT, RTF
- **Archives**: ZIP, RAR, 7Z, TAR, GZ
- **Code Files**: PY, JS, HTML, CSS, MD

### Advanced Filtering
- **Size Range**: Set minimum and maximum file sizes (MB)
- **Date Range**: Filter by modification date
- **File Types**: Pre-defined categories or custom extensions
- **Path Exclusions**: Skip system or temporary directories

## 📋 Requirements

```
PyQt6>=6.4.0
Pillow>=9.0.0  # Optional: For image perceptual hashing
opencv-python>=4.5.0  # Optional: For video processing
sentence-transformers>=2.0.0  # Optional: For text similarity
pydub>=0.25.0  # Optional: For audio processing
numpy>=1.20.0  # Optional: For audio/video processing
```

## 🚀 Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python main_enhanced.py
   ```

3. **Basic Workflow**:
   - 📁 Select directory to scan
   - 🔧 Configure filters and hash algorithm
   - 🚀 Start scan and wait for results
   - ✅ Select duplicates to delete
   - 🗑️ Batch delete with automatic backup
   - ↶ Undo if needed

## 🎮 User Interface Guide

### 🔍 Scanner Tab
- **Directory Selection**: Choose folder to scan for duplicates
- **File Filters**: Set size, date, and type filters
- **Scan Options**: Select hash algorithm and start scan
- **Progress Monitoring**: Real-time progress and status updates

### 📊 Results Tab
- **Duplicate Groups**: Browse groups of identical files
- **File Selection**: Use checkboxes for batch operations
- **File Preview**: View file details and content preview
- **Batch Actions**: Select all, clear selection, delete selected

### ⚙️ Settings Tab
- **Undo History**: View recent delete operations
- **Statistics**: Detailed scan results and space analysis
- **Configuration**: Advanced settings and preferences

## 🔒 Safety Features

- **Automatic Backup**: Files are copied to `.duplicate_finder_backup/` before deletion
- **Confirmation Dialogs**: Multiple confirmations before destructive operations
- **Undo System**: Restore accidentally deleted files with one click
- **Error Recovery**: Graceful handling of file access issues
- **Progress Tracking**: Never lose track of long-running operations

## 🎯 Pro Tips

1. **Start Small**: Test on a small directory first to understand the interface
2. **Use Filters**: Apply size and type filters to focus on relevant duplicates
3. **Preview First**: Always preview files before deletion to avoid mistakes
4. **Batch Operations**: Select multiple files for efficient cleanup
5. **Keep Backups**: Don't empty the backup folder until you're sure

## 🐛 Troubleshooting

- **GUI Not Showing**: Run from Windows Command Prompt, not VS Code terminal
- **Import Errors**: Install optional dependencies for full functionality
- **Performance**: Use filters to reduce scan scope on large directories
- **Permissions**: Run as administrator for system directories

## 📝 License

This project is open source and available under the MIT License.
