---
mode: agent
---
# Duplicate File Finder with AI Enhancements

You are helping develop a desktop application in **Python** using **PyQt** for GUI.  
The tool finds duplicate files in a given directory based on content, not just filenames.  
It must scale to **huge datasets** using **parallel processing** and possibly **GPU acceleration**.  
Provide **modular, production-ready code** with comments and error handling.

## Features to Implement

1. **Hash-based duplicate detection**  
   - Compute SHA256 (or user-selectable hash).  
   - Detect duplicates even if filenames differ.  
   - Use parallel hashing with multiprocessing/threading.  
   - Optional GPU acceleration for hashing.

2. **Smart File Grouping & Cleanup Strategies**  
   - Group duplicates.  
   - Allow rules like: "Keep newest", "Keep largest", "Keep original folder", etc.  
   - One-click cleanup strategies.

3. **Preview Support**  
   - Show thumbnails/previews for images, videos, PDFs, text files.  
   - Allow user to confirm before deletion.

4. **Content-Aware Duplicate Detection (AI)**  
   - For images → perceptual hashing + embeddings (ResNet/CLIP).  
   - For text → BERT embeddings for paraphrased documents.  
   - For audio → YAMNet/Wav2Vec embeddings.  
   - For video → keyframe embeddings.

5. **UI/UX (PyQt)**  
   - Drag-and-drop folder selection.  
   - Progress bar for hashing.  
   - Interactive table/tree view of duplicate sets.  
   - Preview pane.

6. **Extra**  
   - Export duplicate reports (CSV/Excel).  
   - Undo/Recycle bin integration.  
   - Store hash/embedding index in SQLite to avoid rescanning.

## Coding Guidelines

- Use Python 3.10+ best practices.  
- Use PyQt6 for UI.  
- Keep GUI logic separate from business logic (MVC pattern).  
- Provide docstrings, inline comments, and error handling.  
- Suggest **unit tests** for critical functions.  