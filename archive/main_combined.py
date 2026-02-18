#!/usr/bin/env python3
"""
Combined application: Enhanced UI (from main_enhanced.py) + Enterprise backends (from main_enterprise.py).
This module creates a single runnable entrypoint that preserves the enhanced UI and, when the user opts in,
uses the EnterpriseWorker for scanning (database, GPU, plugins, memory, merging).
"""
import sys
from PyQt6.QtWidgets import QMessageBox

try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    raise RuntimeError("PyQt6 is required to run this application. Install with: pip install PyQt6")

from main_enhanced import ModernDuplicateFinder

try:
    # Import enterprise components (may be absent; EnterpriseWorker handles missing modules)
    from main_enterprise import EnterpriseWorker, EnterpriseConfigDialog, ScanProgress
    HAS_ENTERPRISE = True
except Exception:
    EnterpriseWorker = None
    EnterpriseConfigDialog = None
    ScanProgress = None
    HAS_ENTERPRISE = False


class CombinedDuplicateFinder(ModernDuplicateFinder):
    """Enhanced UI that can optionally use enterprise backends for scanning."""

    def __init__(self):
        super().__init__()
        self.enterprise_worker = None

    def scan_duplicates(self):
        """Override scan to optionally use EnterpriseWorker when the user chooses enterprise mode."""
        if not self.directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first")
            return

        # Ask the user whether to use enterprise backends
        if HAS_ENTERPRISE:
            resp = QMessageBox.question(
                self,
                "Use Enterprise Backends?",
                "Use enterprise backends (database persistence, GPU, plugins, merging) for this scan?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if resp == QMessageBox.StandardButton.Yes:
                # Show enterprise config dialog
                dialog = EnterpriseConfigDialog(self)
                if dialog.exec() == dialog.DialogCode.Accepted:
                    config = dialog.get_config()
                else:
                    config = {
                        "use_database": True,
                        "incremental_scan": True,
                        "use_gpu": False,
                        "use_gpu_hashing": False,
                        "use_plugins": True,
                        "optimize_memory": True,
                        "enable_merging": True,
                        "analyze_merges": True,
                    }

                # Start enterprise worker
                folders = [self.directory]
                self.enterprise_worker = EnterpriseWorker(folders, config)
                self.enterprise_worker.progress_updated.connect(self._on_enterprise_progress)
                self.enterprise_worker.duplicates_found.connect(self._on_enterprise_results)
                self.enterprise_worker.error_occurred.connect(self._on_enterprise_error)
                self.enterprise_worker.finished.connect(self._on_enterprise_finished)

                # UI state
                self.scan_btn.setEnabled(False)
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                self.status_label.setText("Starting enterprise scan...")

                self.enterprise_worker.start()
                return

        # Fallback to default enhanced scan
        super().scan_duplicates()

    def _on_enterprise_progress(self, progress: 'ScanProgress'):
        """Map Enterprise ScanProgress to the enhanced UI widgets."""
        try:
            total = getattr(progress, 'total_files', 0) or 0
            processed = getattr(progress, 'files_processed', 0) or 0
            if total > 0:
                percent = int((processed / total) * 100)
            else:
                percent = 0

            self.progress_bar.setValue(percent)
            # Use status_label for phase and progress_label for current file / ETA
            phase = getattr(progress, 'current_phase', '') or ''
            current_file = getattr(progress, 'current_file', '') or ''
            self.status_label.setText(f"{phase}")
            if current_file:
                self.progress_label.setText(current_file)

            # Update stats_label with duplicates/bytes if available
            duplicates = getattr(progress, 'duplicates_found', 0) or 0
            bytes_saved = getattr(progress, 'bytes_saved', 0) or 0
            self.stats_label.setText(f"📊 {processed}/{total} files • {duplicates} duplicates • Saved: {self._format_size(bytes_saved)}")
        except Exception as e:
            print(f"Error mapping enterprise progress: {e}")

    def _on_enterprise_results(self, duplicates: dict):
        """Receive duplicates from EnterpriseWorker and populate enhanced UI results."""
        try:
            # EnterpriseWorker returns groups keyed by hash -> list of files
            self.duplicates = duplicates
            self.populate_groups()
            # Switch to results tab
            try:
                self.tab_widget.setCurrentIndex(1)
            except Exception:
                pass
        except Exception as e:
            print(f"Error receiving enterprise results: {e}")

    def _on_enterprise_error(self, message: str):
        QMessageBox.critical(self, "Enterprise Error", message)
        self._on_enterprise_finished()

    def _on_enterprise_finished(self):
        self.scan_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Enterprise scan finished")

    def _format_size(self, size_bytes: int) -> str:
        try:
            if size_bytes == 0:
                return "0 B"
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:3.1f} {unit}"
                size_bytes /= 1024.0
        except Exception:
            pass
        return str(size_bytes)


def main():
    app = QApplication(sys.argv)
    window = CombinedDuplicateFinder()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
