import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox

import main


APP = QApplication.instance() or QApplication([])


class AppSmokeScanTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dup_app_scan_")
        self.window = main.EnhancedDuplicateFinder()

    def tearDown(self):
        self.window.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def test_md5_scan_finds_exact_duplicates_without_crashing(self):
        self._create_file("a.txt", "same content")
        self._create_file("b.txt", "same content")
        self._create_file("c.txt", "different content")

        self.window.directory = self.tmpdir
        self.window.preset_enabled = False
        for checkbox in self.window.type_checkboxes.values():
            checkbox.setChecked(False)
        self.window.hash_combo.setCurrentText("MD5")
        self.window.find_similar_cb.setChecked(False)

        with patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok), \
             patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok), \
             patch.object(QMessageBox, "critical", return_value=QMessageBox.StandardButton.Ok):
            self.window.scan_duplicates()

            deadline = time.time() + 20
            while self.window.scan_thread is not None and time.time() < deadline:
                APP.processEvents()
                time.sleep(0.01)

        self.assertIsNone(self.window.scan_thread, "scan thread did not finish in time")
        self.assertEqual(len(self.window.duplicates), 1)
        only_group = next(iter(self.window.duplicates.values()))
        self.assertEqual(len(only_group), 2)
        self.assertTrue(self.window.results_status.text().startswith("1 duplicates"))


if __name__ == "__main__":
    unittest.main()
