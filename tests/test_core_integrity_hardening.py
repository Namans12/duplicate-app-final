import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from database_manager import DatabaseManager
from plugin_system import PluginManager
import main


class DatabaseManagerNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dup_db_")
        self.old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_relative_paths_round_trip_through_cache_and_incremental_scan(self):
        os.makedirs("data", exist_ok=True)
        relative_path = os.path.join("data", "sample.txt")
        with open(relative_path, "w", encoding="utf-8") as handle:
            handle.write("hello")

        db = DatabaseManager(os.path.join(self.tmpdir, "cache.db"))
        db.cache_file_info(relative_path, {"md5": "abc123"})

        cached = db.get_cached_file_info(relative_path)
        self.assertIsNotNone(cached)
        self.assertTrue(os.path.isabs(cached["file_path"]))

        new_files, modified_files = db.get_incremental_files("data")
        self.assertEqual(new_files, [])
        self.assertEqual(modified_files, [])


class PluginManagerInstallTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dup_plugins_")
        self.plugin_root = os.path.join(self.tmpdir, "plugins")
        self.manager = PluginManager(self.plugin_root)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_install_plugin_rejects_invalid_plugin_type(self):
        source = os.path.join(self.tmpdir, "bad_plugin.py")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write("print('hi')\n")

        installed = self.manager.install_plugin(source, "../escape")

        self.assertFalse(installed)
        self.assertFalse(os.path.exists(os.path.join(self.plugin_root, "..", "escape", "bad_plugin.py")))

    def test_install_plugin_rejects_non_python_files(self):
        source = os.path.join(self.tmpdir, "notes.txt")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write("not a plugin")

        installed = self.manager.install_plugin(source, "hashing")

        self.assertFalse(installed)
        self.assertFalse(os.path.exists(os.path.join(self.plugin_root, "hashing", "notes.txt")))


class DeleteRollbackTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dup_delete_")
        self.undo_manager = main.UndoManager()
        self.old_backup_dir = self.undo_manager.backup_dir
        self.undo_manager.backup_dir = os.path.join(self.tmpdir, "backup")
        self.undo_manager._ensure_backup_dir()

    def tearDown(self):
        self.undo_manager.backup_dir = self.old_backup_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_failed_delete_discards_backup_and_undo_entry(self):
        file_path = os.path.join(self.tmpdir, "locked.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("data")

        real_remove = main.os.remove

        def fake_remove(path):
            normalized = path.replace("\\\\?\\", "")
            if os.path.abspath(normalized) == os.path.abspath(file_path):
                raise PermissionError("blocked")
            return real_remove(path)

        thread = main.BulkDeleteThread(
            [file_path],
            use_recycle=False,
            undo_manager=self.undo_manager,
            undo_enabled=True,
        )

        with patch.object(main.os, "remove", side_effect=fake_remove):
            thread.run()

        self.assertTrue(os.path.exists(file_path))
        self.assertEqual(self.undo_manager.get_undo_count(), 0)
        self.assertEqual(os.listdir(self.undo_manager.backup_dir), [])


if __name__ == "__main__":
    unittest.main()
