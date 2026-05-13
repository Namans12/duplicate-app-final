import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, PngImagePlugin
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import main
from media_profiles import compare_image_files, get_default_media_profiles


APP = QApplication.instance() or QApplication([])


def _save_png(path, color, size=(24, 24), metadata=None):
    image = Image.new("RGB", size, color)
    if metadata:
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, value)
        image.save(path, pnginfo=pnginfo)
    else:
        image.save(path)


class StrictImageMatchingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dup_app_tests_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_scan_thread(self, hash_type="MD5"):
        settings = {
            "hash_type": hash_type,
            "find_similar": True,
            "compare_by_name": True,
            "name_similarity": 1.0,
            "compare_by_size": True,
            "size_tolerance": 1,
            "compare_text_content": False,
            "content_similarity": 0.9,
            "media_profiles": get_default_media_profiles(),
        }
        return main.EnhancedScanThread(
            directory=self.tmpdir,
            comparison_settings=settings,
            use_parallel=False,
            use_gpu=False,
            use_gpu_hashing=False,
            gpu_batch_size=1,
        )

    def test_exact_image_content_matches_even_with_metadata_difference(self):
        left = os.path.join(self.tmpdir, "left.png")
        right = os.path.join(self.tmpdir, "right.png")

        _save_png(left, "red", metadata={"camera": "A"})
        _save_png(right, "red", metadata={"camera": "B"})

        match, score, reasons = compare_image_files(left, right, {})

        self.assertTrue(match)
        self.assertEqual(score, 1.0)
        self.assertTrue(any(reason.startswith("Exact image content") for reason in reasons))

    def test_same_named_different_images_do_not_get_grouped_by_name_or_size(self):
        folder_a = os.path.join(self.tmpdir, "a")
        folder_b = os.path.join(self.tmpdir, "b")
        os.makedirs(folder_a, exist_ok=True)
        os.makedirs(folder_b, exist_ok=True)

        first = os.path.join(folder_a, "IMG_9203.png")
        second = os.path.join(folder_b, "IMG_9203.png")
        _save_png(first, "red")
        _save_png(second, "blue")

        similar = self._make_scan_thread().find_similar_files([first, second], {})

        self.assertEqual(similar, {})

    def test_identical_pixels_with_different_names_are_grouped_as_content(self):
        first = os.path.join(self.tmpdir, "photo_a.png")
        second = os.path.join(self.tmpdir, "photo_b.png")
        _save_png(first, "green", metadata={"source": "phone"})
        _save_png(second, "green", metadata={"source": "export"})

        similar = self._make_scan_thread().find_similar_files([first, second], {})

        self.assertEqual(len(similar), 1)
        group = next(iter(similar.values()))
        self.assertEqual(group["match_type"], "content")
        self.assertCountEqual(group["files"], [first, second])
        self.assertTrue(any(reason.startswith("Exact image content") for reason in group["reasons"]))

    def test_perceptual_similarity_reuses_phase1_hashes(self):
        first = os.path.join(self.tmpdir, "cached_a.png")
        second = os.path.join(self.tmpdir, "cached_b.png")
        _save_png(first, "red")
        _save_png(second, "blue")

        thread = self._make_scan_thread(hash_type="Perceptual")
        thread._phase1_image_hash_cache = {
            first: "0000000000000000",
            second: "0000000000000001",
        }

        with patch.object(main, "compare_image_files", side_effect=AssertionError("unexpected image reopen")):
            similar = thread.find_similar_files([first, second], {})

        self.assertEqual(len(similar), 1)
        group = next(iter(similar.values()))
        self.assertEqual(group["match_type"], "content")
        self.assertCountEqual(group["files"], [first, second])
        self.assertTrue(any(reason.startswith("phash ") for reason in group["reasons"]))

    def test_resolve_gpu_batch_size_upgrades_legacy_default_only_when_requested(self):
        with patch.object(main, "HAS_CUDA", True), patch.object(main, "_default_gpu_batch_size", return_value=512):
            self.assertEqual(main._resolve_gpu_batch_size(32), 512)
            self.assertEqual(main._resolve_gpu_batch_size(32, upgrade_legacy_default=False), 32)
            self.assertEqual(main._resolve_gpu_batch_size(96), 96)

    def test_gpu_hasher_falls_back_to_cpu_after_failed_probe(self):
        hasher = main.GPUHasher(use_gpu=True, backend="cupy")
        hasher.cuda_available = True
        hasher.accelerator = type(
            "Accel",
            (),
            {
                "batch_perceptual_hash_gpu": staticmethod(lambda paths, batch_size=None: [f"gpu:{path}" for path in paths]),
                "_batch_perceptual_hash_cpu": staticmethod(lambda paths: [f"cpu:{path}" for path in paths]),
            },
        )()
        image_paths = [f"img_{i}.png" for i in range(5)]
        with patch.object(main, "_call_gpu_perceptual_batch", return_value=None) as gpu_call:
            results = hasher.batch_hash_images(image_paths, batch_size=2)

        gpu_call.assert_called_once_with(["img_0.png", "img_1.png"], batch_size=2, timeout_seconds=5.0)
        self.assertEqual(results["img_0.png"], "cpu:img_0.png")
        self.assertEqual(results["img_4.png"], "cpu:img_4.png")

    def test_gpu_hasher_uses_direct_accelerator_after_successful_probe(self):
        hasher = main.GPUHasher(use_gpu=True, backend="cupy")
        hasher.cuda_available = True
        hasher.accelerator = type(
            "Accel",
            (),
            {
                "batch_perceptual_hash_gpu": staticmethod(lambda paths, batch_size=None: [f"gpu:{path}" for path in paths]),
                "_batch_perceptual_hash_cpu": staticmethod(lambda paths: [f"cpu:{path}" for path in paths]),
            },
        )()
        image_paths = [f"img_{i}.png" for i in range(3)]

        with patch.object(main, "_call_gpu_perceptual_batch", return_value=["probe"] * 3) as gpu_call:
            results = hasher.batch_hash_images(image_paths, batch_size=8)

        gpu_call.assert_called_once_with(image_paths, batch_size=3, timeout_seconds=5.0)
        self.assertEqual(results["img_0.png"], "gpu:img_0.png")
        self.assertEqual(results["img_2.png"], "gpu:img_2.png")


class MoveCollisionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dup_app_move_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resolve_move_destination_path_uses_numeric_suffixes(self):
        original = os.path.join(self.tmpdir, "IMG_9203.jpg")
        first_suffix = os.path.join(self.tmpdir, "IMG_9203_1.jpg")
        with open(original, "wb") as handle:
            handle.write(b"one")
        with open(first_suffix, "wb") as handle:
            handle.write(b"two")

        resolved = main.resolve_move_destination_path(self.tmpdir, "IMG_9203.jpg")

        self.assertEqual(resolved, os.path.join(self.tmpdir, "IMG_9203_2.jpg"))


class SelectionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dup_app_select_")
        self.window = None

    def tearDown(self):
        if self.window is not None:
            self.window.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def test_selection_persists_when_switching_duplicate_groups(self):
        group1_a = self._create_file("group1_a.txt", "same-a")
        group1_b = self._create_file("group1_b.txt", "same-a")
        group2_a = self._create_file("group2_a.txt", "same-b")
        group2_b = self._create_file("group2_b.txt", "same-b")

        with patch.object(main.GroupQualityThread, "start", autospec=True, side_effect=lambda self: None):
            self.window = main.EnhancedDuplicateFinder()

        self.window.duplicates = {
            "group_1": [group1_a, group1_b],
            "group_2": [group2_a, group2_b],
        }
        self.window.populate_duplicate_groups()

        first_group = self.window.groups_tree.topLevelItem(0)
        second_group = self.window.groups_tree.topLevelItem(1)

        self.window.show_duplicate_group_files(first_group, None)
        APP.processEvents()

        first_group_paths = set()
        for index in range(self.window.duplicate_files_tree.topLevelItemCount()):
            item = self.window.duplicate_files_tree.topLevelItem(index)
            item.setCheckState(0, Qt.CheckState.Checked)
            first_group_paths.add(item.data(0, Qt.ItemDataRole.UserRole))

        APP.processEvents()
        self.window.show_duplicate_group_files(second_group, first_group)
        APP.processEvents()

        self.assertTrue(first_group_paths.issubset(set(self.window.get_selected_files())))

        self.window.show_duplicate_group_files(first_group, second_group)
        APP.processEvents()

        restored = {}
        for index in range(self.window.duplicate_files_tree.topLevelItemCount()):
            item = self.window.duplicate_files_tree.topLevelItem(index)
            restored[item.data(0, Qt.ItemDataRole.UserRole)] = item.checkState(0) == Qt.CheckState.Checked

        for path in first_group_paths:
            self.assertTrue(restored[path])


if __name__ == "__main__":
    unittest.main()
