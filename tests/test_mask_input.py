"""Regression checks for photos accidentally supplied as manual edit masks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import app
from core import masker


class MaskInputChecks(unittest.TestCase):
    def fixture(self):
        pixels = np.full((64, 80, 3), 210, np.uint8)
        pixels[12:52, 30:50] = (200, 30, 30)
        mask = np.zeros((64, 80), np.uint8)
        mask[12:52, 30:50] = 255
        return Image.fromarray(pixels), Image.fromarray(mask)

    def test_original_photo_cannot_be_used_as_manual_mask(self):
        source, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, '原图'):
            app.select_manual(source, source, 'manual')
        with self.assertRaisesRegex(ValueError, '黑白'):
            masker.manual_masks(source, source.size)

    def test_grayscale_original_also_rejected(self):
        source, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, '原图'):
            app.select_manual(source, source.convert('L'), 'manual')

    def test_manual_mode_requires_a_mask(self):
        source, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, '手绘或上传'):
            app.select_manual(source, None, 'manual')

    def test_manual_mode_accepts_real_mask_and_checks_size(self):
        source, mask = self.fixture()
        np.testing.assert_array_equal(app.select_manual(source, mask, 'manual'), mask)
        with self.assertRaisesRegex(ValueError, '同尺寸'):
            app.select_manual(source, mask.resize((32, 32)), 'manual')

    def test_grayscale_photo_is_not_silently_thresholded(self):
        photo = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (64, 1)))
        with self.assertRaisesRegex(ValueError, '黑白'):
            masker.manual_masks(photo, photo.size)

    def test_auto_mode_ignores_stale_manual_photo_for_preview_and_edit(self):
        source, mask = self.fixture()
        bundle = masker.manual_masks(mask, source.size)
        bundle['source'] = 'grounding_dino_sam'
        for backend in ('grounded_sam', 'clipseg'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(app.editor, 'get_bundle', return_value=bundle) as segment, \
                    patch.object(app.config, 'OUTPUT_DIR', Path(tmp)):
                _, info = app.preview(source, '把外套换成蓝色', source, .4, 15, 21, False,
                                      mask_backend=backend, mask_mode='auto')
                self.assertEqual(info['mask_mode'], 'auto')
                result, _, _, record, _ = app.run(source, '把外套换成蓝色', source,
                    'quality', 30, 9, 0, 42, .4, 15, 21, False, backend, 'auto')
                for call in segment.call_args_list:
                    self.assertIsNone(call.args[2])
                    self.assertEqual(call.kwargs['backend'], backend)
                self.assertEqual(record['mask_backend'], backend)
                self.assertEqual(record['mask_mode'], 'auto')
                np.testing.assert_array_equal(np.asarray(result)[np.asarray(mask)==0],
                                              np.asarray(source)[np.asarray(mask)==0])
                self.assertGreater(np.asarray(result)[30, 40, 2], np.asarray(result)[30, 40, 0])


if __name__ == '__main__':
    unittest.main()
