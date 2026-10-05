"""Run with: python -m unittest discover -s tests -v."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_fx', ROOT / 'scripts/lenticular/prepare_fx.py')
prepare_fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_fx)


class PrepareFxTests(unittest.TestCase):
    def test_writes_line_masks_without_blender(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'assets').mkdir()
            (root / 'card-config.json').write_text(json.dumps({'identity': {}}), encoding='utf-8-sig')
            # Contrasting block so Sobel/FIND_EDGES produce ink.
            image = Image.new('RGB', (64, 64), 'black')
            for x in range(16, 48):
                for y in range(16, 48):
                    image.putpixel((x, y), (255, 255, 255))
            image.save(root / 'assets' / 'image_a.png')
            image.save(root / 'assets' / 'image_b.png')
            prepare_fx.main(root)
            for state in ('a', 'b'):
                path = root / 'assets' / f'line_{state}.png'
                self.assertTrue(path.is_file(), path)
                mask = Image.open(path)
                self.assertEqual(mask.size, (64, 64))
                self.assertGreater(sum(mask.convert("L").histogram()[1:]), 0)


if __name__ == '__main__':
    unittest.main()
