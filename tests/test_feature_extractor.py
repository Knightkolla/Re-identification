import unittest
import numpy as np
import cv2
from feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(hist_bins=8)
        self.frame = np.ones((100, 100, 3), dtype=np.uint8) * 127
        self.bbox = (10, 10, 60, 60)

    def test_extract_features(self):
        features = self.extractor.extract_features(self.frame, self.bbox)
        self.assertIn('color_hist', features)
        self.assertIn('texture', features)
        self.assertIn('position', features)
        self.assertIn('size', features)
        self.assertIn('aspect_ratio', features)
        self.assertIn('velocity', features)
        self.assertIn('context', features)
        self.assertEqual(features['color_hist'].shape[0], 8*8*8)
        self.assertEqual(features['texture'].shape[0], 10)

if __name__ == '__main__':
    unittest.main() 