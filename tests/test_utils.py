import unittest
import numpy as np
from utils import compute_iou, feature_similarity

class TestUtils(unittest.TestCase):
    def test_compute_iou(self):
        boxA = (0, 0, 100, 100)
        boxB = (50, 50, 150, 150)
        iou = compute_iou(boxA, boxB)
        self.assertTrue(0 < iou < 1)
        self.assertAlmostEqual(iou, 0.142857, places=4)

    def test_feature_similarity(self):
        f1 = {'color_hist': np.ones(8), 'texture': np.ones(8), 'position': np.array([0,0]), 'context': np.ones(4)}
        f2 = {'color_hist': np.ones(8), 'texture': np.ones(8), 'position': np.array([0,0]), 'context': np.ones(4)}
        sim = feature_similarity(f1, f2)
        self.assertAlmostEqual(sim, 1.0, places=2)

if __name__ == '__main__':
    unittest.main() 