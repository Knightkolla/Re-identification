import unittest
import numpy as np
from reid_manager import ReIdentificationManager

class DummyTrack:
    def __init__(self, track_id, features, active=True):
        self.track_id = track_id
        self.features = features
        self.active = active

class TestReIdentificationManager(unittest.TestCase):
    def setUp(self):
        self.manager = ReIdentificationManager(feature_threshold=0.1)
        self.features = {'color_hist': np.ones(8), 'texture': np.ones(8), 'position': np.array([0,0]), 'context': np.ones(4)}
        self.tracks = [DummyTrack(1, self.features, True)]

    def test_handle_reidentification(self):
        new_detection = {'features': self.features}
        result = self.manager.handle_reidentification(new_detection, self.tracks)
        self.assertEqual(result['track_id'], 1)
        self.assertGreater(result['reid_confidence'], 0.5)

if __name__ == '__main__':
    unittest.main() 