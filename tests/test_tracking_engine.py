import unittest
import numpy as np
from tracking_engine import TrackingEngine

class TestTrackingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TrackingEngine(iou_threshold=0.1, feature_threshold=0.1, max_missed=2)

    def test_update_tracks_new_and_existing(self):
        detections = [
            {'bbox': (0, 0, 50, 50), 'features': {'color_hist': np.ones(8), 'texture': np.ones(8), 'position': np.array([25,25]), 'context': np.ones(4)}},
            {'bbox': (100, 100, 150, 150), 'features': {'color_hist': np.ones(8), 'texture': np.ones(8), 'position': np.array([125,125]), 'context': np.ones(4)}}
        ]
        features_list = [d['features'] for d in detections]
        updated = self.engine.update_tracks(detections, features_list, 0)
        self.assertEqual(len(self.engine.tracks), 2)
        self.assertTrue(all('track_id' in d for d in updated))
        # Simulate next frame with same detections
        updated2 = self.engine.update_tracks(detections, features_list, 1)
        self.assertEqual(updated2[0]['track_id'], updated[0]['track_id'])
        self.assertEqual(updated2[1]['track_id'], updated[1]['track_id'])

if __name__ == '__main__':
    unittest.main() 