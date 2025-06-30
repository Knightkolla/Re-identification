import unittest
import numpy as np
from player_detector import PlayerDetector

class DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, frame, device=None):
        class Box:
            def __init__(self):
                self.conf = 0.9
                self.cls = 0
                self.xyxy = [10, 10, 50, 50]
        class Result:
            boxes = [Box()]
        return [Result()]

class TestPlayerDetector(unittest.TestCase):
    def setUp(self):
        # Patch YOLO in PlayerDetector
        PlayerDetector.model = DummyYOLO()
        self.detector = PlayerDetector(yolo_weights=None)
        self.detector.model = DummyYOLO()
        self.frame = np.ones((100, 100, 3), dtype=np.uint8)

    def test_detect_players(self):
        detections = self.detector.detect_players(self.frame)
        self.assertIsInstance(detections, list)
        self.assertTrue(all('bbox' in d and 'confidence' in d and 'class' in d for d in detections))

if __name__ == '__main__':
    unittest.main() 