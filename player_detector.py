import numpy as np
import logging
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, yolo_weights, conf_threshold=0.5, device='cpu'):
        self.model = YOLO(yolo_weights)
        self.conf_threshold = conf_threshold
        self.device = device
        self.logger = logging.getLogger('PlayerDetector')

    def detect_players(self, frame):
        try:
            results = self.model(frame, device=self.device)
            detections = []
            if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
                return detections
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            clss = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
            for bbox, conf, cls in zip(xyxy, confs, clss):
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, bbox)
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'class': int(cls)
                    })
            return detections
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return [] 