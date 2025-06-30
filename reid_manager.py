import numpy as np
import logging
from utils import feature_similarity

class ReIdentificationManager:
    def __init__(self, feature_threshold=0.5):
        self.feature_threshold = feature_threshold
        self.logger = logging.getLogger('ReIdentificationManager')

    def handle_reidentification(self, new_detection, lost_tracks):
        best_track = None
        best_score = 0
        for track in lost_tracks:
            sim = feature_similarity(new_detection['features'], track.features)
            self.logger.debug(f"Comparing new detection with lost track {track.track_id}. Similarity: {sim:.4f}")
            if sim > best_score and sim > self.feature_threshold:
                best_score = sim
                best_track = track
        
        if best_track:
            self.logger.info(f"Re-identified detection as lost track {best_track.track_id} with high similarity {best_score:.4f}")
            return best_track, best_score

        return None, best_score 