import numpy as np
from scipy.optimize import linear_sum_assignment
import logging
from utils import compute_iou, feature_similarity

class Track:
    def __init__(self, track_id, bbox, features, frame_idx):
        self.track_id = track_id
        self.bbox = bbox
        self.features = features
        self.last_seen = frame_idx
        self.active = True
        self.missed = 0

class TrackingEngine:
    def __init__(self, iou_threshold=0.1, feature_threshold=0.3, max_missed=50, reid_buffer_size=150):
        self.tracks = []
        self.lost_tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.max_missed = max_missed # For crossovers/short occlusions
        self.reid_buffer_size = reid_buffer_size # For long-term re-entry
        self.logger = logging.getLogger('TrackingEngine')

    def update_tracks(self, detections, features_list, frame_idx):
        active_tracks = [t for t in self.tracks if t.active]
        cost_matrix = np.ones((len(active_tracks), len(detections)))

        for i, track in enumerate(active_tracks):
            for j, det in enumerate(detections):
                iou = compute_iou(track.bbox, det['bbox'])
                sim = feature_similarity(track.features, det['features'])
                if iou > self.iou_threshold and sim > self.feature_threshold:
                    cost = 1.0 - (0.5 * iou + 0.5 * sim)
                    cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_track_indices = set(row_ind)
        matched_det_indices = set(col_ind)

        # Update matched tracks
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1.0:
                track = active_tracks[r]
                det = detections[c]
                track.bbox = det['bbox']
                track.features = det['features']
                track.last_seen = frame_idx
                track.missed = 0
                det['track_id'] = track.track_id
                self.logger.debug(f"Frame {frame_idx}: Matched detection {c} to track {track.track_id}")

        # Handle unmatched tracks
        for i, track in enumerate(active_tracks):
            if i not in matched_track_indices:
                track.missed += 1
                if track.missed > self.max_missed:
                    track.active = False
                    self.lost_tracks.append(track)
                    self.logger.debug(f"Frame {frame_idx}: Moved track {track.track_id} to lost buffer after {track.missed} missed frames")
        
        # Prune the lost_tracks buffer to prevent it from growing indefinitely
        self.lost_tracks = [t for t in self.lost_tracks if frame_idx - t.last_seen < self.reid_buffer_size]

        # Handle unmatched detections (potential new tracks or re-id)
        unmatched_detections = []
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                unmatched_detections.append(det)

        return detections, unmatched_detections 