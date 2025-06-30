import argparse
import cv2
import pandas as pd
import time
import logging
from video_processor import VideoProcessor
from player_detector import PlayerDetector
from feature_extractor import FeatureExtractor
from tracking_engine import TrackingEngine, Track
from reid_manager import ReIdentificationManager
from utils import setup_logging

class PlayerReIDSystem:
    def __init__(self, yolo_weights, device='cpu'):
        setup_logging()
        self.logger = logging.getLogger('PlayerReIDSystem')
        self.detector = PlayerDetector(yolo_weights, device=device)
        self.extractor = FeatureExtractor(device=device)
        self.tracker = TrackingEngine()
        self.reid_manager = ReIdentificationManager()

    def process_video(self, video_path, output_path=None, csv_path='tracking.csv', frame_skip=1):
        video = VideoProcessor(video_path, output_path)
        video.open()
        total_frames = video.get_total_frames()
        video.setup_writer()
        tracking_data = []
        frame_idx = 0
        start_time = time.time()
        prev_features = {}
        while True:
            frame = video.read_frame()
            if frame is None:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            detections = self.detector.detect_players(frame)
            bboxes = [det['bbox'] for det in detections]
            features_list = []
            for i, det in enumerate(detections):
                prev_bbox = prev_features.get(i, {}).get('bbox') if i in prev_features else None
                context_bboxes = bboxes[:i] + bboxes[i+1:]
                features = self.extractor.extract_features(frame, det['bbox'], prev_bbox=prev_bbox, context_bboxes=context_bboxes)
                if features is None:
                    continue
                det['features'] = features
                features_list.append(features)
            detections, unmatched_dets = self.tracker.update_tracks(detections, features_list, frame_idx)

            # Handle re-identification for unmatched detections
            for det in unmatched_dets:
                # Check if the new detection matches any recently lost tracks
                matched_track, score = self.reid_manager.handle_reidentification(
                    det, self.tracker.lost_tracks
                )
                if matched_track:
                    # Reactivate the track
                    matched_track.active = True
                    matched_track.missed = 0
                    matched_track.bbox = det['bbox']
                    matched_track.features = det['features']
                    det['track_id'] = matched_track.track_id
                    # Remove from lost buffer if it was there
                    self.tracker.lost_tracks = [t for t in self.tracker.lost_tracks if t.track_id != matched_track.track_id]
                else:
                    # Assign a new ID if no suitable match is found in the lost buffer
                    new_track = Track(self.tracker.next_id, det['bbox'], det['features'], frame_idx)
                    self.tracker.tracks.append(new_track)
                    det['track_id'] = self.tracker.next_id
                    self.tracker.next_id += 1
            
            # Draw and record all detections that have an ID
            for det in detections:
                if 'track_id' not in det:
                    continue # Skip detections that couldn't be tracked or re-identified
                x1, y1, x2, y2 = det['bbox']
                track_id = det.get('track_id', -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                tracking_data.append({'frame': frame_idx, 'track_id': track_id, 'bbox': det['bbox']})
            video.write_frame(frame)
            prev_features = {i: {'bbox': det['bbox'], 'features': det['features']} for i, det in enumerate(detections)}
            frame_idx += 1
        video.release()
        elapsed = time.time() - start_time
        self.logger.info(f"Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx/elapsed:.2f} FPS)")
        # Save CSV
        df = pd.DataFrame(tracking_data)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Tracking data saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sports Player Re-Identification System')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--output_path', type=str, default=None, help='Output annotated video path')
    parser.add_argument('--yolo_weights', type=str, required=True, help='YOLO weights file')
    parser.add_argument('--csv_path', type=str, default='tracking.csv', help='CSV output path')
    parser.add_argument('--device', type=str, default='cpu', help='Device for YOLO (cpu/cuda)')
    parser.add_argument('--frame_skip', type=int, default=1, help='Process every Nth frame')
    args = parser.parse_args()
    system = PlayerReIDSystem(yolo_weights=args.yolo_weights, device=args.device)
    system.process_video(args.video_path, args.output_path, args.csv_path, frame_skip=args.frame_skip) 