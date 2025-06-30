# Sports Player Re-Identification System

## Overview
A robust, production-ready system for tracking and re-identifying sports players in a single-camera 15-second video clip. Maintains consistent player IDs even when players leave and re-enter the frame, using YOLO-based detection and multi-modal feature matching.

---

## Features
- **YOLO-based player detection** (Ultralytics YOLO, fine-tuned for sports)
- **Multi-modal feature extraction**: color, texture, spatial, temporal, and contextual features
- **Robust tracking and re-identification**: Handles occlusions, re-entries, and similar-looking players
- **Performance optimized**: Frame skipping, efficient data structures, and fast feature computation
- **Comprehensive output**: Annotated video, CSV tracking data, and performance metrics

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repo-url>
cd Computer Vison
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLO Weights
- Place your YOLO weights file (e.g., `yolov8-players.pt`) in the project directory.
- You can use a fine-tuned YOLOv8 model for sports player/ball detection.

### 4. Run the System
```bash
python main.py --video_path <input_video.mp4> --output_path <output_annotated.mp4>
```

---

## File Structure
- `main.py`: Entry point and pipeline orchestration
- `video_processor.py`: Video I/O and frame handling
- `player_detector.py`: YOLO-based detection
- `feature_extractor.py`: Multi-modal feature extraction
- `tracking_engine.py`: Track management and ID assignment
- `reid_manager.py`: Re-identification logic
- `utils.py`: Helper functions and logging
- `tests/`: Unit tests for core modules
- `requirements.txt`: Python dependencies

---

## Output
- **Annotated Video**: Output video with bounding boxes and consistent player IDs
- **CSV Tracking Data**: Frame-by-frame player positions and IDs
- **Performance Metrics**: Accuracy, speed, and ID switch statistics

---

## Example Usage
```python
from main import PlayerReIDSystem

system = PlayerReIDSystem(yolo_weights='yolov8-players.pt')
system.process_video('input.mp4', 'output.mp4')
```

---

## Notes
- Designed for single-camera, short sports clips
- Optimized for reliability and ID consistency
- Easily extensible for more advanced features

---

## License
MIT 