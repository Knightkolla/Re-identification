# Sports Player Re-Identification System

## Overview
A robust, production-ready system for tracking and re-identifying sports players in a single-camera video. Maintains consistent player IDs even when players leave and re-enter the frame, using YOLO-based detection and multi-modal feature matching.

---

## Features
- **YOLO-based player detection** (Ultralytics YOLO, fine-tuned for sports)
- **Multi-modal feature extraction**: color, texture, spatial, temporal, and contextual features
- **Robust tracking and re-identification**: Handles occlusions, re-entries, and similar-looking players
- **Comprehensive output**: Annotated video, CSV tracking data, and performance metrics

---

## System Flow
1. **Video Input**: The system reads the input sports video frame by frame.
2. **Player Detection**: Each frame is processed by a YOLO model to detect all players.
3. **Feature Extraction**: For each detected player, the system extracts appearance and context features (color, position, etc.).
4. **Tracking & Re-Identification**: The system matches detected players across frames using feature similarity and assigns consistent IDs, even if players leave and re-enter the frame.
5. **Output Generation**: The system writes an annotated output video (with bounding boxes and IDs) and a CSV file with tracking data for each frame.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Knightkolla/Re-identification.git
cd Re-identification
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
All required Python packages are listed in `requirements.txt`. Install them with:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### **Dependencies Used**
- `opencv-python` (cv2): Video processing and image operations
- `ultralytics` (YOLO): Player detection
- `numpy`: Numerical operations
- `scikit-learn`: Feature processing (optional, but recommended)
- `scikit-image`: Image feature extraction (if used)
- `tqdm`: Progress bars (optional)
- (See `requirements.txt` for exact versions)

### 4. Download YOLO Weights
- Place your YOLO weights file (e.g., `best.pt`) in the project directory.
- You can use a fine-tuned YOLOv8 model for sports player/ball detection.

### 5. Add Your Input Video
- Place your input video (e.g., `input.mp4`) in the project directory.

### 6. Run the System
```bash
python main.py --video_path input.mp4 --output_path output.mp4 --yolo_weights best.pt
```
- This will process the video, generate an annotated output video, and save tracking results to `tracking.csv`.

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
- **CSV Tracking Data**: Frame-by-frame player positions and IDs (`tracking.csv`)
- **Performance Metrics**: Accuracy, speed, and ID switch statistics (if implemented)

---

## Example Usage in Python
```python
from main import PlayerReIDSystem

system = PlayerReIDSystem(yolo_weights='best.pt')
system.process_video('input.mp4', 'output.mp4')
```

---

## Notes
- Designed for single-camera, short sports clips
- Optimized for reliability and ID consistency
- Easily extensible for more advanced features
- **Tip:** If you notice that some players are not being tracked (missed detections), you can reduce the detection confidence threshold in the code (usually a parameter in the YOLO detection function). Lowering this threshold may help detect more players, but could also increase false positives. Adjust according to your needs.

---

## License
MIT 