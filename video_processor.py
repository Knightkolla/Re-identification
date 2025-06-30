import cv2
import logging

class VideoProcessor:
    def __init__(self, video_path, output_path=None, fps=None):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.fps = fps
        self.frame_width = None
        self.frame_height = None
        self.logger = logging.getLogger('VideoProcessor')

    def open(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.logger.error(f"Cannot open video: {self.video_path}")
            raise IOError(f"Cannot open video: {self.video_path}")
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.fps or self.cap.get(cv2.CAP_PROP_FPS)

    def setup_writer(self):
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def read_frame(self):
        if not self.cap:
            self.open()
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def write_frame(self, frame):
        if self.writer is None:
            self.setup_writer()
        if self.writer:
            self.writer.write(frame)

    def release(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()

    def get_total_frames(self):
        if not self.cap:
            self.open()
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) 