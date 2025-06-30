import cv2
import numpy as np
import logging
from skimage.feature import local_binary_pattern
import torch
import torchvision.transforms as T
import torchreid

class FeatureExtractor:
    def __init__(self, hist_bins=16, device='cpu'):
        self.hist_bins = hist_bins
        self.logger = logging.getLogger('FeatureExtractor')
        self.device = device
        # Load OSNet model from torchreid
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_deep_features(self, crop):
        if crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = crop
        img = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img)
        return feat.cpu().numpy().flatten()

    def extract_features(self, frame, bbox, prev_bbox=None, prev_features=None, context_bboxes=None):
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return None # Invalid bbox
        crop = frame[y1:y2, x1:x2]
        features = {}

        # Deep Features (OSNet)
        features['deep_feature'] = self._extract_deep_features(crop)
        
        # Visual: Color histogram (HSV)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [self.hist_bins]*3, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features['color_hist'] = hist
        # Visual: Texture (LBP)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        lbp_hist = lbp_hist.astype('float') / (lbp_hist.sum() + 1e-6)
        features['texture'] = lbp_hist
        # Spatial: Position, size, aspect ratio
        features['position'] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        features['size'] = np.array([x2 - x1, y2 - y1])
        features['aspect_ratio'] = (x2 - x1) / (y2 - y1 + 1e-6)
        # Temporal: Velocity vector
        if prev_bbox is not None:
            prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2])
            curr_center = features['position']
            features['velocity'] = curr_center - prev_center
        else:
            features['velocity'] = np.zeros(2)
        # Contextual: Relative positions to other players
        if context_bboxes:
            rel_positions = []
            for ctx_bbox in context_bboxes:
                ctx_center = np.array([(ctx_bbox[0] + ctx_bbox[2]) / 2, (ctx_bbox[1] + ctx_bbox[3]) / 2])
                rel_positions.append(ctx_center - features['position'])
            features['context'] = np.array(rel_positions).flatten()
        else:
            features['context'] = np.zeros(0)
        return features 