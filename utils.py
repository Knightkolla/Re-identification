import numpy as np
import logging

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def feature_similarity(f1, f2):
    # Weighted sum of similarities for each feature type
    sim_scores = []
    weights = []

    # Deep Features (cosine similarity) - HIGH WEIGHT
    if 'deep_feature' in f1 and 'deep_feature' in f2:
        df1, df2 = f1['deep_feature'], f2['deep_feature']
        sim_scores.append(np.dot(df1, df2) / (np.linalg.norm(df1) * np.linalg.norm(df2) + 1e-6))
        weights.append(0.8) # High priority

    # Color histogram (cosine similarity)
    if 'color_hist' in f1 and 'color_hist' in f2:
        ch1, ch2 = f1['color_hist'], f2['color_hist']
        sim_scores.append(np.dot(ch1, ch2) / (np.linalg.norm(ch1) * np.linalg.norm(ch2) + 1e-6))
        weights.append(0.1)

    # Texture (chi-squared distance)
    if 'texture' in f1 and 'texture' in f2:
        t1, t2 = f1['texture'], f2['texture']
        chi2 = 0.5 * np.sum(((t1 - t2) ** 2) / (t1 + t2 + 1e-6))
        sim_scores.append(1.0 - chi2)
        weights.append(0.1)
    
    # Return weighted average similarity
    if not weights:
        return 0.0
    return np.average(sim_scores, weights=weights)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[logging.StreamHandler()]
    ) 