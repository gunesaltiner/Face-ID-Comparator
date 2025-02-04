import cv2
import json
import insightface
from typing import List, Dict
import numpy as np


class FaceDetector:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=-1)
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        faces = self.model.get(img=img)

        results = []
        for face in faces:
            x, y, w, h = convert_to_float(face.bbox)
            landmarks = convert_to_float(face.landmark_2d_106)
            det_score = convert_to_float(face.det_score)

            left_eye = landmarks[35]    # Index 94: Left eye center
            right_eye = landmarks[93]   # Index 97: Right eye center
            nose = landmarks[86]        # Index 53: Nose tip
            mouth_left = landmarks[52]  # Index 84: Mouth left corner
            mouth_right = landmarks[61] # Index 90: Mouth right corner


            face_data = {
                "bbox": [x, y, w - x, h - y],
                "det_score": det_score,
                "landmarks": {
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "nose": nose,
                    "mouth_left": mouth_left,
                    "mouth_right": mouth_right
                }
            }
        
             
            results.append(face_data)
        
        return results

def convert_to_float(data):

    if isinstance(data, np.ndarray):
        return data.astype(float).tolist()  # NumPy array to list of floats
    if isinstance(data, (float, np.float32, np.float64)):
        return float(data)  # scalar float values to float
    return [float(val) for val in data]  # list of int or int64 to list of floats