import cv2
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis

class PretrainedEmbeddingExtractor:
    def __init__(self):
        self.deepface_model = "ArcFace"

    def extract_arcface_embedding(self, aligned_face_path: str) -> np.ndarray:
        try:
            image = cv2.imread(aligned_face_path)
            if image is None:
                raise ValueError(f"Failed to load image: {aligned_face_path}")

            embedding = DeepFace.represent(
                img_path=image,
                model_name=self.deepface_model,
                enforce_detection=False  # Skip face detection (already done)
            )[0]["embedding"]

            return np.array(embedding)
        
        except Exception as e:
            print(f"DeepFace error: {e}")
            return np.zeros(512)