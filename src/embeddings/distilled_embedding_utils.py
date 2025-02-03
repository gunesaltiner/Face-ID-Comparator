# distilled_embedding_utils.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from src.training.student_model import StudentModel  # Ensure this file is accessible

class DistilledEmbeddingExtractor:
    def __init__(self, model_path="models/distilled_model.pth", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the student model and load the trained weights.
        self.model = StudentModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Define the transforms identical to those used during training.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def extract_embedding(self, aligned_face_path: str) -> np.ndarray:
        # Load image using OpenCV.
        image = cv2.imread(aligned_face_path)
        if image is None:
            raise ValueError(f"Failed to load image: {aligned_face_path}")
        # Convert from BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply the transform.
        image_tensor = self.transform(image)
        # Add a batch dimension.
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        # Inference: compute the embedding.
        with torch.no_grad():
            embedding = self.model(image_tensor)
        # Convert embedding to a NumPy array.
        embedding = embedding.cpu().numpy().flatten()
        return embedding
