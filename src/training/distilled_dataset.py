import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from src.embeddings.embedding_utils import PretrainedEmbeddingExtractor

class DistilledFaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the folder containing aligned face images.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        self.transform = transform
        # Initialize the teacher extractor (e.g., DeepFace/ArcFace)
        self.teacher_extractor = PretrainedEmbeddingExtractor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms if provided.
        if self.transform:
            image = self.transform(image)
        else:
            # Convert image to tensor if no transform is given.
            image = transforms.ToTensor()(image)

        # Get teacher embedding for the image.
        teacher_embedding = self.teacher_extractor.extract_arcface_embedding(image_path)
        teacher_embedding = torch.tensor(teacher_embedding, dtype=torch.float32)

        return image, teacher_embedding
