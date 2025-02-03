import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.training.distilled_dataset import DistilledFaceDataset
from src.training.student_model import StudentModel

def train_student_model(student, dataloader, criterion, optimizer, device):
    student.train()
    running_loss = 0.0
    for images, teacher_embeddings in dataloader:
        images = images.to(device)
        teacher_embeddings = teacher_embeddings.to(device)
        
        optimizer.zero_grad()
        student_embeddings = student(images)
        loss = criterion(student_embeddings, teacher_embeddings)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the transforms for the images.
    # MobileNetV2 expects images resized to 224x224 with a specific normalization.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Path to the folder with aligned face images.
    image_dir = "processed"
    
    # Create dataset and dataloader.
    dataset = DistilledFaceDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize the student model.
    student = StudentModel().to(device)
    
    # Define loss function and optimizer.
    criterion = nn.MSELoss()  # Minimize the difference between teacher and student embeddings.
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    
    num_epochs = 10  # Adjust based on your dataset and convergence behavior.
    for epoch in range(num_epochs):
        epoch_loss = train_student_model(student, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Save the distilled (student) model.
    model_path = "distilled_model.pth"
    torch.save(student.state_dict(), model_path)
    print(f"Saved distilled model to {model_path}")

if __name__ == "__main__":
    main()
