import torch.nn as nn
from torchvision import models

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)

        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet.last_channel, 512)
        )

    def forward(self, x):
        return self.mobilenet(x)