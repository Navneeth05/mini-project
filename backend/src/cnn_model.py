import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = ["Alternaria", "Anthracnose", "Bacterial_Blight", "Healthy"]

class Head(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.net(x)

def build_model(num_classes=4, backbone="efficientnet_b0", dropout=0.3):
    if backbone == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = Head(in_features, num_classes, dropout)
    else:
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = Head(in_features, num_classes, dropout)
    return base
