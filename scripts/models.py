import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(name: str, num_classes: int = 10):
    """Return a model by name. Supported: 'simple_cnn', 'resnet18', 'mobilenet_v2'."""
    name = name.lower()
    if name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    if name == 'resnet18':
        model = tv_models.resnet18(pretrained=False)
        # Replace final fc
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == 'mobilenet_v2':
        model = tv_models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model name: {name}")
