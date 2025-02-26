import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights


class ResNet50variant(nn.Module):
    def __init__(self):
        super(ResNet50variant, self).__init__()

        # Load the pre-trained ResNet-50 model
        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        state_dict = self.resnet50.state_dict()
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)

        # Modify the input layer to work with grayscale images
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet50.load_state_dict(state_dict)

        # The original fully connected layer has been removed
        self.resnet50.avgpool = nn.Identity()
        self.resnet50.fc = nn.Identity()  # Use Identity to maintain the structure

        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Define new layers
        self.conv = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=2, stride=1, padding=0)
        self.fc = nn.Linear(73728, 2)

    def forward(self, x):
        # Pass through the pre-trained model up to the layer before the FC layer

        x = self.resnet50(x)  # This now returns a tensor of feature maps

        x = x.reshape(x.size(0), 2048, 7, 7)

        # Pass through the new layers
        x = self.conv(x)  # Convolution
        x = torch.flatten(x, 1)  # Flatten
        x = self.fc(x)  # Fully Connected Layer

        return x


class ResNet18variant(nn.Module):
    def __init__(self):
        super(ResNet18variant, self).__init__()

        # Load the pre-trained ResNet-18 model
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        state_dict = self.resnet18.state_dict()
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)

        # Modify the input layer to work with grayscale images
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet18.load_state_dict(state_dict)

        # The original fully connected layer has been removed
        self.resnet18.avgpool = nn.Identity()
        self.resnet18.fc = nn.Identity()  # Use Identity to maintain the structure

        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Define new layers
        self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.fc = nn.Linear(18432, 2)

    def forward(self, x):
        # Pass through the pre-trained model up to the layer before the FC layer

        x = self.resnet18(x)  # This now returns a tensor of feature maps

        x = x.reshape(x.size(0), 512, 7, 7)

        # Pass through the new layers
        x = self.conv(x)  # Convolution
        x = torch.flatten(x, 1)  # Flatten
        x = self.fc(x)  # Fully Connected Layer

        return x
