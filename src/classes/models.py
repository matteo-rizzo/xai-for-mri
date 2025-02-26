import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights


class ResNetVariant(nn.Module):
    def __init__(self, base_model, conv_in_channels, conv_out_channels, fc_in_features, fc_out_features, weights):
        """
        Generalized ResNet model for grayscale images, with custom convolutional and fully connected layers.

        :param base_model: ResNet model (ResNet50 or ResNet18).
        :param conv_in_channels: Number of input channels for the custom convolution layer.
        :param conv_out_channels: Number of output channels for the custom convolution layer.
        :param fc_in_features: Number of input features for the fully connected layer.
        :param fc_out_features: Number of output features for the fully connected layer.
        :param weights: Pre-trained weights to initialize the model.
        """
        super(ResNetVariant, self).__init__()

        # Load the pre-trained model (either ResNet50 or ResNet18)
        self.model = base_model(weights=weights)

        # Modify the input layer to work with grayscale images
        self.modify_first_layer()

        # Remove the fully connected layers
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()

        # Freeze all layers except the new ones
        for param in self.model.parameters():
            param.requires_grad = False

        # Define the new layers
        self.conv = nn.Conv2d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=2, stride=1,
                              padding=0)
        self.fc = nn.Linear(fc_in_features, fc_out_features)

    def modify_first_layer(self):
        """Modifies the first convolutional layer to accept grayscale input."""
        state_dict = self.model.state_dict()
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        # Pass through the pre-trained ResNet model up to the layer before the FC layer
        x = self.model(x)  # Feature maps from the last block of ResNet

        # Reshape based on model size (either 2048 for ResNet50 or 512 for ResNet18)
        x = x.reshape(x.size(0), x.size(1), 7, 7)

        # Pass through the new layers
        x = self.conv(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.fc(x)

        return x


class ResNet50variant(ResNetVariant):
    def __init__(self):
        # Initialize with ResNet50-specific parameters
        super(ResNet50variant, self).__init__(
            base_model=resnet50,
            conv_in_channels=2048,
            conv_out_channels=2048,
            fc_in_features=73728,
            fc_out_features=2,
            weights=ResNet50_Weights.DEFAULT
        )


class ResNet18variant(ResNetVariant):
    def __init__(self):
        # Initialize with ResNet18-specific parameters
        super(ResNet18variant, self).__init__(
            base_model=resnet18,
            conv_in_channels=512,
            conv_out_channels=512,
            fc_in_features=18432,
            fc_out_features=2,
            weights=ResNet18_Weights.DEFAULT
        )
