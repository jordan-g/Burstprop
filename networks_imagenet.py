import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers_imagenet import *

class ImageNetConvNet(nn.Module):
    def __init__(self, input_channels, p_baseline, weight_fa_std, weight_fa_learning, kappa):
        super(ImageNetConvNet, self).__init__()

        self.weight_fa_std = weight_fa_std

        self.conv1   = ConvLayer(input_channels, 48, 1, weight_fa_std, weight_fa_learning, kernel_size=9, stride=4)
        self.conv2   = ConvLayer(48, 48, 1, weight_fa_std, weight_fa_learning, kernel_size=3, stride=2)
        self.conv3   = ConvLayer(48, 96, 1, weight_fa_std, weight_fa_learning, kernel_size=5, stride=1)
        self.conv4   = ConvLayer(96, 96, 1, weight_fa_std, weight_fa_learning, kernel_size=3, stride=2)
        self.conv5   = ConvLayer(96, 192, 1, weight_fa_std, weight_fa_learning, kernel_size=3, stride=1)
        self.conv6   = ConvLayer(192, 192, 1, weight_fa_std, weight_fa_learning, kernel_size=3, stride=2)
        self.conv7   = ConvLayer(192, 384, kappa, weight_fa_std, weight_fa_learning, kernel_size=3, stride=1)
        self.flatten = Flatten()
        self.fc1     = OutputLayer(384, 1000, p_baseline, weight_fa_std, weight_fa_learning, kappa)

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.fc_layers   = [self.fc1]

        self._initialize_weights()

    def forward(self, input, target):
        x = [input, input]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.flatten(x)
        x = self.fc1(x, target)

        return x

    def weight_angles(self):
        weight_angles = []

        for i in range(len(self.conv_layers)):
            weight_angles.append((180/math.pi)*torch.acos(F.cosine_similarity(self.conv_layers[i].weight.flatten(), self.conv_layers[i].weight_fa.flatten(), dim=0)))

        for i in range(len(self.fc_layers)):
            weight_angles.append((180/math.pi)*torch.acos(F.cosine_similarity(self.fc_layers[i].weight.flatten(), self.fc_layers[i].weight_fa.flatten(), dim=0)))

        return weight_angles

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ConvLayer):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

                if self.weight_fa_std > 0:
                    nn.init.normal_(m.weight_fa, 0, self.weight_fa_std)
                else:
                    nn.init.kaiming_normal_(m.weight_fa, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, OutputLayer):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

                if self.weight_fa_std > 0:
                    nn.init.normal_(m.weight_fa, 0, self.weight_fa_std)
                else:
                    nn.init.normal_(m.weight_fa, 0, 0.01)

class ImageNetConvNetBP(nn.Module):
    def __init__(self, input_channels):
        super(ImageNetConvNetBP, self).__init__()

        self.feature_layers = []

        self.feature_layers.append(nn.Conv2d(input_channels, 48, kernel_size=9, stride=4))
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.Conv2d(48, 48, kernel_size=3, stride=2))
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.Conv2d(48, 96, kernel_size=5, stride=1))
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.Conv2d(96, 96, kernel_size=3, stride=2))
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.Conv2d(96, 192, kernel_size=3, stride=1))
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.Conv2d(192, 192, kernel_size=3, stride=2))
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.Conv2d(192, 384, kernel_size=3, stride=1))
        self.feature_layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*(self.feature_layers))

        self.classification_layers = []

        self.classification_layers.append(nn.Linear(384, 1000))

        self.classifier = nn.Sequential(*(self.classification_layers))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
