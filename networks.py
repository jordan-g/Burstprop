import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from layers import *

class Flatten(nn.Module):
    def forward(self, x):
        self.input_size = x.size()
        return x.view(self.input_size[0], -1)

    def backward(self, b_input, b_input_t, b_input_bp):
        return b_input.view(self.input_size), b_input_t.view(self.input_size), b_input_bp.view(self.input_size)

class CIFAR10ConvNet(nn.Module):
    def __init__(self, input_channels, p_baseline, weight_fa_std, weight_r_std, weight_fa_learning, recurrent_input, weight_r_learning, device):
        super(CIFAR10ConvNet, self).__init__()

        self.weight_fa_std      = weight_fa_std
        self.weight_r_std       = weight_r_std
        self.weight_fa_learning = weight_fa_learning
        self.recurrent_input    = recurrent_input
        self.weight_r_learning  = weight_r_learning

        self.feature_layers = []

        if self.weight_fa_learning:
            self.feature_layers.append(Conv2dHiddenLayer(input_channels, 64, p_baseline, weight_fa_learning, False, weight_r_learning, 32, device, kernel_size=5, stride=2))
            self.feature_layers.append(Conv2dHiddenLayer(64, 128, p_baseline, weight_fa_learning, False, weight_r_learning, self.feature_layers[0].out_size, device, kernel_size=5, stride=2))
            self.feature_layers.append(Conv2dHiddenLayer(128, 256, p_baseline, weight_fa_learning, False, weight_r_learning, self.feature_layers[1].out_size, device, kernel_size=3))
        else:
            self.feature_layers.append(Conv2dHiddenLayer(input_channels, 64, p_baseline, weight_fa_learning, False, weight_r_learning, 32, device, kernel_size=5, stride=2))
            self.feature_layers.append(Conv2dHiddenLayer(64, 256, p_baseline, weight_fa_learning, False, weight_r_learning, self.feature_layers[0].out_size, device, kernel_size=5, stride=2))
            self.feature_layers.append(Conv2dHiddenLayer(256, 256, p_baseline, weight_fa_learning, False, weight_r_learning, self.feature_layers[1].out_size, device, kernel_size=3))
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if self.weight_fa_learning:
            self.classification_layers.append(HiddenLayer(2304, 1024, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, device))
            self.classification_layers.append(OutputLayer(1024, 10, p_baseline, weight_fa_learning, device))
        else:
            self.classification_layers.append(HiddenLayer(2304, 1480, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, device))
            self.classification_layers.append(OutputLayer(1480, 10, p_baseline, weight_fa_learning, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        feedback, feedback_t, feedback_bp = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers)-2, -1, -1):
            if self.recurrent_input:
                self.classification_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp = self.classification_layers[i].backward(feedback, feedback_t, feedback_bp)
        
        for i in range(len(self.feature_layers)-1, -1, -1):
            if self.recurrent_input and i < len(self.feature_layers)-1:
                self.feature_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp = self.feature_layers[i].backward(feedback, feedback_t, feedback_bp)

    def update_weights(self, lr, momentum=0, weight_decay=0, recurrent_lr=None, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers)-1:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, recurrent_lr=recurrent_lr, batch_size=batch_size)
            else:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

        for i in range(len(self.feature_layers)-1):
            self.feature_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, recurrent_lr=recurrent_lr, batch_size=batch_size)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles(self):
        weight_angles = []

        for i in range(1, len(self.feature_layers)-1):
            weight_angles.append((180/math.pi)*torch.acos(F.cosine_similarity(self.feature_layers[i].weight.flatten(), self.feature_layers[i].weight_fa.flatten(), dim=0)))

        for i in range(len(self.classification_layers)):
            weight_angles.append((180/math.pi)*torch.acos(F.cosine_similarity(self.classification_layers[i].weight.flatten(), self.classification_layers[i].weight_fa.flatten(), dim=0)))

        return weight_angles

    def delta_angles(self):
        delta_angles = []

        for i in range(len(self.feature_layers)-1):
            a = np.mean([ (180/math.pi)*(torch.acos(F.cosine_similarity(self.feature_layers[i].delta[j].flatten(), self.feature_layers[i].delta_bp[j].flatten(), dim=0))).cpu() for j in range(self.feature_layers[i].delta.shape[0]) ])
            delta_angles.append(a)

        for i in range(len(self.classification_layers)-1):
            a = np.mean([ (180/math.pi)*(torch.acos(F.cosine_similarity(self.classification_layers[i].delta[j].flatten(), self.classification_layers[i].delta_bp[j].flatten(), dim=0))).cpu() for j in range(self.classification_layers[i].delta.shape[0]) ])
            delta_angles.append(a)

        return delta_angles

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2dHiddenLayer) or isinstance(m, HiddenLayer) or isinstance(m, OutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

                init.normal_(m.weight_fa, 0, self.weight_fa_std)

                if self.recurrent_input and (isinstance(m, HiddenLayer) or isinstance(m, Conv2dHiddenLayer)):
                    init.normal_(m.weight_r, 0, self.weight_r_std)

class CIFAR10ConvNetNP(nn.Module):
    def __init__(self, input_channels, xi_mean, xi_std, device):
        super(CIFAR10ConvNetNP, self).__init__()

        self.feature_layers = []

        self.feature_layers.append(Conv2dHiddenLayerNP(input_channels, 64, xi_mean, xi_std, 32, device, kernel_size=5, stride=2))
        self.feature_layers.append(Conv2dHiddenLayerNP(64, 128, xi_mean, xi_std, self.feature_layers[0].out_size, device, kernel_size=5, stride=2))
        self.feature_layers.append(Conv2dHiddenLayerNP(128, 256, xi_mean, xi_std, self.feature_layers[1].out_size, device, kernel_size=3))
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        self.classification_layers.append(HiddenLayerNP(2304, 1024,  xi_mean, xi_std, device))
        self.classification_layers.append(OutputLayerNP(1024, 10,  xi_mean, xi_std, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self.l = 0
        self.u = 0

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def forward_perturb(self, x):
        for i in range(len(self.feature_layers)-1):
            x = self.feature_layers[i].forward_perturb(x)

        x = self.feature_layers[-1].forward(x)

        for i in range(len(self.classification_layers)):
            x = self.classification_layers[i].forward_perturb(x)

    def forward_backward_weight_update_perturb(self, x, target, lr, momentum=0, weight_decay=0, perturb_units=False, batch_size=1):
        if perturb_units:
            if self.l < len(self.feature_layers) - 1 + len(self.classification_layers):
                if self.l < len(self.feature_layers) - 1:
                    selected_layer = self.feature_layers[self.l]
                else:
                    selected_layer = self.classification_layers[self.l-(len(self.feature_layers)-1)]

                if self.u < sum(selected_layer.e.shape[1:]):
                    y = x.clone()

                    # forward pass with perturbed layer
                    for i in range(len(self.feature_layers)):
                        if i < len(self.feature_layers)-1:
                            if i == self.l:
                                y = self.feature_layers[i].forward_perturb(y, unit=self.u)
                            else:
                                y = self.feature_layers[i].forward_perturb(y, perturb_layer=False)
                        else:
                            y = self.feature_layers[i].forward(y)

                    for i in range(len(self.classification_layers)):
                        if i == self.l-(len(self.feature_layers)-1):
                            y = self.classification_layers[i].forward_perturb(y, unit=self.u)
                        else:
                            y = self.classification_layers[i].forward_perturb(y, perturb_layer=False)

                    # backward pass
                    E, E_perturb = self.classification_layers[-1].backward(target)
                    for i in range(len(self.classification_layers)-2, -1, -1):
                        self.classification_layers[i].backward(E, E_perturb)

                    for i in range(len(self.feature_layers)-2, -1, -1):
                        self.feature_layers[i].backward(E, E_perturb)

                    self.update_weights(lr, momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

                    self.u += 1
                else:
                    self.u = 0
                    self.l += 1
            else:
                self.l = 0
                self.u = 0
        else:
            while self.l < len(self.feature_layers) - 1 + len(self.classification_layers):
                if self.l < len(self.feature_layers) - 1:
                    selected_layer = self.feature_layers[self.l]
                else:
                    selected_layer = self.classification_layers[self.l-(len(self.feature_layers)-1)]

                y = x.clone()

                # forward pass with perturbed layer
                for i in range(len(self.feature_layers)):
                    if i < len(self.feature_layers)-1:
                        if i == self.l:
                            y = self.feature_layers[i].forward_perturb(y)
                        else:
                            y = self.feature_layers[i].forward_perturb(y, perturb_layer=False)
                    else:
                        y = self.feature_layers[i].forward(y)

                for i in range(len(self.classification_layers)):
                    if i == self.l-(len(self.feature_layers)-1):
                        y = self.classification_layers[i].forward_perturb(y)
                    else:
                        y = self.classification_layers[i].forward_perturb(y, perturb_layer=False)

                # backward pass
                E, E_perturb = self.classification_layers[-1].backward(target)
                for i in range(len(self.classification_layers)-2, -1, -1):
                    self.classification_layers[i].backward(E, E_perturb)

                for i in range(len(self.feature_layers)-2, -1, -1):
                    self.feature_layers[i].backward(E, E_perturb)

                self.update_weights(lr, momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

                self.l += 1

            self.l = 0

    def backward(self, target):
        E, E_perturb = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers)-2, -1, -1):
            self.classification_layers[i].backward(E, E_perturb)

        for i in range(len(self.feature_layers)-2, -1, -1):
            self.feature_layers[i].backward(E, E_perturb)

    def update_weights(self, lr, momentum=0, weight_decay=0, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers)-1:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)
            else:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

        for i in range(len(self.feature_layers)-1):
            self.feature_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2dHiddenLayerNP) or isinstance(m, HiddenLayerNP) or isinstance(m, OutputLayerNP):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

class CIFAR10ConvNetBP(nn.Module):
    def __init__(self, input_channels):
        super(CIFAR10ConvNetBP, self).__init__()

        self.feature_layers = []

        self.feature_layers.append(nn.Conv2d(input_channels, 64, kernel_size=5, stride=2))
        self.feature_layers.append(nn.Sigmoid())
        self.feature_layers.append(nn.Conv2d(64, 128, kernel_size=5, stride=2))
        self.feature_layers.append(nn.Sigmoid())
        self.feature_layers.append(nn.Conv2d(128, 256, kernel_size=3))
        self.feature_layers.append(nn.Sigmoid())
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        self.classification_layers.append(nn.Linear(2304, 1024))
        self.classification_layers.append(nn.Sigmoid())
        self.classification_layers.append(nn.Linear(1024, 10))
        self.classification_layers.append(nn.Sigmoid())

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

class MNISTNet(nn.Module):
    def __init__(self, input_channels, p_baseline, weight_fa_std, weight_r_std, weight_fa_learning, recurrent_input, weight_r_learning, n_hidden_layers, device):
        super(MNISTNet, self).__init__()

        self.weight_fa_std      = weight_fa_std
        self.weight_r_std       = weight_r_std
        self.weight_fa_learning = weight_fa_learning
        self.recurrent_input    = recurrent_input
        self.weight_r_learning  = weight_r_learning

        self.feature_layers = []

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(OutputLayer(784, 10, p_baseline, weight_fa_learning, device))
        else:
            self.classification_layers.append(HiddenLayer(784, 500, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, device))

            for i in range(1, n_hidden_layers):
                self.classification_layers.append(HiddenLayer(500, 500, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, device))
            
            self.classification_layers.append(OutputLayer(500, 10, p_baseline, weight_fa_learning, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        feedback, feedback_t, feedback_bp = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers)-2, -1, -1):
            if self.recurrent_input:
                self.classification_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp = self.classification_layers[i].backward(feedback, feedback_t, feedback_bp)

    def update_weights(self, lr, momentum=0, weight_decay=0, recurrent_lr=None, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers)-1:
                self.classification_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, recurrent_lr=recurrent_lr, batch_size=batch_size)
            else:
                self.classification_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles(self):
        weight_angles = []

        for i in range(1, len(self.classification_layers)):
            weight_angles.append((180/math.pi)*torch.acos(F.cosine_similarity(self.classification_layers[i].weight.flatten(), self.classification_layers[i].weight_fa.flatten(), dim=0)))

        return weight_angles

    def delta_angles(self):
        delta_angles = []

        for i in range(len(self.classification_layers)-1):
            a = np.mean([ (180/math.pi)*(torch.acos(F.cosine_similarity(self.classification_layers[i].delta[j].flatten(), self.classification_layers[i].delta_bp[j].flatten(), dim=0))).cpu() for j in range(self.classification_layers[i].delta.shape[0]) ])
            delta_angles.append(a)

        return delta_angles

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, HiddenLayer) or isinstance(m, OutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

                init.normal_(m.weight_fa, 0, self.weight_fa_std)

                if self.recurrent_input and isinstance(m, HiddenLayer):
                    init.normal_(m.weight_r, 0, self.weight_r_std)

class MNISTNetNP(nn.Module):
    def __init__(self, input_channels, n_hidden_layers, xi_mean, xi_std, device):
        super(MNISTNetNP, self).__init__()

        self.feature_layers = []

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(OutputLayerNP(784, 10, xi_mean, xi_std, device))
        else:
            self.classification_layers.append(HiddenLayerNP(784, 500, xi_mean, xi_std, device))

            for i in range(1, n_hidden_layers):
                self.classification_layers.append(HiddenLayerNP(500, 500, xi_mean, xi_std, device))
            
            self.classification_layers.append(OutputLayerNP(500, 10, xi_mean, xi_std, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self.l = 0
        self.u = 0

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def forward_perturb(self, x):
        for i in range(len(self.feature_layers)):
            x = self.feature_layers[i].forward(x)

        for i in range(len(self.classification_layers)):
            x = self.classification_layers[i].forward_perturb(x)

    def forward_backward_weight_update_perturb(self, x, target, lr, momentum=0, weight_decay=0, perturb_units=False, batch_size=1):
        if perturb_units:
            if self.l < len(self.classification_layers):
                selected_layer = self.classification_layers[self.l]

                if self.u < sum(selected_layer.e.shape[1:]):
                    y = x.clone()

                    for i in range(len(self.feature_layers)):
                        y = self.feature_layers[i].forward(y)

                    for i in range(len(self.classification_layers)):
                        if i == self.l:
                            y = self.classification_layers[i].forward_perturb(y, unit=self.u)
                        else:
                            y = self.classification_layers[i].forward_perturb(y, perturb_layer=False)

                    # backward pass
                    E, E_perturb = self.classification_layers[-1].backward(target)
                    for i in range(len(self.classification_layers)-2, -1, -1):
                        self.classification_layers[i].backward(E, E_perturb)

                    self.update_weights(lr, momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

                    self.u += 1
                else:
                    self.u = 0
                    self.l += 1
            else:
                self.l = 0
                self.u = 0
        else:
            while self.l < len(self.classification_layers):
                selected_layer = self.classification_layers[self.l]

                y = x.clone()

                # forward pass with perturbed layer
                for i in range(len(self.feature_layers)):
                    y = self.feature_layers[i].forward(y)

                for i in range(len(self.classification_layers)):
                    if i == self.l:
                        y = self.classification_layers[i].forward_perturb(y)
                    else:
                        y = self.classification_layers[i].forward_perturb(y, perturb_layer=False)

                # backward pass
                E, E_perturb = self.classification_layers[-1].backward(target)
                for i in range(len(self.classification_layers)-2, -1, -1):
                    self.classification_layers[i].backward(E, E_perturb)

                self.update_weights(lr, momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

                self.l += 1

            self.l = 0

    def backward(self, target):
        E, E_perturb = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers)-2, -1, -1):
            self.classification_layers[i].backward(E, E_perturb)

    def update_weights(self, lr, momentum=0, weight_decay=0, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers)-1:
                self.classification_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)
            else:
                self.classification_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, HiddenLayerNP) or isinstance(m, OutputLayerNP):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

class MNISTNetBP(nn.Module):
    def __init__(self, input_channels, n_hidden_layers):
        super(MNISTNetBP, self).__init__()

        self.feature_layers = []

        self.feature_layers.append(Flatten())

        self.classification_layers = []

        if n_hidden_layers == 0:
            self.classification_layers.append(nn.Linear(784, 10))
            self.classification_layers.append(nn.Sigmoid())
        else:
            self.classification_layers.append(nn.Linear(784, 500))
            self.classification_layers.append(nn.Sigmoid())

            for i in range(1, n_hidden_layers):
                self.classification_layers.append(nn.Linear(500, 500))
                self.classification_layers.append(nn.Sigmoid())
            
            self.classification_layers.append(nn.Linear(500, 10))
            self.classification_layers.append(nn.Sigmoid())

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

class MNISTConvNet(nn.Module):
    def __init__(self, input_channels, p_baseline, weight_fa_std, weight_r_std, weight_fa_learning, recurrent_input, weight_r_learning, device):
        super(MNISTConvNet, self).__init__()

        self.weight_fa_std      = weight_fa_std
        self.weight_r_std       = weight_r_std
        self.weight_fa_learning = weight_fa_learning
        self.recurrent_input    = recurrent_input
        self.weight_r_learning  = weight_r_learning

        self.feature_layers = []

        self.feature_layers.append(Conv2dHiddenLayer(input_channels, 8, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, 28, device, kernel_size=4, stride=2))
        self.feature_layers.append(Conv2dHiddenLayer(8, 16, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, self.feature_layers[0].out_size, device, kernel_size=3, stride=2))
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        self.classification_layers.append(HiddenLayer(576, 500, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, device))
        self.classification_layers.append(HiddenLayer(500, 500, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, device))
        self.classification_layers.append(OutputLayer(500, 10, p_baseline, weight_fa_learning, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def backward(self, target):
        feedback, feedback_t, feedback_bp = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers)-2, -1, -1):
            if self.recurrent_input:
                self.classification_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp = self.classification_layers[i].backward(feedback, feedback_t, feedback_bp)
        
        for i in range(len(self.feature_layers)-1, -1, -1):
            if self.recurrent_input and i < len(self.feature_layers)-1:
                self.feature_layers[i].backward_pre(feedback)
            feedback, feedback_t, feedback_bp = self.feature_layers[i].backward(feedback, feedback_t, feedback_bp)

    def update_weights(self, lr, momentum=0, weight_decay=0, recurrent_lr=None, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers)-1:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, recurrent_lr=recurrent_lr, batch_size=batch_size)
            else:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

        for i in range(len(self.feature_layers)-1):
            self.feature_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, recurrent_lr=recurrent_lr, batch_size=batch_size)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def weight_angles(self):
        weight_angles = []

        for i in range(1, len(self.feature_layers)-1):
            weight_angles.append((180/math.pi)*torch.acos(F.cosine_similarity(self.feature_layers[i].weight.flatten(), self.feature_layers[i].weight_fa.flatten(), dim=0)))

        for i in range(len(self.classification_layers)):
            weight_angles.append((180/math.pi)*torch.acos(F.cosine_similarity(self.classification_layers[i].weight.flatten(), self.classification_layers[i].weight_fa.flatten(), dim=0)))

        return weight_angles

    def delta_angles(self):
        delta_angles = []

        for i in range(len(self.feature_layers)-1):
            a = np.mean([ (180/math.pi)*(torch.acos(F.cosine_similarity(self.feature_layers[i].delta[j].flatten(), self.feature_layers[i].delta_bp[j].flatten(), dim=0))).cpu() for j in range(self.feature_layers[i].delta.shape[0]) ])
            delta_angles.append(a)

        for i in range(len(self.classification_layers)-1):
            a = np.mean([ (180/math.pi)*(torch.acos(F.cosine_similarity(self.classification_layers[i].delta[j].flatten(), self.classification_layers[i].delta_bp[j].flatten(), dim=0))).cpu() for j in range(self.classification_layers[i].delta.shape[0]) ])
            delta_angles.append(a)

        return delta_angles

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2dHiddenLayer) or isinstance(m, HiddenLayer) or isinstance(m, OutputLayer):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

                init.normal_(m.weight_fa, 0, self.weight_fa_std)

                if self.recurrent_input and (isinstance(m, HiddenLayer) or isinstance(m, Conv2dHiddenLayer)):
                    init.normal_(m.weight_r, 0, self.weight_r_std)

class MNISTConvNetNP(nn.Module):
    def __init__(self, input_channels, xi_mean, xi_std, device):
        super(MNISTConvNetNP, self).__init__()

        self.feature_layers = []

        self.feature_layers.append(Conv2dHiddenLayerNP(input_channels, 8, xi_mean, xi_std, 28, device, kernel_size=4, stride=2))
        self.feature_layers.append(Conv2dHiddenLayerNP(8, 16, xi_mean, xi_std, self.feature_layers[0].out_size, device, kernel_size=3, stride=2))
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        self.classification_layers.append(HiddenLayerNP(576, 500, xi_mean, xi_std, device))
        self.classification_layers.append(HiddenLayerNP(500, 500, xi_mean, xi_std, device))
        self.classification_layers.append(OutputLayerNP(500, 10, xi_mean, xi_std, device))

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self.l = 0
        self.u = 0

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def forward_perturb(self, x):
        for i in range(len(self.feature_layers)-1):
            x = self.feature_layers[i].forward_perturb(x)

        x = self.feature_layers[-1].forward(x)

        for i in range(len(self.classification_layers)):
            x = self.classification_layers[i].forward_perturb(x)

    def forward_backward_weight_update_perturb(self, x, target, lr, momentum=0, weight_decay=0, perturb_units=False, batch_size=1):
        if perturb_units:
            if self.l < len(self.feature_layers) - 1 + len(self.classification_layers):
                if self.l < len(self.feature_layers) - 1:
                    selected_layer = self.feature_layers[self.l]
                else:
                    selected_layer = self.classification_layers[self.l-(len(self.feature_layers)-1)]

                if self.u < sum(selected_layer.e.shape[1:]):
                    y = x.clone()

                    # forward pass with perturbed layer
                    for i in range(len(self.feature_layers)):
                        if i < len(self.feature_layers)-1:
                            if i == self.l:
                                y = self.feature_layers[i].forward_perturb(y, unit=self.u)
                            else:
                                y = self.feature_layers[i].forward_perturb(y, perturb_layer=False)
                        else:
                            y = self.feature_layers[i].forward(y)

                    for i in range(len(self.classification_layers)):
                        if i == self.l-(len(self.feature_layers)-1):
                            y = self.classification_layers[i].forward_perturb(y, unit=self.u)
                        else:
                            y = self.classification_layers[i].forward_perturb(y, perturb_layer=False)

                    # backward pass
                    E, E_perturb = self.classification_layers[-1].backward(target)
                    for i in range(len(self.classification_layers)-2, -1, -1):
                        self.classification_layers[i].backward(E, E_perturb)

                    for i in range(len(self.feature_layers)-2, -1, -1):
                        self.feature_layers[i].backward(E, E_perturb)

                    self.update_weights(lr, momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

                    self.u += 1
                else:
                    self.u = 0
                    self.l += 1
            else:
                self.l = 0
                self.u = 0
        else:
            while self.l < len(self.feature_layers) - 1 + len(self.classification_layers):
                if self.l < len(self.feature_layers) - 1:
                    selected_layer = self.feature_layers[self.l]
                else:
                    selected_layer = self.classification_layers[self.l-(len(self.feature_layers)-1)]

                y = x.clone()

                # forward pass with perturbed layer
                for i in range(len(self.feature_layers)):
                    if i < len(self.feature_layers)-1:
                        if i == self.l:
                            y = self.feature_layers[i].forward_perturb(y)
                        else:
                            y = self.feature_layers[i].forward_perturb(y, perturb_layer=False)
                    else:
                        y = self.feature_layers[i].forward(y)

                for i in range(len(self.classification_layers)):
                    if i == self.l-(len(self.feature_layers)-1):
                        y = self.classification_layers[i].forward_perturb(y)
                    else:
                        y = self.classification_layers[i].forward_perturb(y, perturb_layer=False)

                # backward pass
                E, E_perturb = self.classification_layers[-1].backward(target)
                for i in range(len(self.classification_layers)-2, -1, -1):
                    self.classification_layers[i].backward(E, E_perturb)

                for i in range(len(self.feature_layers)-2, -1, -1):
                    self.feature_layers[i].backward(E, E_perturb)

                self.update_weights(lr, momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

                self.l += 1

            self.l = 0

    def backward(self, target):
        E, E_perturb = self.classification_layers[-1].backward(target)
        for i in range(len(self.classification_layers)-2, -1, -1):
            self.classification_layers[i].backward(E, E_perturb)

        for i in range(len(self.feature_layers)-2, -1, -1):
            self.feature_layers[i].backward(E, E_perturb)

    def update_weights(self, lr, momentum=0, weight_decay=0, batch_size=1):
        for i in range(len(self.classification_layers)):
            if i < len(self.classification_layers)-1:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)
            else:
                self.classification_layers[i].update_weights(lr=lr[len(self.feature_layers)-1+i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

        for i in range(len(self.feature_layers)-1):
            self.feature_layers[i].update_weights(lr=lr[i], momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

    def loss(self, output, target):
        return F.mse_loss(output, target)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2dHiddenLayerNP) or isinstance(m, HiddenLayerNP) or isinstance(m, OutputLayerNP):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)

class MNISTConvNetBP(nn.Module):
    def __init__(self, input_channels):
        super(MNISTConvNetBP, self).__init__()

        self.feature_layers = []

        self.feature_layers.append(nn.Conv2d(input_channels, 8, kernel_size=4, stride=2))
        self.feature_layers.append(nn.Sigmoid())
        self.feature_layers.append(nn.Conv2d(8, 16, kernel_size=3, stride=2))
        self.feature_layers.append(nn.Sigmoid())
        self.feature_layers.append(Flatten())

        self.classification_layers = []

        self.classification_layers.append(nn.Linear(576, 500))
        self.classification_layers.append(nn.Sigmoid())
        self.classification_layers.append(nn.Linear(500, 500))
        self.classification_layers.append(nn.Sigmoid())
        self.classification_layers.append(nn.Linear(500, 10))
        self.classification_layers.append(nn.Sigmoid())

        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

        self._initialize_weights()

    def forward(self, x):
        return self.out(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=3.6)
                nn.init.constant_(m.bias, 0)
