import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torch.nn.parallel.scatter_gather import gather
import numpy as np

from torch.utils.cpp_extension import load
cudnn_convolution = load(name="cudnn_convolution", sources=["cudnn_convolution.cpp"], verbose=True)

class OutputNeuron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, _, target, weight, weight_fa, bias, p_baseline, weight_fa_learning, kappa):
        e = F.linear(input, weight, bias)

        ctx.save_for_backward(input, target, e, weight, weight_fa)

        ctx.p_baseline         = p_baseline
        ctx.weight_fa_learning = weight_fa_learning
        ctx.kappa              = kappa

        return e

    @staticmethod
    def backward(ctx, grad):
        input, target, e, weight, weight_fa = ctx.saved_tensors

        p_baseline         = ctx.p_baseline
        weight_fa_learning = ctx.weight_fa_learning
        kappa              = ctx.kappa

        a = F.softmax(e, dim=1)

        kappa = 1e-5

        b = p_baseline*(torch.ones_like(a))*a
        b_t = torch.min(p_baseline*(kappa*(target - a)/(a + 1e-8) + 1), torch.ones_like(a))*a

        delta = -(1/kappa)*(b_t - b)

        grad_weight = delta.t().mm(input)
        grad_bias   = delta.sum(0).squeeze(0)

        delta_fb = b.clone()
        delta_fb_t = b_t.clone()

        grad_input   = delta_fb.mm(weight_fa)
        grad_input_t = delta_fb_t.mm(weight_fa)

        if weight_fa_learning:
            grad_weight_fa = delta.t().mm(input)
        else:
            grad_weight_fa = None

        grad_target = None

        return grad_input, grad_input_t, grad_target, grad_weight, grad_weight_fa, grad_bias, None, None, None

class OutputLayer(nn.Module):
    def __init__(self, in_features, out_features, p_baseline, weight_fa_gain, weight_fa_learning, kappa):
        super(OutputLayer, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.p_baseline         = p_baseline
        self.weight_fa_learning = weight_fa_learning
        self.kappa              = kappa

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias   = nn.Parameter(torch.Tensor(out_features))

        self.weight_fa = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=weight_fa_learning)

    def forward(self, input, target):
        return OutputNeuron.apply(input[0], input[1], target, self.weight, self.weight_fa, self.bias, self.p_baseline, self.weight_fa_learning, self.kappa)

class HiddenNeuron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, _, weight, weight_fa, bias, weight_fa_learning, kappa):
        e = F.relu(F.linear(input, weight, bias))

        ctx.save_for_backward(input, e, weight, weight_fa)

        ctx.weight_fa_learning = weight_fa_learning
        ctx.kappa              = kappa

        return e, e.clone()

    @staticmethod
    def backward(ctx, grad, grad_t):
        input, e, weight, weight_fa = ctx.saved_tensors

        weight_fa_learning = ctx.weight_fa_learning
        kappa              = ctx.kappa

        p = torch.sigmoid(grad*(e > 0).float()/(e + 1e-8))
        if kappa != 1:
            p_t = torch.sigmoid((((1/kappa)*(grad_t - grad) + grad)*(e > 0).float()/(e + 1e-8)))
        else:
            p_t = torch.sigmoid(grad_t*(e > 0).float()/(e + 1e-8))

        b   = p*e
        b_t = p_t*e

        delta = -(b_t - b)

        grad_weight = delta.t().mm(input)
        grad_bias   = delta.sum(0).squeeze(0)

        delta_fb = b.clone()
        delta_fb_t = b_t.clone()

        grad_input   = delta_fb.mm(weight_fa)
        grad_input_t = delta_fb_t.mm(weight_fa)

        if weight_fa_learning:
            grad_weight_fa = delta.t().mm(input)
        else:
            grad_weight_fa = None

        return grad_input, grad_input_t, grad_weight, grad_weight_fa, grad_bias, None, None,

class HiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, p_baseline, weight_fa_gain, weight_fa_learning, kappa):
        super(HiddenLayer, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.p_baseline         = p_baseline
        self.weight_fa_learning = weight_fa_learning
        self.kappa              = kappa

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias   = nn.Parameter(torch.Tensor(out_features))

        self.weight_fa = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=weight_fa_learning)

    def forward(self, input):
        return HiddenNeuron.apply(input[0], input[1], self.weight, self.weight_fa, self.bias, self.p_baseline, self.weight_fa_learning, self.kappa)

class ConvNeuron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, _, weight, weight_fa, bias, stride, padding, weight_fa_learning, kappa):
        e = F.relu(cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), 1, False, False))

        ctx.save_for_backward(input, e, weight, weight_fa, bias)

        ctx.stride             = stride
        ctx.padding            = padding
        ctx.weight_fa_learning = weight_fa_learning
        ctx.kappa              = kappa

        return e, e.clone()

    @staticmethod
    def backward(ctx, grad, grad_t):
        input, e, weight, weight_fa, bias = ctx.saved_tensors

        stride             = ctx.stride
        padding            = ctx.padding
        weight_fa_learning = ctx.weight_fa_learning
        kappa              = ctx.kappa

        p = torch.sigmoid(grad*(e > 0).float()/(e + 1e-8))
        if kappa != 1:
            p_t = torch.sigmoid((((1/kappa)*(grad_t - grad) + grad)*(e > 0).float()/(e + 1e-8)))
        else:
            p_t = torch.sigmoid(grad_t*(e > 0).float()/(e + 1e-8))

        b   = p*e
        b_t = p_t*e

        delta = -(b_t - b)

        grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, delta, stride, padding, (1, 1), 1, False, False)
        grad_bias   = torch.sum(delta, dim=[0, 2, 3]).squeeze(0)

        delta_fb   = b.clone()
        delta_fb_t = b_t.clone()

        grad_input   = cudnn_convolution.convolution_backward_input(input.shape, weight_fa.data, delta_fb, stride, padding, (1, 1), 1, False, False)
        grad_input_t = cudnn_convolution.convolution_backward_input(input.shape, weight_fa.data, delta_fb_t, stride, padding, (1, 1), 1, False, False)
            
        if weight_fa_learning:
            grad_weight_fa = cudnn_convolution.convolution_backward_weight(input, weight.shape, delta, stride, padding, (1, 1), 1, False, False)
        else:
            grad_weight_fa = None

        return grad_input, grad_input_t, grad_weight, grad_weight_fa, grad_bias, None, None, None, None

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kappa, weight_fa_gain, weight_fa_learning, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()

        kernel_size = _pair(kernel_size)
        stride      = _pair(stride)
        padding     = _pair(padding)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        self.weight_fa_learning = weight_fa_learning
        self.kappa              = kappa

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias   = nn.Parameter(torch.Tensor(out_channels))

        self.weight_fa = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size), requires_grad=weight_fa_learning)

    def forward(self, input):
        return ConvNeuron.apply(input[0], input[1], self.weight, self.weight_fa, self.bias, self.stride, self.padding, self.weight_fa_learning, self.kappa)

class Flatten(nn.Module):
    def forward(self, x):
        return x[0].view(x[0].shape[0], -1), x[1].view(x[1].shape[0], -1)

class BPFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
