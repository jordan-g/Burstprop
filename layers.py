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
import torch.distributions as tdist

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

try:
    from torch.utils.cpp_extension import load
    cudnn_convolution = load(name="cudnn_convolution", sources=["cudnn_convolution.cpp"], verbose=True)
    use_cudnn = True
except:
    use_cudnn = False

class OutputLayer(nn.Module):
    def __init__(self, in_features, out_features, p_baseline, weight_fa_learning, device):
        super(OutputLayer, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.p_baseline         = p_baseline
        self.weight_fa_learning = weight_fa_learning

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias   = torch.Tensor(out_features).to(device)
        
        self.weight_fa = torch.Tensor(out_features, in_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias   = torch.zeros(out_features).to(device)

        self.p = self.p_baseline*torch.ones(self.out_features).to(device)

        if self.weight_fa_learning:
            self.delta_weight_fa = torch.zeros(out_features, in_features).to(device)

    def forward(self, input):
        self.input = input
        self.e = torch.sigmoid(F.linear(input, self.weight, self.bias))
        return self.e

    def backward(self, b_input):
        self.p   = self.p_baseline
        self.p_t = self.p_baseline*((b_input - self.e)*(1 - self.e) + 1)

        self.b   = self.p*self.e
        self.b_t = self.p_t*self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = -(b_input - self.e)*self.e*(1 - self.e)

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input)
        self.grad_bias   = torch.sum(self.delta, dim=0)

        if self.weight_fa_learning:
            delta = -(self.b_t - self.b)
            self.grad_weight_fa = delta.transpose(0, 1).mm(self.input)

        return (self.b).mm(self.weight_fa), (self.b_t).mm(self.weight_fa), self.delta_bp.mm(self.weight)

    def update_weights(self, lr, momentum=0, weight_decay=0, batch_size=1):
        self.delta_weight = -lr*self.grad_weight/batch_size + momentum*self.delta_weight
        self.delta_bias   = -lr*self.grad_bias/batch_size + momentum*self.delta_bias

        self.weight += self.delta_weight - weight_decay*self.weight
        self.bias   += self.delta_bias

        if self.weight_fa_learning:
            self.delta_weight_fa = -lr*self.grad_weight_fa/batch_size + momentum*self.delta_weight_fa

            self.weight_fa += self.delta_weight_fa - weight_decay*self.weight_fa

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class HiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, device):
        super(HiddenLayer, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.p_baseline         = p_baseline
        self.weight_fa_learning = weight_fa_learning
        self.recurrent_input    = recurrent_input
        self.weight_r_learning  = weight_r_learning

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias   = torch.Tensor(out_features).to(device)
        
        self.weight_fa = torch.Tensor(out_features, in_features).to(device)
            
        if self.recurrent_input:
            self.weight_r = torch.Tensor(out_features, out_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias   = torch.zeros(out_features).to(device)

        if self.weight_fa_learning:
            self.delta_weight_fa = torch.zeros(out_features, in_features).to(device)

    def forward(self, input):
        self.input = input
        self.e = torch.sigmoid(F.linear(input, self.weight, self.bias))
        return self.e
        
    def backward_pre(self, b_input):
        p = torch.sigmoid(b_input)
        self.b_pre = p*self.e

    def backward(self, b_input, b_input_t, b_input_bp):
        if self.recurrent_input:
            self.u = b_input*(1 - self.e) - self.b_pre.mm(self.weight_r.transpose(0, 1))
            self.p   = torch.sigmoid(self.u)
            self.p_t = torch.sigmoid((b_input_t*(1 - self.e) - self.b_pre.mm(self.weight_r.transpose(0, 1))))
        else:
            self.p   = torch.sigmoid(b_input*(1 - self.e))
            self.p_t = torch.sigmoid(b_input_t*(1 - self.e))
            
        self.b   = self.p*self.e
        self.b_t = self.p_t*self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = b_input_bp*self.e*(1 - self.e)

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input)
        self.grad_bias   = torch.sum(self.delta, dim=0)

        if self.weight_fa_learning:
            delta               = -(self.b_t - self.b)
            self.grad_weight_fa = delta.transpose(0, 1).mm(self.input)
            
        if self.recurrent_input and self.weight_r_learning:
            self.grad_weight_r = -self.u.transpose(0, 1).mm(self.b_pre)

        return (self.b).mm(self.weight_fa), (self.b_t).mm(self.weight_fa), self.delta_bp.mm(self.weight)

    def update_weights(self, lr, momentum=0, weight_decay=0, recurrent_lr=None, batch_size=1):
        self.delta_weight = -lr*self.grad_weight/batch_size + momentum*self.delta_weight
        self.delta_bias   = -lr*self.grad_bias/batch_size + momentum*self.delta_bias

        self.weight += self.delta_weight - weight_decay*self.weight
        self.bias   += self.delta_bias

        if self.weight_fa_learning:
            self.delta_weight_fa = -lr*self.grad_weight_fa/batch_size + momentum*self.delta_weight_fa

            self.weight_fa += self.delta_weight_fa - weight_decay*self.weight_fa
        
        if self.recurrent_input and self.weight_r_learning:
            self.weight_r += -recurrent_lr*self.grad_weight_r

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class _ConvNdFA(nn.Module):
    """
    Implementation of an N-dimensional convolution module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:

    https://www.nature.com/articles/ncomms13276

    This code is copied from the _ConvNd module in PyTorch, with the addition
    of the random feedback weights.
    """

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, recurrent_input, in_size, device, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNdFA, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).to(device)

            self.weight_fa = torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).to(device)
        else:
            self.weight = torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).to(device)

            self.weight_fa = torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).to(device)

        self.out_size = int((in_size - kernel_size[0] + 2*padding[0])/stride[0] + 1)

        print(in_size, self.out_size)

        if self.recurrent_input:
            self.weight_r = torch.Tensor(self.out_size*self.out_size*self.out_channels, self.out_size*self.out_size*self.out_channels).to(device)
        if bias:
            self.bias = torch.Tensor(out_channels).to(device)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dHiddenLayer(_ConvNdFA):
    def __init__(self, in_channels, out_channels, p_baseline, weight_fa_learning, recurrent_input, weight_r_learning, in_size, device, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        self.weight_fa_learning = weight_fa_learning
        self.recurrent_input    = recurrent_input
        self.weight_r_learning  = weight_r_learning
        
        super(Conv2dHiddenLayer, self).__init__(
            in_channels, out_channels, recurrent_input, in_size, device, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.p_baseline = p_baseline

        self.delta_weight = torch.zeros(self.weight.shape).to(device)
        self.delta_bias   = torch.zeros(self.bias.shape).to(device)

        self.grad_weight = torch.zeros(self.weight.shape).to(device)
        self.grad_bias   = torch.zeros(self.bias.shape).to(device)

        if self.weight_fa_learning:
            self.delta_weight_fa = torch.zeros(self.weight_fa.shape).to(device)

            self.grad_weight_fa = torch.zeros(self.weight_fa.shape).to(device)

    def forward(self, input):
        self.input = input

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            self.input_size = F.pad(input, expanded_padding, mode='circular').size()

            self.e = torch.sigmoid(F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups))

            return self.e

        if use_cudnn:
            self.e = torch.sigmoid(cudnn_convolution.convolution(self.input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, False, False))
        else:
            self.e = torch.sigmoid(F.conv2d(self.input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups))

        return self.e
    
    def backward_pre(self, b_input):
        p = torch.sigmoid(b_input)
        self.b_pre = p*self.e

    def backward(self, b_input, b_input_t, b_input_bp):
        if self.recurrent_input:
            a = self.b_pre.view(-1, self.out_size*self.out_size*self.out_channels).mm(self.weight_r)
            u = b_input*(1 - self.e)
            c = a.view(-1, self.out_channels, self.out_size, self.out_size)
            p   = torch.sigmoid(u - c)
            p_t = torch.sigmoid((b_input_t*(1 - self.e) - c))
        else:
            p   = torch.sigmoid(b_input*(1 - self.e))
            p_t = torch.sigmoid(b_input_t*(1 - self.e))

        self.b   = p*self.e
        self.b_t = p_t*self.e

        self.delta = -(self.b_t - self.b)

        self.delta_bp = b_input_bp*self.e*(1 - self.e)

        if use_cudnn:
            self.grad_weight = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape, self.delta, self.stride, self.padding, self.dilation, self.groups, False, False)
        else:
            self.grad_weight = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta, self.stride, self.padding, self.dilation, self.groups)
        
        self.grad_bias = torch.sum(self.delta_bp, dim=[0, 2, 3])

        if self.recurrent_input and self.weight_fa_learning:
            delta = -(self.b_t - self.b)

            if use_cudnn:
                self.grad_weight_fa = cudnn_convolution.convolution_backward_weight(self.input, self.weight_fa.shape, delta, self.stride, self.padding, self.dilation, self.groups, False, False)
            else:
                self.grad_weight_fa = nn.grad.conv2d_weight(self.input, self.weight_fa.shape, delta, self.stride, self.padding, self.dilation, self.groups)
        
        if self.weight_r_learning:
            self.grad_weight_r = -u.view(-1, self.out_size*self.out_size*self.out_channels).transpose(0, 1).mm(self.b_pre.view(-1, self.out_size*self.out_size*self.out_channels))

        if use_cudnn:
            return cudnn_convolution.convolution_backward_input(self.input.shape, self.weight_fa, self.b, self.stride, self.padding, self.dilation, self.groups, False, False), cudnn_convolution.convolution_backward_input(self.input.shape, self.weight_fa, self.b_t, self.stride, self.padding, self.dilation, self.groups, False, False), cudnn_convolution.convolution_backward_input(self.input.shape, self.weight, self.delta_bp, self.stride, self.padding, self.dilation, self.groups, False, False)
        else:
            return nn.grad.conv2d_input(self.input.shape, self.weight_fa, self.b, self.stride, self.padding, self.dilation, self.groups), nn.grad.conv2d_input(self.input.shape, self.weight_fa, self.b_t, self.stride, self.padding, self.dilation, self.groups), nn.grad.conv2d_input(self.input.shape, self.weight, self.delta_bp, self.stride, self.padding, self.dilation, self.groups)

    def update_weights(self, lr, momentum=0, weight_decay=0, recurrent_lr=None, batch_size=1):
        self.delta_weight = -lr*self.grad_weight/batch_size + momentum*self.delta_weight
        self.delta_bias   = -lr*self.grad_bias/batch_size + momentum*self.delta_bias

        self.weight += self.delta_weight - weight_decay*self.weight
        self.bias   += self.delta_bias

        if self.weight_fa_learning:
            self.delta_weight_fa = -lr*self.grad_weight_fa/batch_size + momentum*self.delta_weight_fa

            self.weight_fa += self.delta_weight_fa - weight_decay*self.weight_fa
        
        if self.recurrent_input and self.weight_r_learning:
            self.weight_r += -recurrent_lr*self.grad_weight_r

class OutputLayerNP(nn.Module):
    def __init__(self, in_features, out_features, xi_mean, xi_std, device):
        super(OutputLayerNP, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.device = device

        self.xi_mean = xi_mean
        self.xi_std = xi_std

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias   = torch.Tensor(out_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias   = torch.zeros(out_features).to(device)

        self.Xi = tdist.Normal(xi_mean, xi_std)

    def forward(self, input):
        self.input = input
        self.y = F.linear(input, self.weight, self.bias)
        self.e = torch.sigmoid(self.y)
        return self.e

    def forward_perturb(self, input, unit=None, perturb_layer=True):
        if perturb_layer:
            if unit is None:
                self.xi = self.Xi.sample(self.e.shape).to(self.device)
            else:
                self.xi = torch.zeros(self.e.shape).to(self.device)
                unravel_unit = unravel_index(unit, self.e.shape[1:])
                self.xi[:, unravel_unit] = self.Xi.sample((self.e.shape[0], 1)).to(self.device)

            self.e_perturb = torch.sigmoid(self.y + self.xi)
        else:
            self.xi = torch.zeros(self.e.shape).to(self.device)
            self.y = F.linear(input, self.weight, self.bias)
            self.e_perturb = torch.sigmoid(self.y)

        return self.e_perturb

    def backward(self, target):
        E = F.mse_loss(self.e, target)
        E_perturb = F.mse_loss(self.e_perturb, target)

        self.delta = -(E - E_perturb)*self.xi/(self.xi_std**2)

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input)
        self.grad_bias   = torch.sum(self.delta, dim=0)

        return E, E_perturb

    def update_weights(self, lr, momentum=0, weight_decay=0, batch_size=1):
        self.delta_weight = -lr*self.grad_weight/batch_size + momentum*self.delta_weight
        self.delta_bias   = -lr*self.grad_bias/batch_size + momentum*self.delta_bias

        self.weight += self.delta_weight - weight_decay*self.weight
        self.bias   += self.delta_bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class HiddenLayerNP(nn.Module):
    def __init__(self, in_features, out_features, xi_mean, xi_std, device):
        super(HiddenLayerNP, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.device = device

        self.xi_mean = xi_mean
        self.xi_std = xi_std

        self.weight = torch.Tensor(out_features, in_features).to(device)
        self.bias   = torch.Tensor(out_features).to(device)

        self.delta_weight = torch.zeros(out_features, in_features).to(device)
        self.delta_bias   = torch.zeros(out_features).to(device)

        self.Xi = tdist.Normal(xi_mean, xi_std)

    def forward(self, input):
        self.input = input
        self.y = F.linear(input, self.weight, self.bias)
        self.e = torch.sigmoid(self.y)
        return self.e

    def forward_perturb(self, input, unit=None, perturb_layer=True):
        if perturb_layer:
            if unit is None:
                self.xi = self.Xi.sample(self.e.shape).to(self.device)
            else:
                self.xi = torch.zeros(self.e.shape).to(self.device)
                unravel_unit = unravel_index(unit, self.e.shape[1:])
                self.xi[:, unravel_unit] = self.Xi.sample((self.e.shape[0], 1)).to(self.device)

            self.e_perturb = torch.sigmoid(self.y + self.xi)
        else:
            self.xi = torch.zeros(self.e.shape).to(self.device)
            self.y = F.linear(input, self.weight, self.bias)
            self.e_perturb = torch.sigmoid(self.y)

        return self.e_perturb

    def backward(self, E, E_perturb):
        self.delta = -(E - E_perturb)*self.xi/(self.xi_std**2)

        self.grad_weight = self.delta.transpose(0, 1).mm(self.input)
        self.grad_bias   = torch.sum(self.delta, dim=0)

    def update_weights(self, lr, momentum=0, weight_decay=0, batch_size=1):
        self.delta_weight = -lr*self.grad_weight/batch_size + momentum*self.delta_weight
        self.delta_bias   = -lr*self.grad_bias/batch_size + momentum*self.delta_bias

        self.weight += self.delta_weight - weight_decay*self.weight
        self.bias   += self.delta_bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class _ConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, in_size, device, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).to(device)
        else:
            self.weight = torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).to(device)

        self.out_size = int((in_size - kernel_size[0] + 2*padding[0])/stride[0] + 1)

        if bias:
            self.bias = torch.Tensor(out_channels).to(device)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dHiddenLayerNP(_ConvNd):
    def __init__(self, in_channels, out_channels, xi_mean, xi_std, in_size, device, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.device = device

        self.xi_mean = xi_mean
        self.xi_std = xi_std
        
        super(Conv2dHiddenLayerNP, self).__init__(
            in_channels, out_channels, in_size, device, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.out_size = int((in_size - kernel_size[0] + 2*padding[0])/stride[0] + 1)

        self.delta_weight = torch.zeros(self.weight.shape).to(device)
        self.delta_bias   = torch.zeros(self.bias.shape).to(device)

        self.grad_weight = torch.zeros(self.weight.shape).to(device)
        self.grad_bias   = torch.zeros(self.bias.shape).to(device)

        self.Xi = tdist.Normal(xi_mean, xi_std)

    def forward(self, input):
        self.input = input

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            self.input_size = F.pad(input, expanded_padding, mode='circular').size()

            self.y = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

            self.e = torch.sigmoid(self.y)

            return self.e

        if use_cudnn:
            self.y = cudnn_convolution.convolution(self.input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, False, False)
        else:
            self.y = F.conv2d(self.input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        self.e = torch.sigmoid(self.y)

        return self.e

    def forward_perturb(self, input, unit=None, perturb_layer=True):
        if perturb_layer:
            if unit is None:
                self.xi = self.Xi.sample(self.e.shape).to(self.device)
            else:
                self.xi = torch.zeros(self.e.shape).to(self.device)
                unravel_unit = unravel_index(unit, self.e.shape[1:])
                self.xi[:, unravel_unit[0], unravel_unit[1], unravel_unit[2]] = self.Xi.sample((self.e.shape[0],)).to(self.device)

            self.e_perturb = torch.sigmoid(self.y + self.xi)
        else:
            self.xi = torch.zeros(self.e.shape).to(self.device)
            self.y = F.conv2d(self.input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
            self.e_perturb = torch.sigmoid(self.y)

        return self.e_perturb

    def backward(self, E, E_perturb):
        self.delta = -(E - E_perturb)*self.xi/(self.xi_std**2)

        if use_cudnn:
            self.grad_weight = cudnn_convolution.convolution_backward_weight(self.input, self.weight.shape, self.delta, self.stride, self.padding, self.dilation, self.groups, False, False)
        else:
            self.grad_weight = nn.grad.conv2d_weight(self.input, self.weight.shape, self.delta, self.stride, self.padding, self.dilation, self.groups)
        
        self.grad_bias = torch.sum(self.delta, dim=[0, 2, 3])

    def update_weights(self, lr, momentum=0, weight_decay=0, batch_size=1):
        self.delta_weight = -lr*self.grad_weight/batch_size + momentum*self.delta_weight
        self.delta_bias   = -lr*self.grad_bias/batch_size + momentum*self.delta_bias

        self.weight += self.delta_weight - weight_decay*self.weight
        self.bias   += self.delta_bias
