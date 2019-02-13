import numpy as np

from mnist import mnist
from cifar10 import cifar10

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import pdb

# use CUDA if it is available
cuda = torch.cuda.is_available()

if cuda:
    dtype      = torch.cuda.FloatTensor
    dtype_byte = torch.cuda.ByteTensor
    print("Using CUDA.")
else:
    dtype      = torch.FloatTensor
    dtype_byte = torch.ByteTensor
    print("Not using CUDA.")

def load_dataset(dataset, use_validation):
    if dataset == "mnist":
        # load MNIST data
        x_train, d_train, x_test, d_test = mnist(path="mnist")

        if use_validation:
            x_train = x_train[:, :50000]
            x_test  = x_train[:, 50000:]
            d_train = d_train[:, :50000]
            d_test  = d_train[:, 50000:]

        x_train = torch.from_numpy(x_train.T).type(dtype)
        d_train = torch.from_numpy(d_train.T).type(dtype)
        x_test  = torch.from_numpy(x_test.T).type(dtype)
        d_test  = torch.from_numpy(d_test.T).type(dtype)
    elif dataset == "cifar10":
        # load CIFAR-10 data
        x_train, d_train, x_test, d_test = cifar10(path="cifar10")

        if use_validation:
            x_train = x_train[:40000]
            x_test  = x_train[40000:]
            d_train = d_train[:40000]
            d_test  = d_train[40000:]

        x_train = torch.from_numpy(x_train.T).type(dtype)
        d_train = torch.from_numpy(d_train.T).type(dtype)
        x_test  = torch.from_numpy(x_test_set.T).type(dtype)
        d_test  = torch.from_numpy(d_test.T).type(dtype)

    return x_train, d_train, x_test, d_test

def update(x, d, parameters, state, gradients, hyperparameters):
    W     = parameters["W"]
    Y     = parameters["Y"]
    Z     = parameters["Z"]
    bias  = parameters["bias"]
    e     = state["e"]
    p     = state["p"]
    p_t   = state["p_t"]
    b     = state["b"]
    b_t   = state["b_t"]
    delta = gradients["delta"]

    state["conv_layers_optimizer"].zero_grad()
    state["conv_layers_output"] = parameters["conv_layers"].forward(x)
    x = state["conv_layers_output"].data

    for i in range(hyperparameters["num_hidden_layers"]+1):
        if i == 0:
            e[i] = torch.sigmoid(W[i].mm(x) + bias[i])
        else:
            e[i] = torch.sigmoid(W[i].mm(e[i-1]) + bias[i])

    train_error = 100.0*int(torch.sum(torch.ne(torch.max(e[-1], 0)[1], torch.max(d, 0)[1])))/x.shape[1]

    for i in range(hyperparameters["num_hidden_layers"], -1, -1):
        if i == hyperparameters["num_hidden_layers"]:
            p[i]   = hyperparameters["gamma"]*torch.ones(e[i].shape).type(dtype)
            p_t[i] = hyperparameters["gamma"]*(d - e[i] + 1)

            b[i]    = p[i]*e[i]
            b_t[i]  = p_t[i]*e[i]

            train_cost = torch.sum((p_t[i] - p[i])**2)/(2*hyperparameters["batch_size"])

            delta[i]                = -hyperparameters["gamma"]*(b_t[i] - b[i])*(1 - e[i])
            gradients["W"][i]       = delta[i].mm(e[i-1].transpose(0, 1))
            gradients["bias"][i]    = torch.sum(delta[i], dim=1).unsqueeze(1)
            gradients["bias_bp"][i] = torch.sum(delta[i], dim=1).unsqueeze(1)
        else:
            if hyperparameters["use_recurrent_input"]:
                c = Z[i].mm(e[i])
            else:
                c = 0

            u = Y[i+1].mm(b[i+1]*(1 - e[i+1])) - c

            p[i]   = torch.sigmoid(hyperparameters["beta"]*u)
            p_t[i] = torch.sigmoid(hyperparameters["beta"]*(Y[i+1].mm(b_t[i+1]*(1 - e[i+1])) - c))

            b[i]    = p[i]*e[i]
            b_t[i]  = p_t[i]*e[i]

            if hyperparameters["use_backprop"]:
                delta[i] = W[i+1].transpose(0, 1).mm(delta[i+1])*e[i]*(1 - e[i])
                gradients["bias_bp"][i] = torch.sum(delta[i], dim=1).unsqueeze(1)
            else:
                delta[i] = -hyperparameters["gamma"]*(b_t[i] - b[i])*(1 - e[i])
                gradients["bias_bp"][i] = torch.sum(W[i+1].transpose(0, 1).mm(delta[i+1])*e[i]*(1 - e[i]), dim=1).unsqueeze(1)

            if i > 0:
                gradients["W"][i] = delta[i].mm(e[i-1].transpose(0, 1))
            else:
                gradients["W"][i] = delta[i].mm(x.transpose(0, 1))

            gradients["bias"][i] = torch.sum(delta[i], dim=1).unsqueeze(1)

            if hyperparameters["use_recurrent_input"]:
                gradients["Z"][i] = -u.mm(e[i].transpose(0, 1))

    if hyperparameters["use_backprop"]:
        gradients["conv"] = W[0].transpose(0, 1).mm(delta[0])
    else:
        u   = Y[0].mm(b[0]*(1 - e[0]))
        u_t = Y[0].mm(b_t[0]*(1 - e[0]))

        gradients["conv"] = u - u_t

    return state, gradients, train_cost, train_error

def update_weights(parameters, state, gradients, hyperparameters):
    for i in range(hyperparameters["num_hidden_layers"]+1):
        parameters["delta_W"][i]    = -hyperparameters["forward_learning_rates"][i+1]*gradients["W"][i] + hyperparameters["momentum"]*parameters["delta_W"][i]
        parameters["delta_bias"][i] = -hyperparameters["forward_learning_rates"][i+1]*gradients["bias"][i] + hyperparameters["momentum"]*parameters["delta_bias"][i]

        parameters["W"][i]    += parameters["delta_W"][i] - hyperparameters["weight_decay"]*parameters["W"][i]
        parameters["bias"][i] += parameters["delta_bias"][i]

        if hyperparameters["use_recurrent_input"] and i < hyperparameters["num_hidden_layers"]:
            parameters["delta_Z"][i] = -hyperparameters["recurrent_learning_rates"][i]*gradients["Z"][i]

            parameters["Z"][i] += parameters["delta_Z"][i]

    if hyperparameters["symmetric_weights"]:
        parameters["Y"]  = [ parameters["W"][i].transpose(0, 1).clone() for i in range(0, hyperparameters["num_hidden_layers"]+1) ]
    elif hyperparameters["same_sign_weights"]:
        for i in range(0, hyperparameters["num_hidden_layers"]):
            mask = (torch.sign(parameters["Y"][i]) != torch.sign(parameters["W"][i].transpose(0, 1))).type(dtype_byte)
            parameters["Y"][i][mask] *= -1

    state["conv_layers_output"].backward(gradient=gradients["conv"])
    state["conv_layers_optimizer"].step()

    return parameters

def test(x, d, parameters, hyperparameters):
    W    = parameters["W"]
    bias = parameters["bias"]

    e = [ [] for i in range(hyperparameters["num_hidden_layers"]+1) ]

    x = parameters["conv_layers"].forward(x).data

    for i in range(hyperparameters["num_hidden_layers"]+1):
        if i == 0:
            e[i] = torch.sigmoid(W[i].mm(x) + bias[i])
        else:
            e[i] = torch.sigmoid(W[i].mm(e[i-1]) + bias[i])

    # print(e[-1])

    test = 100.0*int(torch.sum(torch.ne(torch.max(e[-1], 0)[1], torch.max(d, 0)[1])))/x.shape[1]

    return test

def initialize(hyperparameters):
    num_hidden_layers = hyperparameters["num_hidden_layers"]
    num_hidden_units  = hyperparameters["num_hidden_units"]

    conv_layers           = ConvLayers(dataset=hyperparameters["dataset"])
    conv_layers_criterion = nn.MSELoss()
    conv_layers_optimizer = optim.SGD(conv_layers.parameters(), lr=hyperparameters["forward_learning_rates"][0], momentum=hyperparameters["momentum"], weight_decay=hyperparameters["weight_decay"])

    if hyperparameters["dataset"] == "mnist":
        input_size  = 784
        output_size = 10
    elif hyperparameters["dataset"] == "cifar10":
        input_size  = 1024
        output_size = 10

    x          = torch.zeros((input_size, 1)).type(dtype)
    outputs    = conv_layers.forward(x)
    input_size = conv_layers.output_size

    num_units  = [input_size] + num_hidden_units + [output_size]
    num_layers = num_hidden_layers + 2

    W_range = [ np.sqrt(6/(num_units[i-1] + num_units[i])) for i in range(1, num_layers) ]

    parameters = {
        "W":           [ torch.from_numpy(np.random.uniform(-W_range[i-1], W_range[i-1], size=(num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "bias":        [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "Z":           [ torch.from_numpy(np.random.uniform(-hyperparameters["Z_range"][i-1], hyperparameters["Z_range"][i-1], size=(num_units[i], num_units[i]))).type(dtype) for i in range(1, num_layers-1) ],
        "delta_W":     [ torch.from_numpy(np.zeros((num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "delta_bias":  [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "delta_Z":     [ torch.from_numpy(np.zeros((num_units[i], num_units[i]))).type(dtype) for i in range(1, num_layers-1) ],
        "conv_layers": conv_layers
    }

    if hyperparameters["symmetric_weights"]:
        parameters["Y"] = [ parameters["W"][i].transpose(0, 1).clone() for i in range(0, num_layers-1) ]
    else:
        parameters["Y"] = [ torch.from_numpy(np.random.uniform(-hyperparameters["Y_range"][i], hyperparameters["Y_range"][i], size=(num_units[i], num_units[i+1]))).type(dtype) for i in range(0, num_layers-1) ]

        if hyperparameters["same_sign_weights"]:
            for i in range(0, hyperparameters["num_hidden_layers"]+1):
                mask = (torch.sign(parameters["Y"][i]) != torch.sign(parameters["W"][i].transpose(0, 1))).type(dtype_byte)
                parameters["Y"][i][mask] *= -1
    
    state = {
        "e":                     [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "p":                     [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "p_t":                   [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "b":                     [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "b_t":                   [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "conv_layers_criterion": conv_layers_criterion,
        "conv_layers_optimizer": conv_layers_optimizer
    }

    gradients = {
        "W":       [ torch.from_numpy(np.zeros((num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "bias":    [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "bias_bp": [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "Z":       [ torch.from_numpy(np.zeros((num_units[i], num_units[i]))).type(dtype) for i in range(1, num_layers) ],
        "delta":   [ torch.from_numpy(np.zeros((num_units[i], hyperparameters["batch_size"]))).type(dtype) for i in range(1, num_layers) ]
    }

    return parameters, state, gradients

class ConvLayers(nn.Module):
    def __init__(self, dataset):
        super(ConvLayers, self).__init__()
        self.output_size = None
        self.dataset     = dataset
        
        if dataset == "mnist":
            self.conv1 = nn.Conv2d(1, 8, 5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(8, 16, 5)
            self.pool2 = nn.MaxPool2d(2, 2)

            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
        elif dataset == "cifar10":
            self.conv1 = nn.Conv2d(3, 8, 5, stride=1, padding=2)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(8, 16, 5, stride=1, padding=2)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(16, 20, 5, stride=1, padding=2)
            self.pool3 = nn.MaxPool2d(2, 2)

            torch.nn.init.xavier_uniform_(self.conv1.weight)
            torch.nn.init.xavier_uniform_(self.conv2.weight)
            torch.nn.init.xavier_uniform_(self.conv3.weight)

        if cuda:
            self.cuda()

    def forward(self, x):
        batch_size = x.size()[1]

        if self.dataset == "mnist":
            x = self.pool1(torch.sigmoid(self.conv1(Variable(x.transpose(0, 1)).contiguous().view(batch_size, 1, 28, 28))))
            x = self.pool2(torch.sigmoid(self.conv2(x)))
        elif self.dataset == "cifar10":
            x = self.pool1(torch.sigmoid(self.conv1(Variable(x.transpose(0, 1)).contiguous().view(batch_size, 3, 32, 32))))
            x = self.pool2(torch.sigmoid(self.conv2(x)))
            x = self.pool3(torch.sigmoid(self.conv3(x)))

        self.output_size = self.flat_output_size(x)
        x = x.view(batch_size, self.output_size).transpose(0, 1)

        return x

    def flat_output_size(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension

        flat_output_size = 1
        for s in size:
            flat_output_size *= s

        return flat_output_size
