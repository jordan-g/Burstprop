import numpy as np

from mnist import mnist
from cifar10 import cifar10

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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
    W      = parameters["W"]
    Y      = parameters["Y"]
    Z      = parameters["Z"]
    bias   = parameters["bias"]
    e      = state["e"]
    e_prev = state["e_prev"]
    p      = state["p"]
    p_prev = state["p_prev"]
    b      = state["b"]
    b_prev = state["b_prev"]
    delta  = gradients["delta"]

    for t in range(2*hyperparameters["num_hidden_layers"]+1):
        for i in range(hyperparameters["num_hidden_layers"]+1):
            if i == 0:
                e[i] = torch.sigmoid(W[i].mm(x) + bias[i])
            else:
                e[i] = torch.sigmoid(W[i].mm(e_prev[i-1]) + bias[i])

        for i in range(hyperparameters["num_hidden_layers"], -1, -1):
            if i == hyperparameters["num_hidden_layers"]:
                if t == i+1:
                    p[i] = hyperparameters["gamma"]*(d - e[i] + 1)
                else:
                    p[i] = hyperparameters["gamma"]*torch.ones(e[i].shape).type(dtype)
                
                b[i] = p[i]*e[i]

                if t == i+1:
                    train_cost = torch.sum((p[i] - p_prev[i])**2)/(2*hyperparameters["batch_size"])

                    delta[i]                = -hyperparameters["gamma"]*(b[i] - b_prev[i])*(1 - e_prev[i])
                    gradients["W"][i]       = delta[i].mm(e_prev[i-1].transpose(0, 1))
                    gradients["bias"][i]    = torch.sum(delta[i], dim=1).unsqueeze(1)
                    gradients["bias_bp"][i] = torch.sum(delta[i], dim=1).unsqueeze(1)
            else:
                if hyperparameters["use_recurrent_input"]:
                    c = Z[i].mm(b_prev[i])
                else:
                    c = 0

                u = Y[i].mm(b_prev[i+1]*(1 - b_prev[i+1])) - c

                p[i] = torch.sigmoid(hyperparameters["beta"]*u)

                b[i] = p[i]*e[i]

                if t == i+1:
                    if hyperparameters["use_backprop"]:
                        delta[i] = W[i+1].transpose(0, 1).mm(delta[i+1])*e_prev[i]*(1 - e_prev[i])
                        gradients["bias_bp"][i] = torch.sum(delta[i], dim=1).unsqueeze(1)
                    else:
                        delta[i] = -hyperparameters["gamma"]*(b[i] - b_prev[i])*(1 - e_prev[i])
                        gradients["bias_bp"][i] = torch.sum(W[i+1].transpose(0, 1).mm(delta[i+1])*e_prev[i]*(1 - e_prev[i]), dim=1).unsqueeze(1)

                    if i > 0:
                        gradients["W"][i] = delta[i].mm(e_prev[i-1].transpose(0, 1))
                    else:
                        gradients["W"][i] = delta[i].mm(x.transpose(0, 1))

                    gradients["bias"][i] = torch.sum(delta[i], dim=1).unsqueeze(1)

                    if hyperparameters["use_recurrent_input"]:
                        gradients["Z"][i] = -u.mm(b_prev[i].transpose(0, 1))

            e_prev[i] = e[i].clone()
            p_prev[i] = p[i].clone()
            b_prev[i] = b[i].clone()

        if t == hyperparameters["num_hidden_layers"]:
            train_error = 100.0*int(torch.sum(torch.ne(torch.max(e[-1], 0)[1], torch.max(d, 0)[1])))/x.shape[1]

    return state, gradients, train_cost, train_error

def update_weights(parameters, state, gradients, hyperparameters):
    for i in range(hyperparameters["num_hidden_layers"]+1):
        parameters["delta_W"][i]    = -hyperparameters["forward_learning_rates"][i]*gradients["W"][i] + hyperparameters["momentum"]*parameters["delta_W"][i]
        parameters["delta_bias"][i] = -hyperparameters["forward_learning_rates"][i]*gradients["bias"][i] + hyperparameters["momentum"]*parameters["delta_bias"][i]

        parameters["W"][i]    += parameters["delta_W"][i] - hyperparameters["weight_decay"]*parameters["W"][i]
        parameters["bias"][i] += parameters["delta_bias"][i]

        if hyperparameters["use_recurrent_input"] and i < hyperparameters["num_hidden_layers"]:
            parameters["delta_Z"][i] = -hyperparameters["recurrent_learning_rates"][i]*gradients["Z"][i]

            parameters["Z"][i] += parameters["delta_Z"][i]

    if hyperparameters["symmetric_weights"]:
        parameters["Y"]  = [ parameters["W"][i+1].transpose(0, 1).clone() for i in range(0, hyperparameters["num_hidden_layers"]) ]
    elif hyperparameters["same_sign_weights"]:
        for i in range(0, hyperparameters["num_hidden_layers"]):
            mask = (np.sign(parameters["Y"][i]) != np.sign(parameters["W"][i+1].transpose(0, 1))).type(dtype_byte)
            parameters["Y"][i][mask] *= -1

    return parameters

def test(x, d, parameters, hyperparameters):
    W    = parameters["W"]
    bias = parameters["bias"]

    e = [ [] for i in range(hyperparameters["num_hidden_layers"]+1) ]

    for i in range(hyperparameters["num_hidden_layers"]+1):
        if i == 0:
            e[i] = torch.sigmoid(W[i].mm(x) + bias[i])
        else:
            e[i] = torch.sigmoid(W[i].mm(e[i-1]) + bias[i])

    test = 100.0*int(torch.sum(torch.ne(torch.max(e[-1], 0)[1], torch.max(d, 0)[1])))/x.shape[1]

    return test

def initialize(hyperparameters):
    num_hidden_layers = hyperparameters["num_hidden_layers"]
    num_hidden_units  = hyperparameters["num_hidden_units"]

    if hyperparameters["dataset"] == "mnist":
        input_size  = 784
        output_size = 10
    elif hyperparameters["dataset"] == "cifar10":
        input_size  = 1024
        output_size = 10

    num_units  = [input_size] + num_hidden_units + [output_size]
    num_layers = num_hidden_layers + 2

    W_range = [ np.sqrt(6/(num_units[i-1] + num_units[i])) for i in range(1, num_layers) ]

    parameters = {
        "W":          [ torch.from_numpy(np.random.uniform(-W_range[i-1], W_range[i-1], size=(num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "bias":       [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "Z":          [ torch.from_numpy(np.random.uniform(-hyperparameters["Z_range"][i-1], hyperparameters["Z_range"][i-1], size=(num_units[i], num_units[i]))).type(dtype) for i in range(1, num_layers-1) ],
        "delta_W":    [ torch.from_numpy(np.zeros((num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "delta_bias": [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "delta_Z":    [ torch.from_numpy(np.zeros((num_units[i], num_units[i]))).type(dtype) for i in range(1, num_layers-1) ]
    }

    if hyperparameters["symmetric_weights"]:
        parameters["Y"] = [ parameters["W"][i+1].transpose(0, 1).clone() for i in range(0, num_layers-2) ]
    else:
        parameters["Y"] = [ torch.from_numpy(np.random.uniform(-hyperparameters["Y_range"][i], hyperparameters["Y_range"][i], size=(num_units[i+1], num_units[i+2]))).type(dtype) for i in range(0, num_layers-2) ]

        if hyperparameters["same_sign_weights"]:
            for i in range(0, hyperparameters["num_hidden_layers"]):
                mask = (np.sign(parameters["Y"][i]) != np.sign(parameters["W"][i+1].transpose(0, 1))).type(dtype_byte)
                parameters["Y"][i][mask] *= 1
                
    state = {
        "e":      [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "e_prev": [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "p":      [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "p_prev": [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "b":      [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "b_prev": [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ]
    }

    gradients = {
        "W":       [ torch.from_numpy(np.zeros((num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "bias":    [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "bias_bp": [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "Z":       [ torch.from_numpy(np.zeros((num_units[i], num_units[i]))).type(dtype) for i in range(1, num_layers) ],
        "delta":   [ torch.from_numpy(np.zeros((num_units[i], hyperparameters["batch_size"]))).type(dtype) for i in range(1, num_layers) ]
    }

    return parameters, state, gradients
