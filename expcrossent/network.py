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
            x_test  = x_train[50000:, :]
            x_train = x_train[:50000, :]
            d_test  = d_train[50000:, :]
            d_train = d_train[:50000, :]

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
        x_test  = torch.from_numpy(x_test.T).type(dtype)
        d_test  = torch.from_numpy(d_test.T).type(dtype)

    return x_train, d_train, x_test, d_test

def update(x, d, parameters, state, gradients, hyperparameters):
    W     = parameters["W"]
    Y     = parameters["Y"]
    bias  = parameters["bias"]
    e     = state["e"]
    p     = state["p"]
    delta = gradients["delta"]
    p_0   = hyperparameters["p_0"]
    alpha = inverse_sigmoid(p_0)

    for i in range(hyperparameters["num_hidden_layers"]+1):
        if i == 0:
            e[i] = torch.exp(W[i].mm(x) + bias[i])
            print("layer {}: [{},{}]".format(i,torch.min(e[i]),torch.max(e[i])))
        elif i < hyperparameters["num_hidden_layers"]:
            e[i] = torch.exp(W[i].mm(e[i-1]) + bias[i])
            print("layer {}: [{},{}]".format(i,torch.min(e[i]),torch.max(e[i])))
        else:
            e[i] = torch.softmax(W[i].mm(e[i-1]) + bias[i], dim=0)
            #print("shape of last: {}".format(torch.sum(e[i],dim=0)))
            #print(torch.max(e[i]))

    train_error = 100.0*int(torch.sum(torch.ne(torch.max(e[-1], 0)[1], torch.max(d, 0)[1])))/x.shape[1]

    for i in range(hyperparameters["num_hidden_layers"], -1, -1):
        if i == hyperparameters["num_hidden_layers"]:
            #  compute cost locally
            output     = torch.clamp(e[i], 1e-7, 1.-1e-7) #  clipping to prevent problems with log
            s          = torch.sum(-d*torch.log(output), dim=1)
            train_cost = torch.mean(s)

            delta[i]             = d - e[i]
            gradients["W"][i]    = delta[i].mm(e[i-1].transpose(0, 1))/x.shape[1]
            gradients["bias"][i] = torch.mean(delta[i], dim=1, keepdim=True)

        else:
            if hyperparameters["use_backprop"]:
                delta[i] = W[i+1].transpose(0, 1).mm(delta[i+1])*e[i]
            else:
                du_i = Y[i].mm(delta[i+1])
                #print("deltau in [{},{}]".format(torch.min(du_i), torch.max(du_i)))
                p[i] = torch.sigmoid(alpha + hyperparameters["beta"] * du_i)
                #print("p in [{},{}]".format(torch.min(p[i]), torch.max(p[i])))
                delta[i] = (p[i] - p_0)*e[i]
            gradients["bias"][i] = torch.mean(delta[i], dim=1, keepdim=True)

            if i > 0:
                gradients["W"][i] = delta[i].mm(e[i-1].transpose(0, 1))/x.shape[1]
            else:
                gradients["W"][i] = delta[i].mm(x.transpose(0, 1))/x.shape[1]

    return state, gradients, train_cost, train_error

def update_weights(parameters, state, gradients, hyperparameters):
    eta = hyperparameters["forward_learning_rates"]
    mom = hyperparameters["momentum"]
    for i in range(hyperparameters["num_hidden_layers"]+1):
        if i==hyperparameters["num_hidden_layers"]:
            parameters["momentum"][i] = mom*parameters["momentum"][i] + (1-mom)*(gradients["W"][i] - hyperparameters["weight_decay"] * parameters["W"][i])
            parameters["delta_W"][i] = eta[i]*parameters["momentum"][i]
            #parameters["delta_W"][i] = eta[i]*gradients["W"][i] + hyperparameters["momentum"]*parameters["delta_W"][i]
        else:
            parameters["momentum"][i] = mom * parameters["momentum"][i] + (1 - mom)*(gradients["W"][i] - hyperparameters["heterosyn_plasticity"]*torch.mean(state["e"][i], 1).unsqueeze(1)*parameters["W"][i] - hyperparameters["weight_decay"] * parameters["W"][i])
            parameters["delta_W"][i] = eta[i] * parameters["momentum"][i]
            # parameters["delta_W"][i] = eta[i]*gradients["W"][i] + hyperparameters["momentum"]*parameters["delta_W"][i]

        parameters["momentum_bias"][i] = mom*parameters["momentum_bias"][i] + (1-mom)*gradients["bias"][i]
        parameters["delta_bias"][i] = eta[i]*parameters["momentum_bias"][i]
        #parameters["delta_bias"][i] = eta[i]*gradients["bias"][i] + hyperparameters["momentum"]*parameters["delta_bias"][i]

        parameters["W"][i] += parameters["delta_W"][i]
        parameters["bias"][i] += parameters["delta_bias"][i]
        '''
        if i == hyperparameters["num_hidden_layers"]:
            parameters["W"][i]    += parameters["delta_W"][i] - hyperparameters["weight_decay"]*parameters["W"][i]
            parameters["bias"][i] += parameters["delta_bias"][i]
        else:
            parameters["W"][i] += parameters["delta_W"][i] - hyperparameters["weight_decay"] * parameters["W"][i] - \
                                  eta[i]*hyperparameters["heterosyn_plasticity"]*torch.mean(state["e"][i], 1).unsqueeze(1)*parameters["W"][i]
            parameters["bias"][i] += parameters["delta_bias"][i]
        '''
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
            e[i] = torch.exp(W[i].mm(x) + bias[i])
        else:
            e[i] = torch.exp(W[i].mm(e[i-1]) + bias[i])

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

    batch_size = hyperparameters["batch_size"]

    W_range = [ np.sqrt(6/(num_units[i-1] + num_units[i])) for i in range(1, num_layers) ]
    W_range[-1] = W_range[-1]

    parameters = {
        "W":          [ torch.from_numpy(np.random.uniform(-W_range[i-1], W_range[i-1], size=(num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "bias":       [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "delta_W":    [ torch.from_numpy(np.zeros((num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "delta_bias": [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "momentum":   [ torch.from_numpy(np.zeros((num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "momentum_bias": [torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers)]
    }

    if not hyperparameters["symmetric_weights"]:
        print('User-defined Y-range ignored for this model...')
    Y_range = [np.sqrt(6./(num_units[i+1] + num_units[i+2])) for i in range(0, num_layers - 2)]
    #Y_range[-1] = 10.*Y_range[-1]

    if hyperparameters["symmetric_weights"]:
        parameters["Y"] = [ parameters["W"][i+1].transpose(0, 1).clone() for i in range(0, num_layers-2) ]
    else:
        parameters["Y"] = [ torch.from_numpy(np.random.uniform(-Y_range[i], Y_range[i], size=(num_units[i+1], num_units[i+2]))).type(dtype) for i in range(0, num_layers-2) ]

        if hyperparameters["same_sign_weights"]:
            for i in range(0, hyperparameters["num_hidden_layers"]):
                mask = (np.sign(parameters["Y"][i]) != np.sign(parameters["W"][i+1].transpose(0, 1))).type(dtype_byte)
                parameters["Y"][i][mask] *= 1
                
    state = {
        "e":   [ torch.from_numpy(np.zeros((num_units[i], batch_size))).type(dtype) for i in range(1, num_layers) ],
        "p":   [ torch.from_numpy(np.zeros((num_units[i], batch_size))).type(dtype) for i in range(1, num_layers) ],
    }

    gradients = {
        "W":       [ torch.from_numpy(np.zeros((num_units[i], num_units[i-1]))).type(dtype) for i in range(1, num_layers) ],
        "bias":    [ torch.from_numpy(np.zeros((num_units[i], 1))).type(dtype) for i in range(1, num_layers) ],
        "delta":   [ torch.from_numpy(np.zeros((num_units[i], batch_size))).type(dtype) for i in range(1, num_layers) ]
    }

    return parameters, state, gradients

def inverse_sigmoid(x):
    return np.log(x/(1.-x))