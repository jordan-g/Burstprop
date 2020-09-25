'''
Main script for training a network on MNIST using backprop, feedback alignment, node pertubation or burstprop as presented in

"Payeur, A., Guerguiev, J., Zenke, F., Richards, B., & Naud, R. (2020).
Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits. bioRxiv."

     Author: Jordan Guergiuev
     E-mail: jordan.guerguiev@me.com
       Date: April 5, 2020
Institution: University of Toronto Scarborough

Copyright (C) 2020 Jordan Guerguiev

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import torch
import torchvision

import os
import datetime
import shutil
import argparse

from tqdm import tqdm
from tensorboardX import SummaryWriter

from networks import *

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Path to directory where data will be saved')
parser.add_argument("-n_epochs", type=int, help="Number of epochs", default=500)
parser.add_argument("-batch_size", type=int, help="Batch size", default=32)
parser.add_argument('-validation', default=False, help="Whether to the validation set", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-conv_net', default=False, help="Whether to use the convolutional network", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-n_hidden_layers", type=int, help="Number of hidden layers (in the fully-connected network)", default=3)
parser.add_argument("-hidden_lr", help="Learning rate for hidden layers", type=float, default=0.1)
parser.add_argument("-output_lr", help="Learning rate for output layer", type=float, default=0.01)
parser.add_argument("-weight_fa_std", help="Range of initial feedback weights for hidden layers", type=float, default=1.0)
parser.add_argument("-momentum", type=float, help="Momentum", default=0.8)
parser.add_argument("-weight_decay", type=float, help="Weight decay", default=1e-6)
parser.add_argument("-p_baseline", type=float, help="Output layer baseline burst probability", default=0.2)
parser.add_argument('-use_backprop', default=False, help="Whether to train using backprop", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-weight_fa_learning', default=False, help="Whether to update feedback weights", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-recurrent_input', default=True, help="Whether to use recurrent input", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-weight_r_std", help="Range of initial recurrent weights", type=float, default=0.01)
parser.add_argument('-weight_r_learning', default=True, help="Whether to update recurrent weights", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-recurrent_lr", help="Learning rate for recurrent weights", type=float, default=0.0001)
parser.add_argument('-use_node_pertubation', default=False, help="Whether to use node pertubation", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-sequential_pertubation', default=False, help="Whether to sequentially perturb layers or units rather than all units in the network", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-perturb_units', default=False, help="Whether to sequentially perturb single units", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-xi_mean", type=float, help="Mean of node pertubation noise", default=0.0)
parser.add_argument("-xi_std", type=float, help="Standard deviation of node pertubation noise", default=0.001)
parser.add_argument("-info", type=str, help="Any other information about the simulation", default="")

args=parser.parse_args()

directory              = args.directory
n_epochs               = args.n_epochs
batch_size             = args.batch_size
validation             = args.validation
conv_net               = args.conv_net
n_hidden_layers        = args.n_hidden_layers
hidden_lr              = args.hidden_lr
output_lr              = args.output_lr
weight_fa_std          = args.weight_fa_std
momentum               = args.momentum
weight_decay           = args.weight_decay
p_baseline             = args.p_baseline
use_backprop           = args.use_backprop
weight_fa_learning     = args.weight_fa_learning
recurrent_input        = args.recurrent_input
weight_r_std           = args.weight_r_std
weight_r_learning      = args.weight_r_learning
recurrent_lr           = args.recurrent_lr
use_node_pertubation   = args.use_node_pertubation
sequential_pertubation = args.sequential_pertubation
perturb_units          = args.perturb_units
xi_mean                = args.xi_mean
xi_std                 = args.xi_std
info                   = args.info

if use_backprop:
    use_node_pertubation = False
    weight_fa_learning   = False
    recurrent_input      = False

if not recurrent_input:
    weight_r_learning = False

if use_backprop:
    if conv_net:
        lr = [output_lr]*5
    else:
        lr = [output_lr]*(n_hidden_layers+2)
else:
    if conv_net:
        lr = [hidden_lr]*4 + [output_lr]
    else:
        lr = [hidden_lr]*n_hidden_layers + [output_lr]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.MNIST(root='./Data', train=True, download=True, transform=transform_train)

if validation:
    train_set, test_set = torch.utils.data.random_split(train_set, [50000, 10000])
else:
    test_set = torchvision.datasets.MNIST(root='./Data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

if use_node_pertubation:
    if conv_net:
        net = MNISTConvNetNP(input_channels=1, xi_mean=xi_mean, xi_std=xi_std, device=device).to(device)
    else:
        net = MNISTNetNP(input_channels=1, n_hidden_layers=n_hidden_layers, xi_mean=xi_mean, xi_std=xi_std, device=device).to(device)
elif use_backprop:
    if conv_net:
        net = MNISTConvNetBP(input_channels=1).to(device)
    else:
        net = MNISTNetBP(input_channels=1, n_hidden_layers=n_hidden_layers).to(device)
else:
    if conv_net:
        net = MNISTConvNet(input_channels=1, p_baseline=p_baseline, weight_fa_std=weight_fa_std, weight_r_std=weight_r_std, weight_fa_learning=weight_fa_learning, recurrent_input=recurrent_input, weight_r_learning=weight_r_learning, device=device).to(device)
    else:
        net = MNISTNet(input_channels=1, p_baseline=p_baseline, weight_fa_std=weight_fa_std, weight_r_std=weight_r_std, weight_fa_learning=weight_fa_learning, recurrent_input=recurrent_input, weight_r_learning=weight_r_learning, n_hidden_layers=n_hidden_layers, device=device).to(device)

if use_backprop:
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=output_lr, momentum=momentum, weight_decay=weight_decay)

def train(epoch):
    net.train()

    train_loss = 0
    correct    = 0
    total      = 0

    if conv_net:
        avg_delta_angles = [0 for i in range(4)]
    else:
        avg_delta_angles = [0 for i in range(n_hidden_layers)]

    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        t = F.one_hot(targets, num_classes=10).float()

        if use_backprop:
            optimizer.zero_grad()

        outputs = net(inputs)

        if sequential_pertubation:
            net.forward_backward_weight_update_perturb(inputs, t, lr=lr, momentum=momentum, weight_decay=weight_decay, perturb_units=perturb_units, batch_size=inputs.shape[0])

            loss = net.loss(outputs, t)

            train_loss += loss
        else:
            if use_node_pertubation:
                net.forward_perturb(inputs)

            if use_backprop:
                loss = criterion(outputs, t)

                loss.backward()
                optimizer.step()

                train_loss  += loss.item()
            else:
                loss = net.loss(outputs, t)

                net.backward(t)

                if not use_node_pertubation:
                    delta_angles = net.delta_angles()
                    for i in range(len(delta_angles)):
                        avg_delta_angles[i] += delta_angles[i]

                    if epoch == 0 and batch_idx == 0 and not use_node_pertubation:
                        weight_angles = net.weight_angles()
                        delta_angles = net.delta_angles()

                        for i in range(len(weight_angles)):
                            writer.add_scalar('weight_angle/{}'.format(i), weight_angles[i], 0)
                            writer.add_scalar('delta_angle/{}'.format(i), delta_angles[i], 0)

                if use_node_pertubation:
                    net.update_weights(lr=lr, momentum=momentum, weight_decay=weight_decay, batch_size=inputs.shape[0])
                else:
                    net.update_weights(lr=lr, momentum=momentum, weight_decay=weight_decay, recurrent_lr=recurrent_lr, batch_size=inputs.shape[0])

                train_loss  += loss

        _, predicted = outputs.max(1)
        total       += targets.size(0)
        correct     += predicted.eq(targets).sum().item()

        progress_bar.set_description("Train Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(train_loss/(batch_idx+1), 100*correct/total, correct, total))

    if not use_backprop:
        for i in range(len(avg_delta_angles)):
            avg_delta_angles[i] /= len(train_loader)

    return 100*(1 - correct/total), train_loss/(batch_idx+1), avg_delta_angles

def test():
    net.eval()

    test_loss = 0
    correct   = 0
    total     = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            t = F.one_hot(targets, num_classes=10).float()

            if use_backprop:
                optimizer.zero_grad()

            outputs = net(inputs)

            if not use_backprop:
                loss = net.loss(outputs, t)

                test_loss   += loss

            if use_backprop:
                loss = criterion(outputs, t)

                test_loss  += loss.item()

            _, predicted = outputs.max(1)
            total       += targets.size(0)
            correct     += predicted.eq(targets).sum().item()

            progress_bar.set_description("Test Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(test_loss/(batch_idx+1), 100*correct/total, correct, total))

    return 100*(1 - correct/total), test_loss/(batch_idx+1)

if directory is not None:
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save a human-readable text file containing simulation details
    timestamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    with open(os.path.join(directory, "params.txt"), "w") as f:
        f.write("Simulation run @ {}\n".format(timestamp))
        f.write("Number of epochs: {}\n".format(n_epochs))
        f.write("Batch size: {}\n".format(batch_size))
        f.write("Using validation set: {}\n".format(validation))
        f.write("Convolutional network: {}\n".format(conv_net))
        if not conv_net:
            f.write("Number of hidden layers: {}\n".format(n_hidden_layers))
        f.write("Feedforward learning rates: {}\n".format(lr))
        f.write("Feedback weight initialization standard deviation: {}\n".format(weight_fa_std))
        f.write("Momentum: {}\n".format(momentum))
        f.write("Weight decay: {}\n".format(weight_decay))
        f.write("Output layer baseline burst probability: {}\n".format(p_baseline))
        f.write("Using backprop: {}\n".format(use_backprop))
        f.write("Feedback weight learning: {}\n".format(weight_fa_learning))
        f.write("Recurrent input: {}\n".format(recurrent_input))
        f.write("Recurrent weight initialization standard deviation: {}\n".format(weight_r_std))
        f.write("Recurrent weight learning: {}\n".format(weight_r_learning))
        f.write("Recurrent weight learning rate: {}\n".format(recurrent_lr))
        f.write("Using node pertubation: {}\n".format(use_node_pertubation))
        f.write("Sequential node pertubation: {}\n".format(sequential_pertubation))
        f.write("Perturb only single units: {}\n".format(perturb_units))
        f.write("Mean of node pertubation noise: {}\n".format(xi_mean))
        f.write("Standrd deviation of node pertubation noise: {}\n".format(xi_std))
        if info != "":
            f.write("Other info: {}\n".format(info))

    filename = os.path.basename(__file__)
    if filename.endswith('pyc'):
        filename = filename[:-1]
    shutil.copyfile(filename, os.path.join(directory, filename))
    shutil.copyfile("networks.py", os.path.join(directory, "networks.py"))
    shutil.copyfile("layers.py", os.path.join(directory, "layers.py"))

    # initialize a Tensorboard writer
    writer = SummaryWriter(log_dir=directory)

    torch.save(net, os.path.join(directory, "initial_model.pth"))

if directory is not None:
    test_error, test_loss = test()

    writer.add_scalar('Test Error', test_error, 0)
    writer.add_scalar('Test Loss', test_loss, 0)

for epoch in range(n_epochs):
    print("\nEpoch {}.".format(epoch+1))

    train_error, train_loss, delta_angles = train(epoch)
    test_error, test_loss   = test()

    if directory is not None:
        writer.add_scalar('Train Error', train_error, epoch+1)
        writer.add_scalar('Train Loss', train_loss, epoch+1)
        writer.add_scalar('Test Error', test_error, epoch+1)
        writer.add_scalar('Test Loss', test_loss, epoch+1)

        if not use_backprop and not use_node_pertubation:
            weight_angles = net.weight_angles()
            delta_angles = net.delta_angles()

            for i in range(len(weight_angles)):
                writer.add_scalar('weight_angle/{}'.format(i), weight_angles[i], epoch+1)
                writer.add_scalar('delta_angle/{}'.format(i), delta_angles[i], epoch+1)

    torch.save(net, os.path.join(directory, "model.pth"))
