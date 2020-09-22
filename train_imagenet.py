'''
Main script for training a network on ImageNet using backprop, feedback alignment or burstprop as presented in

"Payeur, A., Guerguiev, J., Zenke, F., Richards, B., & Naud, R. (2020).
Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits. bioRxiv."

This code was partially adapted from https://github.com/pytorch/examples/tree/master/imagenet.

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
import time
import datetime

from tqdm import tqdm
from tensorboardX import SummaryWriter

from networks_imagenet import *

parser = argparse.ArgumentParser()
parser.add_argument('folder_prefix', help='Prefix of folder name where data will be saved')
parser.add_argument('data_path', help='Path to the dataset', type=str)
parser.add_argument("-n_epochs", type=int, help="Number of epochs", default=500)
parser.add_argument("-batch_size", type=int, help="Batch size", default=128)
parser.add_argument('-validation', default=False, help="Whether to the validation set", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-hidden_lr", help="Learning rate for hidden layers", type=float, default=0.01)
parser.add_argument("-output_lr", help="Learning rate for output layer", type=float, default=0.01)
parser.add_argument("-weight_fa_std", help="Standard deviation of initial feedback weights for hidden layers", type=float, default=0.01)
parser.add_argument("-momentum", type=float, help="Momentum", default=0.9)
parser.add_argument("-weight_decay", type=float, help="Weight decay", default=1e-4)
parser.add_argument("-p_baseline", type=float, help="Output layer baseline burst probability", default=0.2)
parser.add_argument('-use_backprop', default=False, help="Whether to train using backprop", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-weight_fa_learning', default=True, help="Whether to update feedback weights", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-kappa", type=float, help="Scaling factor used in target burst probability at output layer", default=1e-5)
parser.add_argument('-use_adam', default=False, help="Whether to use the Adam optimizer", type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-resume_path', default='', help=' (Optional) Path to latest saved checkpoint to resume from', type=str)
parser.add_argument("-info", type=str, help="Any other information about the simulation", default="")

args=parser.parse_args()

folder_prefix          = args.folder_prefix
data_path              = args.data_path
n_epochs               = args.n_epochs
batch_size             = args.batch_size
validation             = args.validation
hidden_lr              = args.hidden_lr
output_lr              = args.output_lr
weight_fa_std          = args.weight_fa_std
momentum               = args.momentum
weight_decay           = args.weight_decay
p_baseline             = args.p_baseline
use_backprop           = args.use_backprop
weight_fa_learning     = args.weight_fa_learning
kappa                  = args.kappa
use_adam               = args.use_adam
resume_path            = args.resume_path
info                   = args.info

n_gpus_per_node = torch.cuda.device_count()

best_acc1 = 0

if use_backprop:
    weight_fa_learning = False

if use_backprop:
    lr = [output_lr]*8
else:
    lr = [hidden_lr]*7 + [output_lr]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if use_backprop:
    net = ImageNetConvNetBP(input_channels=3)
else:
    net = ImageNetConvNet(input_channels=3, p_baseline=p_baseline, weight_fa_std=weight_fa_std, weight_fa_learning=weight_fa_learning, kappa=kappa)

net = torch.nn.DataParallel(net).cuda()

module = net.module

criterion = torch.nn.CrossEntropyLoss().cuda()

if use_backprop:
    if not use_adam:
        optimizer = torch.optim.SGD(net.parameters(), lr=output_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=output_lr, betas=[0.9, 0.99], eps=0.1)
else:
    if not use_adam:
        optimizer = torch.optim.SGD([
                                    {"params": module.conv1.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": module.conv2.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": module.conv3.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": module.conv4.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": module.conv5.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": module.conv6.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": module.conv7.parameters(), "lr": hidden_lr, "weight_decay": weight_decay, "momentum": momentum},
                                    {"params": module.fc1.parameters(), "lr": output_lr, "weight_decay": weight_decay, "momentum": momentum}
                                    ], output_lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam([
                                    {"params": module.conv1.parameters(), "lr": hidden_lr},
                                    {"params": module.conv2.parameters(), "lr": hidden_lr},
                                    {"params": module.conv3.parameters(), "lr": hidden_lr},
                                    {"params": module.conv4.parameters(), "lr": hidden_lr},
                                    {"params": module.conv5.parameters(), "lr": hidden_lr},
                                    {"params": module.conv6.parameters(), "lr": hidden_lr},
                                    {"params": module.conv7.parameters(), "lr": hidden_lr},
                                    {"params": module.fc1.parameters(), "lr": output_lr}
                                    ], output_lr, betas=[0.9, 0.99], eps=0.1)

start_epoch = 0
if len(resume_path) > 0:
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

train_dir = os.path.join(data_path, 'train')
test_dir  = os.path.join(data_path, 'val')

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)

if validation:
    train_set, test_set = torch.utils.data.random_split(train_set, [1231167, 50000])
else:
    test_set  = torchvision.datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'best_model.pth.tar'))

def adjust_learning_rate(optimizer, starting_lrs, epoch):
    # set the learning rate to the initial learning rate decayed by 10 every 30 epochs
    for i in range(len(optimizer.param_groups)):
        param_group = optimizer.param_groups[i]
        lr = starting_lrs[i] * (0.1 ** (epoch // 30))
        param_group['lr'] = lr

class AverageMeter(object):
    # computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    # computes the accuracy over the k top predictions for the specified values of k
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    net.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        targets = targets.cuda(non_blocking=True)

        if not use_backprop:
            t = F.one_hot(targets, num_classes=1000).float()

        # compute output
        if use_backprop:
            outputs = net(inputs)
        else:
            outputs = net(inputs, t)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            progress.display(batch_idx)

    return top1.avg, top5.avg, losses.avg

def test():
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    net.eval()

    with torch.no_grad():
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            targets = targets.cuda(non_blocking=True)

            if not use_backprop:
                t = F.one_hot(targets, num_classes=1000).float()

            # compute output
            if use_backprop:
                outputs = net(inputs)
            else:
                outputs = net(inputs, t)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                progress.display(batch_idx)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

if folder_prefix is not None:
    # generate a name for the folder where data will be stored
    lr_string            = " ".join([ str(i) for i in lr ])
    weight_fa_std_string = "{}".format(weight_fa_std)

    folder = "{} - {} - {} - {} - {} - {} - {}".format(folder_prefix, lr_string, weight_fa_std_string, batch_size, momentum, weight_decay, p_baseline) + " - BP"*(use_backprop == True) + " - {}".format(info)*(info != "")
else:
    folder = None

if folder is not None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save a human-readable text file containing simulation details
    timestamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    with open(os.path.join(folder, "params.txt"), "w") as f:
        f.write("Simulation run @ {}\n".format(timestamp))
        f.write("Number of epochs: {}\n".format(n_epochs))
        f.write("Batch size: {}\n".format(batch_size))
        f.write("Using validation set: {}\n".format(validation))
        f.write("Feedforward learning rates: {}\n".format(lr))
        f.write("Feedback weight initialization standard deviation: {}\n".format(weight_fa_std))
        f.write("Momentum: {}\n".format(momentum))
        f.write("Weight decay: {}\n".format(weight_decay))
        f.write("Output layer baseline burst probability: {}\n".format(p_baseline))
        f.write("Using backprop: {}\n".format(use_backprop))
        f.write("Feedback weight learning: {}\n".format(weight_fa_learning))
        f.write("Output layer target burst probability scaling factor: {}\n".format(kappa))
        f.write("Using Adam optimizer: {}\n".format(use_adam))
        f.write("Resuming from path: {}\n".format(resume_path))
        if info != "":
            f.write("Other info: {}\n".format(info))

    filename = os.path.basename(__file__)
    if filename.endswith('pyc'):
        filename = filename[:-1]
    shutil.copyfile(filename, os.path.join(folder, filename))
    shutil.copyfile("networks_imagenet.py", os.path.join(folder, "networks_imagenet.py"))
    shutil.copyfile("layers_imagenet.py", os.path.join(folder, "layers_imagenet.py"))

    # initialize a Tensorboard writer
    writer = SummaryWriter(log_dir=folder)

test_acc1, test_acc5, test_loss = test()

if folder is not None:
    writer.add_scalar('Test Top-1 Accuracy', test_acc1, 0)
    writer.add_scalar('Test Top-5 Accuracy', test_acc5, 0)
    writer.add_scalar('Test Loss', test_loss, 0)

starting_lrs = [ param_group['lr'] for param_group in optimizer.param_groups ]

for epoch in range(start_epoch, n_epochs):
    print("\nEpoch {}.".format(epoch+1))

    adjust_learning_rate(optimizer, starting_lrs, epoch)

    train_acc1, train_acc5, train_loss = train(epoch)
    test_acc1, test_acc5, test_loss = test()

    if folder is not None:
        writer.add_scalar('Train Top-1 Accuracy', train_acc1, epoch+1)
        writer.add_scalar('Train Top-5 Accuracy', train_acc5, epoch+1)
        writer.add_scalar('Train Loss', train_loss, epoch+1)
        writer.add_scalar('Test Top-1 Accuracy', test_acc1, epoch+1)
        writer.add_scalar('Test Top-5 Accuracy', test_acc5, epoch+1)
        writer.add_scalar('Test Loss', test_loss, epoch+1)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()},
            is_best, folder)
