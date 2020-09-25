## Training on MNIST, CIFAR-10 and ImageNet Using Burstprop

Code from ["Payeur, A., Guerguiev, J., Zenke, F., Richards, B., & Naud, R. (2020). **Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits.** bioRxiv.](https://www.biorxiv.org/content/10.1101/2020.03.30.015511v1)

## System Requirements
This software requires a working Python 3.x installation as well as the following libraries:
- torch
- torchvision
- numpy
- tqdm
- tensorboardx

It has been tested on CentOS Linux 7 and Python 3.6.8.

**Note**: Training convolutional networks requires a GPU accessible to PyTorch.

## Installation
Python 3.x can be installed by downloading the appropriate installer: https://www.python.org

All of the required libraries can be installed using **pip** in the command line:

```
pip install torch torchvision numpy tqdm tensorboardx
```

When running the scripts to train on MNIST or CIFAR-10 for the first time, the datasets will be downloaded automatically by the torchvision library. For training on ImageNet, the dataset must be downloaded by the user (http://www.image-net.org/challenges/LSVRC/2012/), and the path to the dataset must be supplied as a command line argument.

## Instructions
There are three main scripts to run:
- `train_mnist.py` for training networks on MNIST
- `train_cifar10.py` for training networks on CIFAR-10
- `train_imagenet.py` for training networks on ImageNet

When running a script, a path to a directory where results will be stored must be provided:
```
python3 train_mnist.py path/to/results/directory
```

When training on ImageNet, the path to the dataset must be provided as a second argument:
```
python3 train_imagenet.py path/to/results/directory path/to/imagenet/dataset
```

Each script has a collection of arguments that can be provided to set training hyperparameters, which can be listed by using the `--help` argument:

```
python3 train_cifar10.py --help
```

## Demo
The Jupyter notebook `mnist_demo.ipynb` contains a simple demo for training on MNIST using learned feedback weights and observing weight alignment. To run the notebook, Jupyter must be installed (https://jupyter.org/install). The notebook should take around 10 minutes to run on a laptop.
