"""
Command line argument options parser.

Usage with two minuses "- -". Options are written with a minus "-" in command line, but
appear with an underscore "_" in the attributes' list.

Attributes:
    train (str): Path to train dataset directory
    val (str): Path to val dataset directory
    dataset (str): Name of dataset (default: CUSTOM)
    workers (int): Number of data loading workers (default: 4)
    normalize_data (bool): Turns on dataset normalization (default: False)
    patch_size (int): patch size for image crops (default: 32)
    architecture (str): Model architecture (default: cnn_maxpool_bn)
    weight_init (str): Weight-initialization scheme (default: kaiming-normal)
    vgg_depth (int): Amount of layers in VGG architecture (default: 16 -> VGG-D)
    epochs (int): Number of epochs to train (default: 30)
    batch_size (int): Mini-batch size (default: 128)
    learning_rate (float): Initial learning rate (default: 0.1)
    lr_wr_epochs (int): epochs defining one warm restart cycle (default: 10)
    lr_wr_mul (int): factor to grow warm restart cycle length after each cycle (default: 2)
    lr_wr_min (float): minimum learning rate to use in warm restarts (default: 1e-5)
    momentum (float): Momentum value (default: 0.9)
    nesterov (bool): Turns on nesterov momentum
    weight_decay (float): L2-norm/weight-decay value (default: 0.0005)
    batch_norm (float): Batch normalization value (default: 0.001)
    print_freq (int): Print frequency in amount of batches (default: 100)
"""

import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset and loading
parser.add_argument('-train', '--train-data', metavar='TRAINDIR',
                    help='path to (train)dataset (for ImageNet)')
parser.add_argument('-val', '--val-data', metavar='VALDIR',
                    help='path to test-dataset (for ImageNet)')
parser.add_argument('--dataset', default='CIFAR10',
                    help='name of dataset')
parser.add_argument('--normalize-data', default=False, type=bool,
                    help='turns on dataset normalization')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--patch-size', default=32, type=int,
                    metavar='P', help='patch size for crops')

# Architecture and weight-init
parser.add_argument('-a', '--architecture', metavar='ARCH', default='VGG',
                    help='model architecture (default: VGG)')
parser.add_argument('--weight-init', default='kaiming-normal', metavar='W',
                    help='weight-initialization scheme (default: kaiming-normal)')
parser.add_argument('--vgg-depth', default=16, type=int,
                    help='amount of layers in the vgg network (default: 16 -> VGG-D)')

# Training hyper-parameters
parser.add_argument('--resume-model-id', default=0, type=int,
                    help='number of model to resume from')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default 0.9)')
parser.add_argument('--nesterov', default=False, type=bool,
                    help='turns on nesterov momentum')
parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-bn', '--batch-norm', default=1e-4, type=float,
                    metavar='BN', help='batch normalization (default 1e-4)')
parser.add_argument('-pf', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

# Learning rate schedules
parser.add_argument('--lr-wr-epochs', default=10, type=int,
                    help='epochs defining one warm restart cycle')
parser.add_argument('--lr-wr-mul', default=2, type=int,
                    help='factor to increase warm restart cycle epochs after restart')
parser.add_argument('--lr-wr-min', default=1e-5, type=float,
                    help='minimum learning rate used in each warm restart cycle')
