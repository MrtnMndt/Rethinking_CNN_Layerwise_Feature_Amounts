import os
import struct
import gzip
import errno

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math


class IMAGENET:
    """
    Actually just a variant of CUSTOM dataset and uses ImageFolder
    for any data-loading. Main difference is in definition of transforms
    and pre-calculated preprocessing parameters for 1000 class ILSVRC
    2012 dataset.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int), workers(int)
            and two paths train_data (str) and val_data (str)
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        traindir (str): Path to train dataset as required by
            torchvision.datasets.ImageFolder
        valdir (str): Path to validation dataset as required by
            torchvision.datasets.ImageFolder
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            crops of size 224 x 224 and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, scaling of shorter side to
            256 followed by a center crop of 224 x 224 and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.valdir = args.val_data
        self.traindir = args.train_data
        self.num_classes = 1000

        if args.normalize_data:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        else:
            self.normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                  std=[1.0, 1.0, 1.0])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Resize(patch_size + 32),
            transforms.RandomSizedCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Resize(patch_size + 32),
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.ImageFolder to load dataset
        from traindir and valdir respectively.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        trainset = datasets.ImageFolder(self.traindir, self.train_transforms)
        valset = datasets.ImageFolder(self.valdir, self.val_transforms)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR10:
    """
    CIFAR-10 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader adapted from CIFAR10.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):

        self.num_classes = 10

        if args.normalize_data:
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                  std=[0.2023, 0.1994, 0.2010])
        else:
            self.normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                  std=[1.0, 1.0, 1.0])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR10 to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR10('datasets/CIFAR10/train/', train=True, transform=self.train_transforms,
                                    target_transform=None, download=True)
        valset = datasets.CIFAR10('datasets/CIFAR10/test/', train=False, transform=self.val_transforms,
                                  target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR100:
    """
    CIFAR-100 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader adapted from CIFAR10.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):

        self.num_classes = 100

        if args.normalize_data:
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                                  std=[0.2009, 0.1984, 0.2023])
        else:
            self.normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                  std=[1.0, 1.0, 1.0])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self, patch_size):
        train_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.RandomCrop(patch_size, int(math.ceil(patch_size * 0.1))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
            ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR100 to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR100('datasets/CIFAR100/train/', train=True, transform=self.train_transforms,
                                     target_transform=None, download=True)
        valset = datasets.CIFAR100('datasets/CIFAR100/test/', train=False, transform=self.val_transforms,
                                   target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class MNIST:
    """
    MNIST dataset featuring gray-scale 28x28 images of
    hand-written characters belonging to ten different classes.
    Dataset implemented with torchvision.datasets.MNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):

        self.num_classes = 10

        if args.normalize_data:
            self.normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        else:
            self.normalize = transforms.Normalize(mean=[0.0], std=[1.0])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision MNIST data loader does not provide class_to_idx
        self.val_loader.dataset.class_to_idx = {'0': 0,
                                                '1': 1,
                                                '2': 2,
                                                '3': 3,
                                                '4': 4,
                                                '5': 5,
                                                '6': 6,
                                                '7': 7,
                                                '8': 8,
                                                '9': 9}

    def __get_transforms(self, patch_size):
        # scale the images (e.g. to 32x32, so the same model
        # as for CIFAR10 can be used for comparison
        # for analogous reasons we also define a lambda transform
        # to duplicate the gray-scale image to 3 channels
        train_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.MNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.MNIST('datasets/MNIST/train/', train=True, transform=self.train_transforms,
                                  target_transform=None, download=True)
        valset = datasets.MNIST('datasets/MNIST/test/', train=False, transform=self.val_transforms,
                                target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class FashionMNIST:
    """
    FashionMNIST dataset featuring gray-scale 28x28 images of
    Zalando clothing items belonging to ten different classes.
    Dataset implemented with torchvision.datasets.FashionMNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        normalize (dict): Contains per-channel means and stds of the dataset.
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):

        self.num_classes = 10

        if args.normalize_data:
            self.normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        else:
            self.normalize = transforms.Normalize(mean=[0.0], std=[1.0])

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self, patch_size):
        # scale the images (e.g. to 32x32, so the same model
        # as for CIFAR10 can be used for comparison
        # for analogous reasons we also define a lambda transform
        # to duplicate the gray-scale image to 3 channels
        train_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
            self.normalize,
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.FashionMNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.FashionMNIST('datasets/FashionMNIST/train/', train=True, transform=self.train_transforms,
                                         target_transform=None, download=True)
        valset = datasets.FashionMNIST('datasets/FashionMNIST/test/', train=False, transform=self.val_transforms,
                                       target_transform=None, download=True)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class KMNIST(Dataset):
    """
    KuzushijiMNIST dataset featuring gray-scale 28x28 images of
    ten selected Japanese Hiragana from old literature manuscripts.

    https://github.com/rois-codh/kmnist
    Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. arXiv:1812.01718

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.__path = os.path.expanduser('datasets/KuzushijiMNIST')
        self.__download()

        self.trainset, self.valset = self.get_dataset(args.patch_size, args.normalize_data)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __check_exists(self):
        """
        Check if dataset has already been downloaded

        Returns:
             bool: True if downloaded dataset has been found
        """

        return os.path.exists(os.path.join(self.__path, 'train-images-idx3-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 'train-labels-idx1-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 't10k-images-idx3-ubyte.gz')) and \
               os.path.exists(os.path.join(self.__path, 't10k-labels-idx1-ubyte.gz'))

    def __download(self):
        """
        Downloads the KMNIST dataset from the web if dataset
        hasn't already been downloaded.
        """
        from six.moves import urllib

        if self.__check_exists():
            return

        print("Downloading KMNIST dataset")

        urls = [
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
            'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz',
        ]

        # download files
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.__path, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        print('Done!')

    def __get_kmnist(self, path, kind='train'):
        """
        Load Kuzushiji-MNIST data

        Parameters:
            path (str): Base directory path containing .gz files for
                the Kuzushiji-MNIST dataset
            kind (str): Accepted types are 'train' and 't10k' for
                training and validation set stored in .gz files

        Returns:
            numpy.array: images, labels

        """
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            struct.unpack('>II', lbpath.read(8))
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            struct.unpack(">IIII", imgpath.read(16))
            images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    def get_dataset(self, patch_size, normalize):
        """
        Loads and wraps training and validation datasets

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        x_train, y_train = self.__get_kmnist(self.__path, kind='train')
        x_val, y_val = self.__get_kmnist(self.__path, kind='t10k')

        # convert to torch tensors in range [0, 1]
        x_train = torch.from_numpy(x_train).float() / 255
        y_train = torch.from_numpy(y_train).long()
        x_val = torch.from_numpy(x_val).float() / 255
        y_val = torch.from_numpy(y_val).long()

        # resize flattened array of images for input to a CNN
        x_train.resize_(x_train.size(0), 1, 28, 28)
        x_val.resize_(x_val.size(0), 1, 28, 28)

        if normalize:
            mean = 0.190362019974657
            std = 0.3474631837640068
            x_train = (x_train - mean) / std
            x_val = (x_val - mean) / std

        # up and downsampling
        x_train = torch.nn.functional.interpolate(x_train, size=patch_size, mode='bilinear')
        x_val = torch.nn.functional.interpolate(x_val, size=patch_size, mode='bilinear')

        # TensorDataset wrapper
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader
