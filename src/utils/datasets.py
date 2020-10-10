from __future__ import print_function

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

mean = {
  'cifar10': (0.4914, 0.4822, 0.4465),
  'cifar100': (0.5071, 0.4867, 0.4408),
  'svhn': (0.5, 0.5, 0.5),
  'fmnist': (0.5,),
}
std = {
  'cifar10': (0.2023, 0.1994, 0.2010),
  'cifar100': (0.2675, 0.2565, 0.2761),
  'svhn': (0.5, 0.5, 0.5),
  'fmnist': (0.5,),
}


def get_data(args):
  """ Applies general preprocess transformations to Datasets
      ops -
      transforms.Normalize = Normalizes each channel of the image
      args - mean tensor, standard-deviation tensor
      transforms.RandomCrops - crops the image at random location
      transforms.HorizontalFlip - randomly flips the image
  """
  if args.mode == 'ood':
    dataset = args.target_dataset
  else:
    dataset = args.dataset

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean[dataset], std[dataset]),
  ])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean[dataset], std[dataset])
  ])

  if dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
      testset, batch_size=100, shuffle=False, num_workers=4)
    args.num_classes = 10

  elif dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(
      root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.SVHN(
      root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
      testset, batch_size=100, shuffle=False, num_workers=4)
    args.num_classes = 10

  elif dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
      root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
      root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
      testset, batch_size=100, shuffle=False, num_workers=4)
    args.num_classes = 100

  elif dataset == 'fmnist':
    transform = transforms.Compose(
      [transforms.Resize(96),
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(
      root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.FashionMNIST(
      root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
      testset, batch_size=100, shuffle=False, num_workers=4)
    args.num_classes = 10

  return trainloader, testloader
