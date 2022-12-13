import torch
import os
import torchvision.datasets.mnist as mnist
from torch.utils.data import TensorDataset, DataLoader
import random
import torchvision.transforms as transforms
import torchvision.datasets as dsets


def data_gen(new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label):
    train_dataset = TensorDataset(new_train_set, new_train_TF_label, new_train_SL_label)
    train_loader =  DataLoader(dataset=train_dataset,
                              batch_size = 256,
                              shuffle = True)

    test_dataset = TensorDataset(new_test_set, new_test_label)
    test_loader =  DataLoader(dataset=test_dataset,
                              batch_size = 128,
                              shuffle=True)
    return train_loader, test_loader


def read_data_minist_TF(data_set, i_train_iter):
    num_data = 160000
    random.seed(i_train_iter)
    P1 = 7
    P2 = 8
    P3 = 9
    P4 = 1
    if data_set == 'mnist':
        root="./data/mnist/MNIST/raw"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    elif data_set == 'fashion':
        root = "D:/code/data/fashion_minist/fashion_minist/"
        root = "./data/FashionMNIST/raw"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
    elif data_set == 'kuzushiji':
        root = "D:/code/data/Kuzushiji_minist/Kuzushiji_minist/"
        train_set = (
            mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )



    train_label = torch.tensor(train_set[1])
    test_label = torch.tensor(test_set[1])
    train_data = torch.tensor(train_set[0])
    test_data = torch.tensor(test_set[0])
    num_data_t = 0
    index = []
    for i in range(train_label.size()[0]):
        if (train_label[i] == P1  or train_label[i] == P2 or train_label[i] == P3 or train_label[i] == P4) and num_data_t <= num_data:
            index.append(i)
            num_data_t = num_data_t + 1

    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_TF_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_SL_label_index = torch.index_select(train_label, dim=0, index=index)

    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P1:
            new_train_label[i] = 0
        if new_train_label_index[i] == P2:
            new_train_label[i] = 1
        if new_train_label_index[i] == P3:
            new_train_label[i] = 2
        if new_train_label_index[i] == P4:
            new_train_label[i] = 3


    for i in range(new_train_set.size()[0]):
        r = random.randint(0, 3)
        # print('r = ', r)
        if new_train_label[i] == r:
            new_train_TF_label_index[i] = 1
            new_train_SL_label_index[i] = r
        else:
            new_train_TF_label_index[i] = 0
            new_train_SL_label_index[i] = r

    num_ALL = new_train_set.size()[0]
    num_t_label = torch.eq(new_train_TF_label_index, 1).sum()
    data_prior = num_t_label.float()/num_ALL

    print('data_prior', data_prior)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P1 or test_label[i] == P2 or test_label[i] == P3 or test_label[i] == P4:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P1:
            new_test_label[i] = 0
        if new_test_label_index[i] == P2:
            new_test_label[i] = 1
        if new_test_label_index[i] == P3:
            new_test_label[i] = 2
        if new_test_label_index[i] == P4:
            new_test_label[i] = 3

    return new_train_set.unsqueeze(1), new_train_TF_label_index, new_train_SL_label_index, new_test_set.unsqueeze(1), new_test_label, data_prior


def read_data_cifar10_TF(i_train_iter):
    num_data = 160000
    random.seed(i_train_iter)
    P1 = 7
    P2 = 8
    P3 = 9
    P4 = 1

    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_set = dsets.CIFAR10(root='./data/cifar10', train=True, transform=train_transform, download=True)
    test_set = dsets.CIFAR10(root='./data/cifar10', train=False, transform=test_transform)

    train_label = torch.tensor(train_set.targets)
    test_label = torch.tensor(test_set.targets)
    train_data = torch.tensor(train_set.data)
    test_data = torch.tensor(test_set.data)
    num_data_t = 0
    index = []
    for i in range(train_label.size()[0]):
        if (train_label[i] == P1 or train_label[i] == P2 or train_label[i] == P3 or train_label[i] == P4)\
                and num_data_t <= num_data:
            index.append(i)
            num_data_t = num_data_t + 1

    index = torch.tensor(index)
    new_train_set = torch.index_select(train_data, dim=0, index=index)
    new_train_label = torch.index_select(train_label, dim=0, index=index)
    new_train_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_TF_label_index = torch.index_select(train_label, dim=0, index=index)
    new_train_SL_label_index = torch.index_select(train_label, dim=0, index=index)

    for i in range(new_train_set.size()[0]):
        if new_train_label_index[i] == P1:
            new_train_label[i] = 0
        if new_train_label_index[i] == P2:
            new_train_label[i] = 1
        if new_train_label_index[i] == P3:
            new_train_label[i] = 2
        if new_train_label_index[i] == P4:
            new_train_label[i] = 3

    for i in range(new_train_set.size()[0]):
        r = random.randint(0, 2)
        # print('r = ', r)
        if new_train_label[i] == r:
            new_train_TF_label_index[i] = 1
            new_train_SL_label_index[i] = r
        else:
            new_train_TF_label_index[i] = 0
            new_train_SL_label_index[i] = r

    num_ALL = new_train_set.size()[0]
    num_t_label = torch.eq(new_train_TF_label_index, 1).sum()
    data_prior = num_t_label.float() / num_ALL

    print('data_prior', data_prior)
    # print(index)
    # print(new_train_label.size())
    index = []
    for i in range(test_data.size()[0]):
        if test_label[i] == P1 or test_label[i] == P2 or test_label[i] == P3:
            index.append(i)
    index = torch.tensor(index)
    new_test_set = torch.index_select(test_data, dim=0, index=index)
    new_test_label = torch.index_select(test_label, dim=0, index=index)
    new_test_label_index = torch.index_select(test_label, dim=0, index=index)
    for i in range(new_test_set.size()[0]):
        if new_test_label_index[i] == P1:
            new_test_label[i] = 0
        if new_test_label_index[i] == P2:
            new_test_label[i] = 1
        if new_test_label_index[i] == P3:
            new_test_label[i] = 2
        # if new_test_label_index[i] == P4:
        #     new_test_label[i] = 3

    return new_train_set, new_train_TF_label_index, new_train_SL_label_index, new_test_set, new_test_label, data_prior
