import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  #每过step_size个epoch，做一次学习率更新
from utils_data import *
from utils_algo import *
from test import *
from models.Resnet import ResNet
from models.Cnn import Cnn
from models.Network import Net
import time

parser = argparse.ArgumentParser(description='Pytorch MNIST Example')  # ArgumentParser类生成一个对象，description描述信息
parser.add_argument('--batch-size', type=int, default=128, metavar='N',  # 增加参数
                    help='input batch size for training (default:64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default:1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',  # 避免跑太久，先设为1
                    help='number of eopchs to train （default:14)')  # help用来描述这个选项的作用
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default:1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--data_set', type=str, choices=['mnist', 'cifar10'], default='mnist', metavar='N',)
parser.add_argument('--method_choice', type=int, default=1, metavar='N',
                    help='TFL or CLPC')
parser.add_argument('--trials_choice', type=int, default=1, metavar='N',
                    help='number of iter')

args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

K = 4

max_test_acc = torch.zeros(args.trials_choice)
for i_train_iter in range(args.trials_choice):
    if args.data_set == 'mnist':
        new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label, data_prior = read_data_minist_TF(
            args.data_set, i_train_iter)  # read data 1:minist 2:fashion_minist
        if args.method_choice == 1:
            model = Net(n_outputs=K)
            if i_train_iter == 0:
                parser.add_argument('--lr_known', type=float, default=7e-2, metavar='LR_m',  # metavar用在help信息的输出中
                                    help='learning rate for known(default:1.0)')
                parser.add_argument('--known_gamma', type=float, default=0.05, metavar='M',
                                    help='Learning rate step gamma (default:0.5)')
                parser.add_argument('--step_size', type=int, default=20, metavar='N',
                                    help='size')
        elif args.method_choice == 2:
            model = Net(n_outputs=K)
            if i_train_iter == 0:
                parser.add_argument('--lr_known', type=float, default=7e-2, metavar='LR_m',  # metavar用在help信息的输出中
                                    help='learning rate for known(default:1.0)')
                parser.add_argument('--known_gamma', type=float, default=0.1, metavar='M',
                                    help='Learning rate step gamma (default:0.5)')
                parser.add_argument('--step_size', type=int, default=20, metavar='N',
                                    help='size')
    elif args.data_set == 'cifar10':
        new_train_set, new_train_TF_label, new_train_SL_label, new_test_set, new_test_label, data_prior = read_data_cifar10_TF(i_train_iter)
        input_channels = 3
        dropout_rate = 0.25
        K = 4
        model = Cnn(input_channels=input_channels, n_outputs=K, dropout_rate=dropout_rate)
        parser.add_argument('--lr_known', type=float, default=7e-2, metavar='LR_m',  # metavar用在help信息的输出中
                            help='learning rate for known(default:1.0)')
        parser.add_argument('--known_gamma', type=float, default=0.1, metavar='M',
                            help='Learning rate step gamma (default:0.5)')
        parser.add_argument('--step_size', type=int, default=20, metavar='N',
                            help='size')

    args = parser.parse_args()  # parser对象的parse_args()获取解析的参数

    model = model.to(device)
    print(model)

    optimizer_known_TF = optim.Adadelta(model.parameters(), lr=args.lr_known)
    scheduler_known_TF = StepLR(optimizer_known_TF, step_size=args.step_size, gamma=args.known_gamma)  ####0.98

    optimizer_known_CF = optim.Adadelta(model.parameters(), lr=7e-2)
    scheduler_known_CF = StepLR(optimizer_known_TF, step_size=20, gamma=0.9)  ####0.98

    print(
        'data choice: {}\n 1:minist 2:SVHN 3:cifar \n mnist data set: {} \n 1: minist 2: fashion minsit 3: Kuzushiji_minist\n Train Parameter：step size {}, known gamma {}, known lr {}\n model chioce: {}'.format(
            args.data_set, args.data_set, args.step_size, args.known_gamma, args.lr_known,
            args.method_choice))  # item获得一个元素tensor的value

    train_loader, test_loader = data_gen(new_train_set, new_train_TF_label, new_train_SL_label, new_test_set,
                                         new_test_label)

    test_acc = torch.zeros(args.epochs)
    test_tpr = torch.zeros(args.epochs)
    for epoch in range(1, args.epochs + 1):

        start = time.perf_counter()

        if args.method_choice == 1:
            train_multi_split(args, model, train_loader, optimizer_known_TF, optimizer_known_CF, epoch, data_prior, data_set=args.data_set, device=device)
        elif args.method_choice == 2:
            train_multi_split2(args, model, train_loader, optimizer_known_TF, epoch, 1, data_set=args.data_set, device=device)
        test_acc[epoch - 1] = test_multi(model, test_loader, data_set=args.data_set, device=device)
        scheduler_known_TF.step()
        if args.method_choice == 1:
            scheduler_known_CF.step()

        end = time.perf_counter()

        print('Running time: %s Seconds' % (end - start))
    max_test_acc[i_train_iter] = torch.max(test_acc, 0)[0]

acc_mean = torch.mean(test_acc)
acc_std = torch.std(test_acc)


print(max_test_acc)
# print(acc_mean, acc_std)
