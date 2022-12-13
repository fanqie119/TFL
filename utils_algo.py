import torch
import torch.nn.functional as F
import torch.nn as nn


def ramp_loss(output, label):
    k_t = 2
    margin = 0.5
    class_mun = 4
    one_hot = F.one_hot(label.to(torch.int64), class_mun) * 2 - 1
    sig_out = output * one_hot
    y_label = torch.ones(sig_out.size())
    loss_min = torch.min(k_t*y_label, margin*k_t*y_label-sig_out)
    loss_sig = (1/k_t) * torch.max(0*y_label, loss_min)
    output = torch.mean(loss_sig)
    return output


def c_loss(output, label, device):
    loss = nn.MSELoss(size_average=True)
    class_mun = 4
    one_hot = F.one_hot(label.to(torch.int64), class_mun) * 2 - 1
    one_hot = one_hot.to(device)
    sig_out = output * one_hot
    sig_out = sig_out.to(device)
    y_label = torch.ones(sig_out.size())
    y_label = y_label.to(device)
    output = loss(sig_out, y_label)
    return output


def c_TF_loss(output_F, target_SL_F, device):
    label_0 = torch.zeros(target_SL_F.size())
    label_1 = torch.ones(target_SL_F.size())
    label_2 = torch.ones(target_SL_F.size()) * 2
    label_3 = torch.ones(target_SL_F.size()) * 3
    loss = c_loss(output_F, label_0.long(), device) + c_loss(output_F, label_1.long(), device) + c_loss(output_F, label_2.long(), device) + c_loss(output_F, label_3.long(), device) - c_loss(output_F, target_SL_F, device)
    return loss


def TF_risk(output, target_TF, target_SL, data_prior, device):
    class_mun = 4
    index_T = []
    index_F = []
    for i in range(target_TF.size()[0]):
        if target_TF[i] == 1:
            index_T.append(i)
        elif target_TF[i] == 0:
            index_F.append(i)
    index_T = torch.tensor(index_T).long()
    index_T = index_T.to(device)
    index_F = torch.tensor(index_F).long()
    index_F = index_F.to(device)
    target_SL_T = torch.index_select(target_SL, dim=0, index=index_T)
    target_SL_T = target_SL_T.to(device)
    target_SL_F = torch.index_select(target_SL, dim=0, index=index_F)
    target_SL_F = target_SL_F.to(device)
    output_T = torch.index_select(output, dim=0, index=index_T)
    output_F = torch.index_select(output, dim=0, index=index_F)
    loss = data_prior * c_loss(output_T, target_SL_T, device) + (1-data_prior)/(class_mun-1) * c_TF_loss(output_F, target_SL_F, device)
    return loss


def W_TL_loss(output, target_TF, target_SL, data_prior, device):
    class_mun = 4
    index_T = []
    index_F = []
    for i in range(target_TF.size()[0]):
        if target_TF[i] == 1:
            index_T.append(i)
        elif target_TF[i] == 0:
            index_F.append(i)
    index_T = torch.tensor(index_T).long()
    index_T = index_T.to(device)
    index_F = torch.tensor(index_F).long()
    index_F = index_F.to(device)
    target_SL_T = torch.index_select(target_SL, dim=0, index=index_T)
    target_SL_T = target_SL_T.to(device)
    target_SL_F = torch.index_select(target_SL, dim=0, index=index_F)
    target_SL_F = target_SL_F.to(device)
    output_T = torch.index_select(output, dim=0, index=index_T)
    output_F = torch.index_select(output, dim=0, index=index_F)
    loss = data_prior * c_loss(output_T, target_SL_T, device)
    return loss


def W_CL_loss(output, target_TF, target_SL, data_prior, device):
    class_mun = 4
    index_T = []
    index_F = []
    for i in range(target_TF.size()[0]):
        if target_TF[i] == 1:
            index_T.append(i)
        elif target_TF[i] == 0:
            index_F.append(i)
    index_T = torch.tensor(index_T).long()
    index_T = index_T.to(device)
    index_F = torch.tensor(index_F).long()
    index_F = index_F.to(device)
    target_SL_T = torch.index_select(target_SL, dim=0, index=index_T)
    target_SL_T = target_SL_T.to(device)
    target_SL_F = torch.index_select(target_SL, dim=0, index=index_F)
    target_SL_F = target_SL_F.to(device)
    output_T = torch.index_select(output, dim=0, index=index_T)
    output_F = torch.index_select(output, dim=0, index=index_F)
    loss = (1 - data_prior) / (class_mun - 1) * c_TF_loss(output_F, target_SL_F, device)
    return loss


def train_multi_split(args, model, train_loader, optimizer_kn_TF, optimizer_kn_CF, epoch, data_prior, data_set, device):
    model.train()  # 针对在网络train和eval时采用不同方式的情况，比如 BatchNormalization 和 Dropout
    for batch_idx, (data, target_TF, target_SL) in enumerate(train_loader):
        data = torch.tensor(data, dtype=torch.float32)
        data, target_TF, target_SL = data.to(device), target_TF.to(device), target_SL.to(device)
        optimizer_kn_TF.zero_grad()
        optimizer_kn_CF.zero_grad()
        if data_set == 'cifar10':
            data = data.transpose(1, 3)
        output = model(data)
        # loss = TF_risk(output, target_TF, target_SL, data_prior, device=device)
        loss1 = W_TL_loss(output, target_TF, target_SL, data_prior, device=device)
        loss1.backward(retain_graph=True)
        loss2 = W_CL_loss(output, target_TF, target_SL, data_prior, device=device)
        loss2.backward()
        optimizer_kn_TF.step()
        optimizer_kn_CF.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch： {} [{}/{} ({:.0f}%)]\tLoss_TF: {:.6f}\tLoss_CL: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss1.item(), loss2.item()))  # item获得一个元素tensor的value


def train_multi_split2(args, model, train_loader, optimizer_kn, epoch, data_prior, data_set, device):
    model.train()  # 针对在网络train和eval时采用不同方式的情况，比如 BatchNormalization 和 Dropout
    for batch_idx, (data, target_TF, target_SL) in enumerate(train_loader):
        data = torch.tensor(data, dtype=torch.float32)
        data, target_TF, target_SL = data.to(device), target_TF.to(device), target_SL.to(device)
        optimizer_kn.zero_grad()
        if data_set == 'cifar10':
            data = data.transpose(1, 3)
        output = model(data)
        loss = TF_risk(output, target_TF, target_SL, data_prior, device=device)
        loss.backward()
        optimizer_kn.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch： {} [{}/{} ({:.0f}%)]\tLoss_TF: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))  # item获得一个元素tensor的value
