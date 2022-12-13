import torch
from utils_algo import *


def test_multi(model,  test_loader, data_set, device):
    model.eval()    # 测试模式
    test_loss = 0   # torch.zeros([6, 1])
    correct = 0     # torch.zeros([6, 1])
    with torch.no_grad():  # 数据不需要计算梯度，也不会进行反向传播
        for data, target in test_loader:
            data = torch.tensor(data, dtype=torch.float32)
            data, target = data.to(device), target.to(device)
            if data_set == 'cifar10':
                data = data.transpose(1, 3)
            output = model(data)
            test_loss += c_loss(output, target.long(), device).item()  # sum up batch loss
            # test_loss += torch.mean(train.c_loss(output, target.long()))
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = pred.to(device)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss = test_loss / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

