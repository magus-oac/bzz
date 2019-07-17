import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

from nets import ZZBNet
from mydataset import load_data_list, data_dic


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = ZZBNet().to(device)
    model.load_state_dict(torch.load('./zzbnet_best.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    from mydataset import num_time_steps
    rst = ''
    l = load_data_list(data_dic + '/submission_format.csv')
    for i, (im_id, _) in enumerate(l):
        im = Image.open(data_dic + '/test/' + str(im_id) + '.png').convert('L')
        im = transform(im)
        # im = torch.unsqueeze(im, 0)

        l = []
        step = 173 // num_time_steps
        for j in range(num_time_steps):
            l.append(im[:, :, j * step:(j + 1) * step])

        im = torch.stack(l, dim=0)
        im = im.unsqueeze(0)
        im = im.to(device)

        cls_score = model(im)
        cls_score = F.softmax(cls_score)
        cls = torch.argmax(cls_score).item()
        rst += str(im_id) + ',' + str(cls) + '\n'
        if i % 100 == 0:
            print(i / len(l))

    with open('./rst.csv', 'w') as f:
        f.write('file_id,accent\n')
        f.write(rst)


if __name__ == '__main__':
    main()
