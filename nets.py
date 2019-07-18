import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch


class MyBlock(nn.Module):
    def __init__(self, n):
        super(MyBlock, self).__init__()
        a = (n+n//8)//3
        self.conv1 = MyConv(n, a-n//8, [3, 3], 1, 1, is_pooling=False)
        self.conv2 = MyConv(n, a, [5, 5], 1, 2, is_pooling=False)
        self.conv3 = MyConv(n, a, [7, 7], 1, 3, is_pooling=False)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        # print(y1.size(), y2.size(), y3.size())
        y = torch.cat([y1, y2, y3], 1)
        return y


class MyConv(nn.Module):
    """
    H = H_in + 2p[0] -kernel_size[0] + 1
    W = W_in + 2p[1] -kernel_size[1] + 1
    H /= 2
    W /= 2
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, is_pooling=True):
        super(MyConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.is_pooling = is_pooling
        if is_pooling:
            self.pooling = nn.MaxPool2d([2, 2])
        self._initialize_weights()

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        if self.is_pooling:
            y = self.pooling(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ZZBNet_conv1d(nn.Module):
    def __init__(self):
        super(ZZBNet, self).__init__()
        # self.conv1 = MyConv(1, 16, [3, 3], 1, 1, is_pooling=False)
        self.conv1 = MyConv(1, 16, [4, 4], [2, 1], [0, 1], is_pooling=False)
        self.conv2 = MyConv(16, 32, [4, 4], [4, 1], [0, 1], is_pooling=False)
        self.conv3 = MyConv(32, 16, [4, 4], [4, 1], [0, 1], is_pooling=False)
        self.conv4 = MyConv(16, 1, [3, 3], [4, 1], [0, 1], is_pooling=False)

        self.conv5 = torch.nn.Conv1d(1, 1, 3, 3, 0)
        self.conv6 = torch.nn.Conv1d(1, 1, 3, 3, 0)
        self.conv7 = torch.nn.Conv1d(1, 1, 3, 3, 0)

        # self.fc1 = nn.Linear(16*21*64, 1024)
        self.fc1 = nn.Linear(1*6, 3)
        # self.drp1 = nn.Dropout(0.8)
        # self.fc2 = nn.Linear(1024, 3)
        # self.drp2 = nn.Dropout(0.8)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.squeeze(2)
        # x = x.unsqueeze(1)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        # print(x.size())
        x = x.view(-1, 6)

        # x = self.drp1(x)
        x = self.fc1(x)
        # x = F.relu(x)

        # self.drp2(x)
        # x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ZZBNet(nn.Module):
    def __init__(self):
        super(ZZBNet, self).__init__()
        torchvision.models.inception_v3()
        self.conv1 = MyConv(1, 16, [3, 3], 1, 1, is_pooling=True)
        self.conv2 = MyConv(16, 32, [3, 3], 1, 1, is_pooling=True)

        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv3 = MyConv(32, 64, [3, 3], 1, 1, is_pooling=True)
        # self.conv4 = MyConv(64, 64, [3, 3], 1, 1, is_pooling=False)
        self.conv4 = MyBlock(64)

        self.bn2 = torch.nn.BatchNorm2d(64)

        self.conv5 = MyConv(64, 128, [3, 3], 1, 1, is_pooling=True)
        # self.conv6 = MyConv(128, 128, [3, 3], 1, 1, is_pooling=False)
        self.conv6 = MyBlock(128)

        self.bn3 = torch.nn.BatchNorm2d(128)

        self.conv7 = MyConv(128, 256, [3, 3], 1, 1, is_pooling=False)
        # self.conv8 = MyConv(256, 256, [3, 3], 1, 1, is_pooling=False)
        self.conv8 = MyBlock(256)

        self.bn4 = torch.nn.BatchNorm2d(256)

        self.conv9 = MyConv(256, 512, [3, 3], 1, 1, is_pooling=False)
        self.conv10 = MyConv(512, 512, [3, 3], 1, [1, 0], is_pooling=True)

        # self.fc1 = nn.Linear(16*21*64, 1024)
        self.fc1 = nn.Linear(4 * 4 * 512, 1024)
        self.drp1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, 3)
        self.drp2 = nn.Dropout(0.6)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.bn3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.bn4(x)

        x = self.conv9(x)
        x = self.conv10(x)
        # print(x.size())
        # x = x.view(-1, 16*21*64)
        # print(x.size())
        x = x.view(-1, 4*4*512)

        x = self.drp1(x)
        x = self.fc1(x)
        x = F.relu(x)

        self.drp2(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ZZBNet_conv5(nn.Module):
    def __init__(self):
        super(ZZBNet_conv5, self).__init__()
        # self.conv1 = MyConv(1, 16, [3, 3], 1, 1, is_pooling=False)
        self.conv1 = MyConv(1, 16, [3, 3], 1, 1)
        self.conv2 = MyConv(16, 32, [3, 3], 1, 1)
        self.conv3 = MyConv(32, 64, [3, 3], 1, 1)
        self.conv4 = MyConv(64, 128, [3, 3], 1, 1)
        self.conv5 = MyConv(128, 256, [3, 3], 1, [1, 0])

        # self.fc1 = nn.Linear(16*21*64, 1024)
        self.fc1 = nn.Linear(4 * 4 * 256, 1024)
        self.drp1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, 3)
        self.drp2 = nn.Dropout(0.6)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = x.view(-1, 16*21*64)
        x = x.view(-1, 4*4*256)

        x = self.drp1(x)
        x = self.fc1(x)
        x = F.relu(x)

        self.drp2(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ZZBNetvgg(nn.Module):
    def __init__(self, init_weights=False):
        super(ZZBNetvgg, self).__init__()
        self.vgg = torchvision.models.vgg13_bn(pretrained=True)

        self.vgg.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

        # if init_weights:
        #     self._initialize_weights()
        # self._initialize_weights()

    def forward(self, x):
        x = self.vgg(x)
        print(x.size())
        x = self.classifer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ZZBNet1(nn.Module):
    def __init__(self, pretrained=True):
        super(ZZBNet1, self).__init__()
        self.resnet = torchvision.models.resnet50(num_classes=3)

        self._initialize_weights()

        if pretrained:
            # load part
            pretrained_dict = torch.load('/home/waxnkw/.torch/models/resnet50-19c8e357.pth')
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ZZBNet_rnn(nn.Module):
    def __init__(self, batch=64):
        super(ZZBNet_rnn, self).__init__()
        self.conv1 = MyConv(1, 16, [3, 3], 1, 1)
        self.conv2 = MyConv(16, 32, [3, 3], 1, 1)
        self.conv3 = MyConv(32, 64, [3, 3], 1, 1)
        self.conv4 = MyConv(64, 128, [3, 3], 1, 1)

        # self.fc1 = nn.Linear(16*21*64, 1024)
        self.fc1 = nn.Linear(2 * 8 * 128, 128)
        self.drp1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, 3)
        self.drp2 = nn.Dropout(0.2)

        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.lengths = [torch.tensor(128) for i in range(batch)]

        self._initialize_weights()

    def forward(self, x):
        batch, seq_len, channel, h, w = x.size()
        x = x.view([-1, channel, h, w])

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # x = x.view(-1, 16*21*64)
        x = x.view(-1, 2*8*128)

        x = self.drp1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = x.view([batch, seq_len, -1])
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths=self.lengths, batch_first=True)
        x, (hn, cn) = self.rnn(x)
        x = x[:, -1, :]

        x = self.drp2(x)
        x = self.fc2(x)
        # print(x.size())
        # x = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # print(hn.squeeze(0).size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param)
