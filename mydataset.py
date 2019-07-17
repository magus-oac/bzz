import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import pandas as pd
import json

# data_dic = '/home/magus/zhanga-workspace/zzb/data/'
data_dic = '/usr/local/cv/zzb/data/'
train_dic = data_dic + 'train/'

num_time_steps = 5


def load_data_list(fname):
    if '.json' in fname:
        with open(fname, 'r') as f:
            return json.load(f)
    elif '.csv' in fname:
        l = pd.read_csv(fname)
        return l.values.tolist()


class ZZBDataset(torch.utils.data.Dataset):

    def __init__(self, base_dic, data_file_name, aug=False, test_mode=False):
        self.base_dic = base_dic
        self.data_list = load_data_list(data_file_name)
        # self.scale_aug = scale_aug
        # self.rot_aug = rot_aug
        self.aug = aug
        self.test_mode = test_mode

        self.transform = transforms.Compose([
            # transforms.Resize([224,224]),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.288], std=[0.231])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        im_id, class_id = self.data_list[idx]
        # Use PIL instead
        im = Image.open(self.base_dic+str(im_id)+'.png').convert('L')

        # im = im[:, :, 0:3]
        if self.transform:
            im = self.transform(im)
        #
        # l = []
        # step = 173//num_time_steps
        # for i in range(num_time_steps):
        #     l.append(im[:, :, i*step:(i+1)*step])
        #
        # im = torch.stack(l, dim=0)

        return im, class_id
