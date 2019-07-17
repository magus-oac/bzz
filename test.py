import torch
from skimage import io
import numpy as np
# x = torch.tensor([1.,2.,3.])
# x = x.unsqueeze(0)
# y = torch.tensor([1.,2.,3.])+1
# y = y.unsqueeze(0)
# x = torch.stack([x,y],0)
#
# length = torch.tensor([3, 3])
# print(x.size())
# y = torch.nn.utils.rnn.pack_padded_sequmence(x, lengths=length, batch_first=True)
# print(y)

x = io.imread('/usr/local/cv/zzb/data/train/10021.png', as_gray=True)
# print(x)

s = ''
for i in range(128):
    for j in range(173):
        s += str(round(x[i,j],1))+' '
    s += '\n'

for i in range(128):
    print(x[i, 90])

