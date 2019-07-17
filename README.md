# bzz

## 最好结果
10 层 conv [3x3] kernel

## 尝试过的:    
5层，和10层差不了多少   
沿时间轴切成5片4层cnn后lstm,效果和10层差不多，也许切成10片会有效果???    
将非时间轴用stride=4的conv压缩到1后，跟conv1d: 完全无效。 接个rnn可能还有点用？   

## 未尝试：
conv10层里面加点batchnorm？ 
dialtion，估计没用，因为没啥道理？   
