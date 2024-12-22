import torch
import torch.nn as nn
import torch.nn.functional as F
# This is the draft code of our ESA block, we provide it here for those who are eager to know
# the implementation details. The official version will be released later.


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
       if not padding and stride==1:
           padding = kernel_size // 2
       return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
       
class ESALayer(nn.Module):
     def __init__(self, n_feats, conv=default_conv):
         super(ESALayer, self).__init__()
         f = n_feats // 4
         self.conv1 = conv(n_feats, f, kernel_size=1)
         self.conv_f = conv(f, f, kernel_size=1)
         self.conv_max = conv(f, f, kernel_size=3, padding=1)
         self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
         self.conv3 = conv(f, f, kernel_size=3, padding=1)
         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
         self.conv4 = conv(f, n_feats, kernel_size=1)
         self.sigmoid = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)
  
     def forward(self, x, f): 
         c1_ = (self.conv1(f))  # 1*1卷积，分辨率不变
         c1 = self.conv2(c1_)   # 分辨率减半
         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # 
         v_range = self.relu(self.conv_max(v_max))  # 分辨率不变
         c3 = self.relu(self.conv3(v_range)) # 分辨率不变
         c3 = self.conv3_(c3) # 分辨率不变
         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear') # 回到输入的分辨率
         cf = self.conv_f(c1_)  # 1*1卷积，分辨率不变
         c4 = self.conv4(c3+cf) # 1*1卷积，分辨率不变
         m = self.sigmoid(c4)
         
         return x * m
