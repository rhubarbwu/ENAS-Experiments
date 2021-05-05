import torch
from torch import nn
import torch.nn.functional as F


class FactorizedReduction(nn.Module):
    '''
    Reduce both spatial dimensions (width and height) by a factor of 2, and 
    potentially to change the number of output filters
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L129
    '''
    def __init__(self, in_planes, out_planes, stride=2):
        super(FactorizedReduction, self).__init__()

        assert out_planes % 2 == 0, (
            "Need even number of filters when using this factorized reduction."
        )

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        if stride == 1:
            self.fr = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes, track_running_stats=False))
        else:
            self.path1 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes,
                          out_planes // 2,
                          kernel_size=1,
                          bias=False))

            self.path2 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes,
                          out_planes // 2,
                          kernel_size=1,
                          bias=False))
            self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        if self.stride == 1:
            return self.fr(x)
        else:
            path1 = self.path1(x)

            # pad the right and the bottom, then crop to include those pixels
            path2 = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
            path2 = path2[:, :, 1:, 1:]
            path2 = self.path2(path2)

            out = torch.cat([path1, path2], dim=1)
            out = self.bn(out)
            return out