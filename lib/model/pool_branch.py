import torch
from torch import nn


class PoolBranch(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L546
    '''

    def __init__(self, in_planes, out_planes, avg_or_max, return_indices=False):
        super(PoolBranch, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.avg_or_max = avg_or_max

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes, track_running_stats=False), nn.ReLU())

        if avg_or_max == 'avg':
            self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        elif avg_or_max == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           return_indices=return_indices)
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out