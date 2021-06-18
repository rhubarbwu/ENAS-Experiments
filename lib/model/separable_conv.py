from torch import nn


class SeparableConv(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 bias,
                 padding=0,
                 dilation=1,
                 stride=1):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes,
                                   in_planes,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_planes,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_planes,
                                   out_planes,
                                   kernel_size=1,
                                   padding=0,
                                   groups=in_planes,
                                   bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out