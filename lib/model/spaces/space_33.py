from ..conv_branch import ConvBranch
from ..pool_branch import PoolBranch

from torch.nn import ConvTranspose2d

n_branches = 16


def set_func(layer, in_planes, out_planes):

    layer.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3)
    layer.branch_1 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                separable=True)
    layer.branch_2 = ConvBranch(in_planes, out_planes, kernel_size=5)
    layer.branch_3 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                separable=True)
    layer.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
    layer.branch_5 = PoolBranch(in_planes, out_planes, 'max')

    layer.branch_6 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                dilation=2,
                                extra_padding=1)
    layer.branch_7 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                dilation=3,
                                extra_padding=2)
    layer.branch_8 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                dilation=2,
                                extra_padding=2)
    layer.branch_9 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                dilation=3,
                                extra_padding=4)
    layer.branch_10 = ConvBranch(in_planes,
                                 out_planes,
                                 kernel_size=5,
                                 dilation=6,
                                 extra_padding=10)

    layer.branch_11 = ConvTranspose2d(in_planes,
                                      out_planes,
                                      kernel_size=3,
                                      dilation=2,
                                      padding=2)
    layer.branch_12 = ConvTranspose2d(in_planes,
                                      out_planes,
                                      kernel_size=3,
                                      dilation=3,
                                      padding=3)
    layer.branch_13 = ConvTranspose2d(in_planes,
                                      out_planes,
                                      kernel_size=5,
                                      dilation=2,
                                      padding=4)
    layer.branch_14 = ConvTranspose2d(in_planes,
                                      out_planes,
                                      kernel_size=5,
                                      dilation=3,
                                      padding=6)
    layer.branch_15 = ConvTranspose2d(in_planes,
                                      out_planes,
                                      kernel_size=5,
                                      dilation=6,
                                      padding=12)

    return n_branches


def pick_func(layer, layer_type, x):
    if layer_type == 0:
        out = layer.branch_0(x)
    elif layer_type == 1:
        out = layer.branch_1(x)
    elif layer_type == 2:
        out = layer.branch_2(x)
    elif layer_type == 3:
        out = layer.branch_3(x)
    elif layer_type == 4:
        out = layer.branch_4(x)
    elif layer_type == 5:
        out = layer.branch_5(x)
    elif layer_type == 6:
        out = layer.branch_6(x)
    elif layer_type == 7:
        out = layer.branch_7(x)
    elif layer_type == 8:
        out = layer.branch_8(x)
    elif layer_type == 9:
        out = layer.branch_9(x)
    elif layer_type == 10:
        out = layer.branch_10(x)
    elif layer_type == 11:
        out = layer.branch_11(x)
    elif layer_type == 12:
        out = layer.branch_12(x)
    elif layer_type == 13:
        out = layer.branch_13(x)
    elif layer_type == 14:
        out = layer.branch_14(x)
    elif layer_type == 15:
        out = layer.branch_15(x)

    return out


functions = (set_func, pick_func)
