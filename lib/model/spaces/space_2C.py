from ..conv_branch import ConvBranch
from ..pool_branch import PoolBranch

from torch.nn import ConvTranspose2d

n_branches = 46


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

    layer.branch_6 = ConvTranspose2d(in_planes,
                                     out_planes,
                                     kernel_size=3,
                                     padding=(3 - 1) // 2,
                                     bias=False)
    layer.branch_7 = ConvTranspose2d(in_planes,
                                     out_planes,
                                     kernel_size=5,
                                     padding=(5 - 1) // 2,
                                     bias=False)

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
    elif 6 <= layer_type < 26:
        out = layer.branch_6(x)
    elif 26 <= layer_type < 46:
        out = layer.branch_7(x)

    return out


functions = (set_func, pick_func)
