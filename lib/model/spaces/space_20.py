from ..conv_branch import ConvBranch
from ..pool_branch import PoolBranch

from torch.nn import Identity

n_branches = 12


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

    layer.branch_6 = Identity(None, None)
    layer.branch_7 = Identity(None, None)
    layer.branch_8 = Identity(None, None)
    layer.branch_9 = Identity(None, None)
    layer.branch_10 = Identity(None, None)
    layer.branch_11 = Identity(None, None)

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

    return out


functions = (set_func, pick_func)
