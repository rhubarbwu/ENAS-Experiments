from lib.model.conv_branch import ConvBranch
from lib.model.pool_branch import PoolBranch

from .gaussian import GaussianNoise

from torch.nn import ConvTranspose2d, Dropout, Identity

n_branches = 42


def set_func(layer, in_planes, out_planes):

    layer.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3, padding=1)
    layer.branch_1 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1,
                                separable=True)
    layer.branch_2 = ConvBranch(in_planes, out_planes, kernel_size=5, padding=2)
    layer.branch_3 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2,
                                separable=True)
    layer.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
    layer.branch_5 = PoolBranch(in_planes, out_planes, 'max')

    layer.branch_6 = Identity(None, None)
    layer.branch_7 = GaussianNoise(10.)
    layer.branch_8 = Dropout(1.)
    layer.branch_9 = ConvTranspose2d(in_planes,
                                     out_planes,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False)
    layer.branch_10 = ConvTranspose2d(in_planes,
                                      out_planes,
                                      kernel_size=5,
                                      padding=2,
                                      bias=False)
    layer.branch_11 = layer.branch_7

    return n_branches


def pick_func(layer, layer_type, x):
    if layer_type < 6:
        out = getattr(layer, "branch_{}".format(layer_type.cpu().item()))(x)
    elif 6 <= layer_type < 42:
        out = getattr(
            layer,
            "branch_{}".format((layer_type.cpu().item() - 6) // 6 + 6))(x)

    return out


functions = (set_func, pick_func)
