from lib.model.conv_branch import ConvBranch
from lib.model.pool_branch import PoolBranch

from .generate_convs import ConvSettings

from math import ceil

settings = ConvSettings(0, 0, 3, 1, 1,
                        1).generate_settings_type_eq(None, (1, 16), (1, 16))
extra_pools = int(ceil(len(settings) / 4))
n_branches = 6 + len(settings) + 2 * extra_pools


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

    for (i, setting) in enumerate(settings):
        setattr(
            layer, "branch_{}".format(6 + i),
            ConvBranch(in_planes, out_planes, setting.kernel_size,
                       setting.padding, setting.dilation, setting.stride))

    for i in range(6 + len(settings), 6 + len(settings) + extra_pools):
        max_pool_i = i + extra_pools
        setattr(layer, "branch_{}".format(i),
                PoolBranch(in_planes, out_planes, 'avg'))
        setattr(layer, "branch_{}".format(max_pool_i),
                PoolBranch(in_planes, out_planes, 'max'))

    return n_branches


def pick_func(layer, layer_type, x):
    if not (0 <= layer_type < n_branches):
        exit(1)
    return getattr(layer, "branch_{}".format(layer_type.cpu().item()))(x)
