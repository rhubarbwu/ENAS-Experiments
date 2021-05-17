from .conv_branch import ConvBranch
from .pool_branch import PoolBranch

from torch import nn


def set_branches(layer, in_planes, out_planes):
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


def switch_branch(layer, layer_type, x):
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
    else:
        raise ValueError("Unknown layer_type {}".format(layer_type))

    return out


class ENASLayer(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
    '''

    def __init__(self, layer_id, in_planes, out_planes):
        super(ENASLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes
        # TO POISTON THE SEARCH SPACE TAKE OUT VARIOUS COMBINATIONS OF THE
        # FOLLOWING BRANCHES AND MAKE CHANGES TO THE FORWARD FUNCTION BELOW!
        # WE SUGGEST TAKING OUT 3 BY 3 CONVOLUTIOJNS AND ONE OF THE POOLING
        # BRANCHES AS AN APPROPRIATE POISONING.
        set_branches(self, in_planes, out_planes)

        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x, prev_layers, sample_arc):
        layer_type = sample_arc[0]
        if self.layer_id > 0:
            skip_indices = sample_arc[1]
        else:
            skip_indices = []

        out = switch_branch(self, layer_type, x)

        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out