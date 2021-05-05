from .conv_branch import ConvBranch
from .pool_branch import PoolBranch

from torch import nn


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
        self.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3)
        self.branch_1 = ConvBranch(in_planes,
                                   out_planes,
                                   kernel_size=3,
                                   separable=True)
        self.branch_2 = ConvBranch(in_planes, out_planes, kernel_size=5)
        self.branch_3 = ConvBranch(in_planes,
                                   out_planes,
                                   kernel_size=5,
                                   separable=True)
        self.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
        self.branch_5 = PoolBranch(in_planes, out_planes, 'max')

        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x, prev_layers, sample_arc):
        layer_type = sample_arc[0]
        if self.layer_id > 0:
            skip_indices = sample_arc[1]
        else:
            skip_indices = []

        # CHANGE THIS BASED ON THE POISONING MENTIONED ABOVE
        if layer_type == 0:
            out = self.branch_0(x)
        elif layer_type == 1:
            out = self.branch_1(x)
        elif layer_type == 2:
            out = self.branch_2(x)
        elif layer_type == 3:
            out = self.branch_3(x)
        elif layer_type == 4:
            out = self.branch_4(x)
        elif layer_type == 5:
            out = self.branch_5(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out