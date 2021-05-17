from torch import nn


class ENASLayer(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
    '''

    def __init__(self, layer_id, in_planes, out_planes, set_func, pick_func):
        super(ENASLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes
        # TO POISTON THE SEARCH SPACE TAKE OUT VARIOUS COMBINATIONS OF THE
        # FOLLOWING BRANCHES AND MAKE CHANGES TO THE FORWARD FUNCTION BELOW!
        # WE SUGGEST TAKING OUT 3 BY 3 CONVOLUTIOJNS AND ONE OF THE POOLING
        # BRANCHES AS AN APPROPRIATE POISONING.
        self.n_branches = set_func(self, in_planes, out_planes)
        self.pick_func = pick_func

        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x, prev_layers, sample_arc):
        layer_type = sample_arc[0]
        if self.layer_id > 0:
            skip_indices = sample_arc[1]
        else:
            skip_indices = []

        if layer_type >= self.n_branches:
            raise ValueError("Unknown layer_type {}".format(layer_type))
        out = self.pick_func(self, layer_type, x)

        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out