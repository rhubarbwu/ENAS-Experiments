from torch import nn


class FixedLayer(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
    '''

    def __init__(self, layer_id, in_planes, out_planes, sample_arc):
        super(FixedLayer, self).__init__()

        self.layer_id = layer_id
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.sample_arc = sample_arc

        self.layer_type = sample_arc[0]
        if self.layer_id > 0:
            self.skip_indices = sample_arc[1]
        else:
            self.skip_indices = torch.zeros(1)

        if self.layer_type == 0:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=5)
        elif self.layer_type == 1:
            self.branch = ConvBranch(in_planes,
                                     out_planes,
                                     kernel_size=5,
                                     separable=True)
        elif self.layer_type == 2:
            self.branch = PoolBranch(in_planes, out_planes, 'avg')

        else:
            raise ValueError("Unknown layer_type {}".format(self.layer_type))

        # Use concatentation instead of addition in the fixed layer for some reason
        in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
        self.dim_reduc = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(out_planes, track_running_stats=False))

    def forward(self, x, prev_layers, sample_arc):
        out = self.branch(x)

        res_layers = []
        for i, skip in enumerate(self.skip_indices):
            if skip == 1:
                res_layers.append(prev_layers[i])
        prev = res_layers + [out]
        prev = torch.cat(prev, dim=1)

        out = self.dim_reduc(prev)
        return out