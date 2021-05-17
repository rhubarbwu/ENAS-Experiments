from .enas_layer import ENASLayer
from .spaces import spaces

from .factorized_reduction import FactorizedReduction

from torch import nn

from sys import argv

arg = argv[1]
set_func, pick_func = spaces[arg]


class SharedCNN(nn.Module):

    def __init__(self,
                 num_layers=12,
                 num_branches=6,
                 out_filters=24,
                 keep_prob=1.0,
                 fixed_arc=None):
        super(SharedCNN, self).__init__()

        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters
        self.keep_prob = keep_prob
        self.fixed_arc = fixed_arc

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, out_filters, kernel_size=3, padding=1, bias=False),
            # nn.Conv2d(1, out_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_filters, track_running_stats=False))

        self.layers = nn.ModuleList([])
        self.pooled_layers = nn.ModuleList([])

        for layer_id in range(self.num_layers):
            if self.fixed_arc is None:
                layer = ENASLayer(layer_id, self.out_filters, self.out_filters,
                                  set_func, pick_func)

            else:
                layer = FixedLayer(layer_id, self.out_filters, self.out_filters,
                                   self.fixed_arc[str(layer_id)])
            self.layers.append(layer)

            if layer_id in self.pool_layers:
                for i in range(len(self.layers)):
                    if self.fixed_arc is None:
                        self.pooled_layers.append(
                            FactorizedReduction(self.out_filters,
                                                self.out_filters))
                    else:
                        self.pooled_layers.append(
                            FactorizedReduction(self.out_filters,
                                                self.out_filters * 2))
                if self.fixed_arc is not None:
                    self.out_filters *= 2

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=1. - self.keep_prob)
        self.classify = nn.Linear(self.out_filters, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,
                                         mode='fan_in',
                                         nonlinearity='relu')

    def forward(self, x, sample_arc):

        x = self.stem_conv(x)

        prev_layers = []
        pool_count = 0
        for layer_id in range(self.num_layers):
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)
            if layer_id in self.pool_layers:
                for i, prev_layer in enumerate(prev_layers):
                    # Go through the outputs of all previous layers and downsample them
                    prev_layers[i] = self.pooled_layers[pool_count](prev_layer)
                    pool_count += 1
                x = prev_layers[-1]

        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        out = self.classify(x)

        return out