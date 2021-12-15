from numpy import arcsin
from ..hparams import args
from .qenas import init_Q, sample

import torch
from torch import nn

from torch.distributions.categorical import Categorical


class Controller(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
    '''

    def __init__(self,
                 search_for="macro",
                 search_whole_channels=True,
                 num_layers=12,
                 num_branches=6,
                 out_filters=36,
                 lstm_size=32,
                 lstm_num_layers=2,
                 tanh_constant=1.5,
                 temperature=None,
                 skip_target=0.4,
                 skip_weight=0.8):
        super(Controller, self).__init__()

        self.Q = init_Q(num_branches)
        self.sample_entropy = torch.Tensor(0)

        self.search_for = search_for
        self.search_whole_channels = search_whole_channels
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self._create_params()

    def _create_params(self):
        '''
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
        '''
        self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                              hidden_size=self.lstm_size,
                              num_layers=self.lstm_num_layers)

        self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the starting input

        if self.search_whole_channels:
            self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
            self.w_soft = nn.Linear(self.lstm_size,
                                    self.num_branches,
                                    bias=False)
        else:
            assert False, "Not implemented error: search_whole_channels = False"

        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self):
        '''
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
        '''

        arc_seq = []

        for layer_id in range(self.num_layers):
            arc_seq = sample(self.Q, arc_seq)

        self.sample_arc = {str(i): layer for i, layer in enumerate(arc_seq)}
