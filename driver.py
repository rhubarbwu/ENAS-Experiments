from lib.dataset import load_datasets
from lib.hparams import args
from lib.model.controller import Controller
from lib.model.shared_cnn import SharedCNN
from lib.train import train_enas, train_fixed
from lib.model.spaces import ns_branches

import numpy as np
from os.path import isfile
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

np.random.seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

data_loaders = load_datasets(args)

from sys import argv

arg = argv[1]
n_branches = ns_branches[arg]

controller = Controller(search_for=args['search_for'],
                        search_whole_channels=True,
                        num_layers=args['child_num_layers'],
                        num_branches=n_branches,
                        out_filters=args['child_out_filters'],
                        lstm_size=args['controller_lstm_size'],
                        lstm_num_layers=args['controller_lstm_num_layers'],
                        tanh_constant=args['controller_tanh_constant'],
                        temperature=None,
                        skip_target=args['controller_skip_target'],
                        skip_weight=args['controller_skip_weight'])
controller = controller.cuda()

shared_cnn = SharedCNN(num_layers=args['child_num_layers'],
                       num_branches=n_branches,
                       out_filters=args['child_out_filters'],
                       keep_prob=args['child_keep_prob'])

shared_cnn = shared_cnn.cuda()

# https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                        lr=args['controller_lr'],
                                        betas=(0.0, 0.999),
                                        eps=1e-3)

# https://github.com/melodyguan/enas/blob/master/src/utils.py#L213
shared_cnn_optimizer = torch.optim.SGD(params=shared_cnn.parameters(),
                                       lr=args['child_lr_max'],
                                       momentum=0.9,
                                       nesterov=True,
                                       weight_decay=args['child_l2_reg'])

# https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
shared_cnn_scheduler = CosineAnnealingLR(optimizer=shared_cnn_optimizer,
                                         T_max=args['child_lr_T'],
                                         eta_min=args['child_lr_min'])

if args['resume']:
    if isfile(args['resume']):
        print("Loading checkpoint '{}'".format(args['resume']))
        checkpoint = torch.load(args['resume'])
        start_epoch = checkpoint['epoch']
        # args = checkpoint['args']
        print(checkpoint.keys())
        shared_cnn.load_state_dict(checkpoint['shared_cnn_state_dict'])
        controller.load_state_dict(checkpoint['controller_state_dict'])
        shared_cnn_optimizer.load_state_dict(checkpoint['shared_cnn_optimizer'])
        controller_optimizer.load_state_dict(checkpoint['controller_optimizer'])
        shared_cnn_scheduler.optimizer = shared_cnn_optimizer  # Not sure if this actually works
        print("Loaded checkpoint '{}' (epoch {})".format(
            args['resume'], checkpoint['epoch']))
    else:
        raise ValueError("No checkpoint found at '{}'".format(args['resume']))
else:
    start_epoch = 0

if not args['fixed_arc']:
    train_enas(start_epoch, controller, shared_cnn, data_loaders,
               shared_cnn_optimizer, controller_optimizer, shared_cnn_scheduler,
               args)
else:
    assert args[
        'resume'] != '', 'A pretrained model should be used when training a fixed architecture.'
    train_fixed(start_epoch, controller, shared_cnn, data_loaders)
