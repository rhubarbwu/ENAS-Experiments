### Args and Modules
from lib.hparams import args
from sys import argv
if len(argv) == 4:
    args["set"], args["experiment"], args["num_epochs"] = argv[1], argv[2], int(
        argv[3])
args["output_filename"] = args["resume"] = "checkpoints/{}.{}.pth.tar".format(
    args["set"], args["experiment"])

### Modules
from importlib import import_module

experiment = import_module("experiments.{}.{}".format(args["set"],
                                                      args["experiment"]))

### NumPy Seeds
import numpy as np
import torch

np.random.seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

### DataLoader
from lib.dataset import load_datasets

data_loaders = load_datasets(args)

### Controller
from lib.model.controller import Controller

controller = Controller(search_for=args['search_for'],
                        search_whole_channels=True,
                        num_layers=args['child_num_layers'],
                        num_branches=experiment.n_branches,
                        out_filters=args['child_out_filters'],
                        lstm_size=args['controller_lstm_size'],
                        lstm_num_layers=args['controller_lstm_num_layers'],
                        tanh_constant=args['controller_tanh_constant'],
                        temperature=None,
                        skip_target=args['controller_skip_target'],
                        skip_weight=args['controller_skip_weight'])
controller = controller.cuda()

### SharedCNN
from lib.model.shared_cnn import SharedCNN

shared_cnn = SharedCNN(experiment,
                       num_layers=args['child_num_layers'],
                       num_branches=experiment.n_branches,
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

### CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR

# https://github.com/melodyguan/enas/blob/master/src/utils.py#L154
shared_cnn_scheduler = CosineAnnealingLR(optimizer=shared_cnn_optimizer,
                                         T_max=args['child_lr_T'],
                                         eta_min=args['child_lr_min'])

start_epoch = 0
from os.path import isfile
if args['resume'] and isfile(args['resume']):
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
    print("Loaded checkpoint '{}' (epoch {})".format(args['resume'],
                                                     checkpoint['epoch']))

### TRAINING
from lib.train import train_enas, train_fixed

if not args['fixed_arc']:
    train_enas(start_epoch, controller, shared_cnn, data_loaders,
               shared_cnn_optimizer, controller_optimizer, shared_cnn_scheduler,
               args)
else:
    assert args[
        'resume'] != '', 'A pretrained model should be used when training a fixed architecture.'
    train_fixed(start_epoch, controller, shared_cnn, data_loaders)
