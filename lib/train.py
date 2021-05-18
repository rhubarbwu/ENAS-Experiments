from .model.average_meter import AverageMeter
from .eval import evaluate_model
from .hparams import args

from datetime import datetime
import pandas as pd
import numpy as np
from time import time
import torch
from torch import nn


def train_shared_cnn(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     shared_cnn_optimizer,
                     args,
                     fixed_arc=None):
    """Train shared_cnn by sampling architectures from the controller.
    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        shared_cnn_optimizer: Optimizer for the shared_cnn.
        fixed_arc: Architecture to train, overrides the controller sample
        ...
    
    Returns: Nothing.
    """

    controller.eval()

    if fixed_arc is None:
        # Use a subset of the training set when searching for an arhcitecture
        train_loader = data_loaders['train_subset']
    else:
        # Use the full training set when training a fixed architecture
        train_loader = data_loaders['train_dataset']

    train_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    log_every = args["log_every"]

    for i, (images, labels) in enumerate(train_loader):
        start = time()
        images = images.cuda()
        labels = labels.cuda()

        if fixed_arc is None:
            with torch.no_grad():
                controller(
                )  # perform forward pass to generate a new architecture
            sample_arc = controller.sample_arc
        else:
            sample_arc = fixed_arc

        shared_cnn.zero_grad()
        pred = shared_cnn(images, sample_arc)
        loss = nn.CrossEntropyLoss()(pred, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(),
                                                   args['child_grad_bound'])
        shared_cnn_optimizer.step()

        train_acc = torch.mean((torch.max(pred,
                                          1)[1] == labels).type(torch.float))

        train_acc_meter.update(train_acc.item())
        loss_meter.update(loss.item())

        end = time()

        if log_every and (i % args['log_every'] == 0):
            learning_rate = shared_cnn_optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tch_step=' + str(i) + \
                      '\tloss=%.6f' % (loss_meter.val) + \
                      '\tlr=%.4f' % (learning_rate) + \
                      '\t|g|=%.4f' % (grad_norm.item()) + \
                      '\tacc=%.4f' % (train_acc_meter.val) + \
                      '\ttime=%.2fit/s' % (1. / (end - start))
            print(display)

    controller.train()


def train_controller(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     controller_optimizer,
                     args,
                     baseline=None):
    """Train controller to optimizer validation accuracy using REINFORCE.
    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        controller_optimizer: Optimizer for the controller.
        baseline: The baseline score (i.e. average val_acc) from the previous epoch
    
    Returns: 
        baseline: The baseline score (i.e. average val_acc) for the current epoch
    For more stable training we perform weight updates using the average of
    many gradient estimates. controller_num_aggregate indicates how many samples
    we want to average over (default = 20). By default PyTorch will sum gradients
    each time .backward() is called (as long as an optimizer step is not taken),
    so each iteration we divide the loss by controller_num_aggregate to get the 
    average.
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L270
    """
    print('Epoch ' + str(epoch) + ': Training controller')

    shared_cnn.eval()
    valid_loader = data_loaders['valid_subset']

    reward_meter = AverageMeter()
    baseline_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    controller.zero_grad()
    for i in range(args['controller_train_steps'] *
                   args['controller_num_aggregate']):
        start = time()
        images, labels = next(iter(valid_loader))
        images = images.cuda()
        labels = labels.cuda()

        controller()  # perform forward pass to generate a new architecture
        sample_arc = controller.sample_arc

        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        val_acc = torch.mean((torch.max(pred,
                                        1)[1] == labels).type(torch.float))

        # detach to make sure that gradients aren't backpropped through the reward
        reward = val_acc.clone().detach()
        reward += args['controller_entropy_weight'] * controller.sample_entropy

        if baseline is None:
            baseline = val_acc
        else:
            baseline -= (1 - args['controller_bl_dec']) * (baseline - reward)
            # detach to make sure that gradients are not backpropped through the baseline
            baseline = baseline.detach()

        loss = -1 * controller.sample_log_prob * (reward - baseline)

        if args['controller_skip_weight'] is not None:
            loss += args['controller_skip_weight'] * controller.skip_penaltys

        reward_meter.update(reward.item())
        baseline_meter.update(baseline.item())
        val_acc_meter.update(val_acc.item())
        loss_meter.update(loss.item())

        # Average gradient over controller_num_aggregate samples
        loss = loss / args['controller_num_aggregate']

        loss.backward(retain_graph=True)

        end = time()

        # Aggregate gradients for controller_num_aggregate iterationa, then update weights
        if (i + 1) % args['controller_num_aggregate'] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(),
                                                       args['child_grad_bound'])
            controller_optimizer.step()
            controller.zero_grad()

            if (i + 1) % (2 * args['controller_num_aggregate']) == 0:
                learning_rate = controller_optimizer.param_groups[0]['lr']
                display = 'ctrl_step=' + str(i) + \
                          '\tloss=%.3f' % (loss_meter.val) + \
                          '\tent=%.2f' % (controller.sample_entropy.item()) + \
                          '\tlr=%.4f' % (learning_rate) + \
                          '\t|g|=%.4f' % (grad_norm.item()) + \
                          '\tacc=%.4f' % (val_acc_meter.val) + \
                          '\tbl=%.2f' % (baseline_meter.val) + \
                          '\ttime=%.2fit/s' % (1. / (end - start))
                print(display)

    shared_cnn.train()
    return baseline, (reward_meter.avg, reward)


def train_enas(start_epoch, controller, shared_cnn, data_loaders,
               shared_cnn_optimizer, controller_optimizer, shared_cnn_scheduler,
               args):
    """Perform architecture search by training a controller and shared_cnn.
    Args:
        start_epoch: Epoch to begin on.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        shared_cnn_optimizer: Optimizer for the shared_cnn.
        controller_optimizer: Optimizer for the controller.
        shared_cnn_scheduler: Learning rate schedular for shared_cnn_optimizer
    
    Returns: Nothing.
    """

    baseline = None

    val_accs, test_accs, reward_avgs, reward_finals = [], [], [], []

    for epoch in range(start_epoch, args['num_epochs']):

        train_shared_cnn(epoch, controller, shared_cnn, data_loaders,
                         shared_cnn_optimizer, args)

        baseline, reward_vals = train_controller(epoch, controller, shared_cnn,
                                                 data_loaders,
                                                 controller_optimizer, args,
                                                 baseline)

        if epoch % args['eval_every_epochs'] == 0:
            val_acc, test_acc = evaluate_model(epoch, controller, shared_cnn,
                                               data_loaders)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            reward_avgs.append(reward_vals[0])
            reward_finals.append(reward_vals[1].detach().cpu())

        # shared_cnn_scheduler.step(epoch)
        shared_cnn_scheduler.step()

        state = {
            'epoch': epoch + 1,
            'args': args,
            'shared_cnn_state_dict': shared_cnn.state_dict(),
            'controller_state_dict': controller.state_dict(),
            'shared_cnn_optimizer': shared_cnn_optimizer.state_dict(),
            'controller_optimizer': controller_optimizer.state_dict()
        }
        filename = args['output_filename'] + '.pth.tar'
        torch.save(state, filename)

    metrics = pd.DataFrame({
        "val": np.array(val_acc),
        "test": np.array(test_acc),
        "reward_avg": np.array(reward_avgs),
        "reward_finals": np.array(reward_finals),
    })
    metrics.to_csv("experiments/{}_{}".format(args["space"], datetime.now()),
                   index=False)


def train_fixed(start_epoch, controller, shared_cnn, data_loaders):
    """Train a fixed cnn architecture.
    Args:
        start_epoch: Epoch to begin on.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
    
    Returns: Nothing.
    Given a fully trained controller and shared_cnn, we sample many architectures,
    and then train a new cnn from scratch using the best architecture we found. 
    We change the number of filters in the new cnn such that the final layer 
    has 512 channels.
    """

    best_arc, best_val_acc = get_best_arc(controller,
                                          shared_cnn,
                                          data_loaders,
                                          n_samples=100,
                                          verbose=True)
    print('Best architecture:')
    print_arc(best_arc)
    print('Validation accuracy: ' + str(best_val_acc))

    fixed_cnn = SharedCNN(
        num_layers=args['child_num_layers'],
        num_branches=args['child_num_branches'],
        out_filters=512 // 4,  # args.child_out_filters
        keep_prob=args['child_keep_prob'],
        fixed_arc=best_arc)
    fixed_cnn = fixed_cnn.cuda()

    fixed_cnn_optimizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                          lr=args['child_lr_max'],
                                          momentum=0.9,
                                          nesterov=True,
                                          weight_decay=args['child_l2_reg'])

    fixed_cnn_scheduler = CosineAnnealingLR(optimizer=fixed_cnn_optimizer,
                                            T_max=args['child_lr_T'],
                                            eta_min=args['child_lr_min'])

    test_loader = data_loaders['test_dataset']

    for epoch in range(args['num_epochs']):

        train_shared_cnn(
            epoch,
            controller,  # not actually used in training the fixed_cnn
            fixed_cnn,
            data_loaders,
            fixed_cnn_optimizer,
            args,
            best_arc)

        if epoch % args['eval_every_epochs'] == 0:
            test_acc = get_eval_accuracy(test_loader, fixed_cnn, best_arc)
            print('Epoch ' + str(epoch) + ': Eval')
            print('test_accuracy: %.4f' % (test_acc))

        fixed_cnn_scheduler.step()

        state = {
            'epoch': epoch + 1,
            'args': args,
            'best_arc': best_arc,
            'fixed_cnn_state_dict': shared_cnn.state_dict(),
            'fixed_cnn_optimizer': fixed_cnn_optimizer.state_dict()
        }
        filename = args['output_filename'] + '_fixed.pth.tar'
        torch.save(state, filename)

    return {}
