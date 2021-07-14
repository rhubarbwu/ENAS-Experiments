from .hparams import args
from .model.spaces import bad_op_split

import numpy as np
import torch


def evaluate_model(epoch, controller, shared_cnn, data_loaders, n_samples=10):
    """Print the validation and test accuracy for a controller and shared_cnn.
    Args:
        epoch: Current epoch.
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        n_samples: Number of architectures to test when looking for the best one.

    Returns: Nothing.
    """

    controller.eval()
    shared_cnn.eval()

    print('Here are ' + str(n_samples) + ' architectures:')
    best_arc, _, avg_good, avg_bad, best_good, best_bad = get_best_arc(
        controller, shared_cnn, data_loaders, n_samples, verbose=True)

    valid_loader = data_loaders['valid_subset']
    test_loader = data_loaders['test_dataset']

    valid_acc = get_eval_accuracy(valid_loader, shared_cnn, best_arc)
    test_acc = get_eval_accuracy(test_loader, shared_cnn, best_arc)

    print('Epoch ' + str(epoch) + ': Eval')
    print('valid_accuracy: %.4f' % (valid_acc))
    print('test_accuracy: %.4f' % (test_acc))

    controller.train()
    shared_cnn.train()
    return valid_acc, test_acc, avg_good, avg_bad, best_good, best_bad


def get_best_arc(controller,
                 shared_cnn,
                 data_loaders,
                 n_samples=10,
                 verbose=False):
    """Evaluate several architectures and return the best performing one.
    Args:
        controller: Controller module that generates architectures to be trained.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        data_loaders: Dict containing data loaders.
        n_samples: Number of architectures to test when looking for the best one.
        verbose: If True, display the architecture and resulting validation accuracy.

    Returns:
        best_arc: The best performing architecture.
        best_vall_acc: Accuracy achieved on the best performing architecture.
    All architectures are evaluated on the same minibatch from the validation set.
    """

    controller.eval()
    shared_cnn.eval()

    valid_loader = data_loaders['valid_subset']

    images, labels = next(iter(valid_loader))
    images = images.cuda()
    labels = labels.cuda()

    arcs = []
    val_accs = []
    for i in range(n_samples):
        with torch.no_grad():
            controller()  # perform forward pass to generate a new architecture
        sample_arc = controller.sample_arc
        arcs.append(sample_arc)

        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        val_acc = torch.mean(
            (torch.max(pred, 1)[1] == labels).type(torch.float)).cpu().numpy()
        val_accs.append(val_acc)

        if verbose:
            print_arc(sample_arc)
            print('val_acc=' + str(val_acc))
            print('-' * 80)

    # tracking number of good / bad layers in each arc generated
    arc_goods = [0 for i in range(n_samples)]
    arc_bads = [0 for i in range(n_samples)]

    for i in range(n_samples):
        for _, value in arcs[i].items():
            branch_type = value[0].cpu().numpy().tolist()[0]

            if branch_type >= bad_op_split:
                arc_bads[i] += 1
            else:
                arc_goods[i] += 1

    avg_good = round(sum(arc_goods) / n_samples, 2)
    avg_bad = round(sum(arc_bads) / n_samples, 2)

    best_iter = np.argmax(val_accs)
    best_arc = arcs[best_iter]
    best_val_acc = val_accs[best_iter]

    controller.train()
    shared_cnn.train()
    return best_arc, best_val_acc, avg_good, avg_bad, arc_goods[
        best_iter], arc_bads[best_iter]


def get_eval_accuracy(loader, shared_cnn, sample_arc):
    """Evaluate a given architecture.
    Args:
        loader: A single data loader.
        shared_cnn: CNN that contains all possible architectures, with shared weights.
        sample_arc: The architecture to use for the evaluation.

    Returns:
        acc: Average accuracy.
    """
    total = 0.
    acc_sum = 0.
    for (images, labels) in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        acc_sum += torch.sum((torch.max(pred,
                                        1)[1] == labels).type(torch.float))
        total += pred.shape[0]

    acc = acc_sum / total
    return acc


def print_arc(sample_arc):
    """Display a sample architecture in a readable format.

    Args: 
        sample_arc: The architecture to display.
    Returns: Nothing.
    """
    for key, value in sample_arc.items():
        if len(value) == 1:
            branch_type = value[0].cpu().numpy().tolist()
            print('[' + ' '.join(str(n) for n in branch_type) + ']')
        else:
            branch_type = value[0].cpu().numpy().tolist()
            skips = value[1].cpu().numpy().tolist()
            print('[' + ' '.join(str(n) for n in (branch_type + skips)) + ']')
