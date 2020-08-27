from torch import nn, optim
import torch

""" This file contains various methods for maximum likelihood post-hoc calibration.
"""


def temperature_scaling(logits, labels):
    """ Performs maximum likelihood temperature scaling by minimizing the NLL on the given logits/labels.

    Args:
        logits: torch tensor ; shape (batch_size , num_classes)
        labels: torch tensor ; shape (batch_size, )
    Returns:
        The scalar temperature that minimizes the NLL on the given data.
    """
    init_weight = torch.tensor(1., requires_grad=True)

    out = _scaling_nll_optimizer(logits, labels, init_weight)

    temperature = 1. / out.pop('weights')  # Map weights to temperatures
    out['temperature'] = temperature

    return out


def vector_scaling(logits, labels):
    """ Performs maximum likelihood vector scaling by minimizing the NLL on the given logits/labels.

    Args:
        logits: torch tensor ; shape (batch_size, num_classes)
        labels: torch tensor ; shape (batch_size, )
    Returns:
        A tensor of temperatures of shape (num_classes,) that minimizes the NLL on the given data.
    """
    num_classes = logits.shape[1]
    init_weights = torch.ones(num_classes, requires_grad=True)

    out = _scaling_nll_optimizer(logits, labels, init_weights)

    temperature_vector = 1. / out.pop('weights')  # Map weights to temperatures
    out['temperature'] = temperature_vector

    return out


def _scaling_nll_optimizer(logits, labels, weights):
    """ This function optimizes the NLL through scaling the logits by a matrix of weights.

    The weights can be a scalar (temperature scaling), vector (vector scaling), or a full matrix.
    """
    logits = logits.detach().clone()
    labels = labels.detach().clone()

    nll = nn.CrossEntropyLoss()

    # Use Adam with some reasonable defaults to optimize
    optimizer = optim.Adam([weights], lr=0.01)
    num_steps = 500
    loss_tr = []
    loss = None
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = nll(weights * logits, labels)
        loss.backward()
        optimizer.step()
        loss_tr.append(loss.item())

    out = {'weights': weights.detach(),
           'loss': loss.item(),
           'loss_trace': loss_tr}

    return out
