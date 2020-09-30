from torch import nn, optim
import torch

""" This file contains various methods for maximum likelihood post-hoc calibration.
"""


def temperature_scaling(logits, labels, optimizer='adam'):
    """ Performs maximum likelihood temperature scaling by minimizing the NLL on the given logits/labels.

    Args:
        logits: torch tensor ; shape (batch_size , num_classes)
        labels: torch tensor ; shape (batch_size, )
    Returns:
        The scalar temperature that minimizes the NLL on the given data.
    """
    init_weight = torch.tensor(1., requires_grad=True)

    out = _scaling_nll_optimizer(logits, labels, init_weight, optimizer=optimizer)

    temperature = 1. / out.pop('weights')  # Map weights to temperatures
    out['temperature'] = temperature

    return out


def vector_scaling(logits, labels, bias=False, optimizer='adam'):
    """ Performs maximum likelihood vector scaling by minimizing the NLL on the given logits/labels.

    Args:
        logits: torch tensor ; shape (batch_size, num_classes)
        labels: torch tensor ; shape (batch_size, )
    Returns:
        A tensor of temperatures of shape (num_classes,) that minimizes the NLL on the given data.
    """
    num_classes = logits.shape[1]
    init_weights = torch.ones(num_classes, requires_grad=True)
    if bias:
        bias = torch.zeros(num_classes, requires_grad=True)
    else:
        bias = None

    out = _scaling_nll_optimizer(logits, labels, init_weights, bias=bias, optimizer=optimizer)

    temperature_vector = 1. / out.pop('weights')  # Map weights to temperatures
    out['temperature'] = temperature_vector

    return out


def _scaling_nll_optimizer(logits, labels, weights, bias=None, optimizer='adam'):
    """ This function optimizes the NLL through scaling the logits by a matrix of weights.

    The weights can be a scalar (temperature scaling), vector (vector scaling), or a full matrix.
    """
    logits = logits.detach().clone()
    labels = labels.detach().clone()

    nll = nn.CrossEntropyLoss()

    if optimizer == 'adam':
        # Use Adam with some reasonable defaults to optimize
        if bias is None:
            params = [weights]
            bias = torch.zeros(logits.shape[1])
        else:
            params = [weights, bias]
        optimizer = optim.Adam(params, lr=0.01)
        num_steps = 7500
        loss_tr = []
        loss = None
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = nll(weights * logits + bias, labels)
            loss.backward()
            optimizer.step()
            loss_tr.append(loss.item())

        out = {'weights': weights.detach(),
               'bias': bias.detach(),
               'loss': loss.item(),
               'loss_trace': loss_tr}

    elif optimizer == 'LBFGS':
        if bias is None:
            params = [weights]
            bias = torch.zeros(logits.shape[1])
        else:
            params = [weights, bias]

        optimizer = optim.LBFGS(params, lr=0.01)
        loss = None

        def closure():
            optimizer.zero_grad()
            loss = nll(weights * logits + bias, labels)
            loss.backward()
            # loss_tr.append(loss.item())
            return loss

        for _ in range(1000):
            optimizer.step(closure)

        out = {'weights': weights.detach(),
               'bias': bias.detach(),
               'loss': closure().item(),
               'optimizer': optimizer}

    return out
