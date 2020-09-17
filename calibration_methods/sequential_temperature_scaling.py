from .maximum_likelihood import temperature_scaling

import torch
from torch.nn.functional import softmax
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


class SequentialBatchTS:
    """ Performs sequential calibration by optimizing the temperature only on the given data.
    """

    def __init__(self, temperature=1.):
        self.temperature = temperature

    def update(self, logits, labels):
        self.temperature = temperature_scaling(logits, labels)['temperature']

    def calibrate(self, logits):
        tempered_logits = 1. / self.temperature * logits
        return softmax(tempered_logits, dim=1)


class SequentialSGDTS:
    """ Performs sequential calibration by taking gradient steps on the given data.
    """

    def __init__(self, temperature=1., **kwargs):
        self.temperature = torch.tensor(temperature, requires_grad=True)

        self.optimizer = SGD([self.temperature], lr=0.1)
        self.num_grad_steps = kwargs.pop('num_grad_steps', 1)
        self.loss = CrossEntropyLoss()

    def update(self, logits, labels):
        """ Updates the temperature using SGD.
        """
        # TODO: How many steps? Should it just be one per batch? Or should each batch go to convergence?
        for _ in range(self.num_grad_steps):
            self.optimizer.zero_grad()
            loss = self.loss(1./self.temperature * logits, labels)
            loss.backward()
            self.optimizer.step()

    def calibrate(self, logits):
        tempered_logits = 1. / self.temperature * logits
        return softmax(tempered_logits, dim=1).detach()


class MovingWindowTS:

    def __init__(self, window_size, temperature=1.):
        self.temperature = torch.tensor(temperature, requires_grad=True)
        self.window_size = window_size
        self.stored_logits = None
        self.stored_labels = None

    def update(self, logits, labels):
        # Store the last window_size datapoints
        if self.stored_logits is None:
            self.stored_logits = logits[:self.window_size, :]
            self.stored_labels = labels[:self.window_size]
        else:
            self.stored_logits = torch.cat([self.stored_logits, logits], dim=0)[-self.window_size:, :]
            self.stored_labels = torch.cat([self.stored_labels, labels], dim=0)[-self.window_size:]

        # Update the temperature by fitting TS (to convergence) on the current window.
        self.temperature = temperature_scaling(self.stored_logits, self.stored_labels)['temperature']

    def calibrate(self, logits):
        tempered_logits = 1. / self.temperature * logits
        return softmax(tempered_logits, dim=1)


