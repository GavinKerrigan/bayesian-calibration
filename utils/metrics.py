import numpy as np
import torch
from sklearn.preprocessing import label_binarize

"""
This file contains various metrics for measuring calibration.
"""


def brier_score(probs, labels):
    """
    Computes the average multi-class Brier score of the given probabilities and labels.

    Args:
        probs: torch tensor ; shape (batch_size, num_classes)
        labels: torch tensor ; shape (batch_size, )
    Returns:
        The Brier score, averaged over all instances in the batch.
    """
    assert probs.shape[0] == labels.shape[0], 'Shape mismatch'

    num_classes = probs.shape[1]
    onehot_labels = label_binarize(labels, classes=np.arange(num_classes))  # Convert labels to one-hot encoding

    instance_wise_bs = torch.sum((probs - onehot_labels) ** 2, axis=1)
    mean_bs = torch.mean(instance_wise_bs).item()

    return mean_bs


# Generously borrowed from:
# https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/metrics_lib.py
def expected_calibration_error(probs, labels, bins=15):
    """ Estimates the ECE using the given predicted probabilities and ground-truth labels.

    Args:
        probs: torch tensor ; (batch_size, num_classes)
        labels: torch tensor ; (batch_size, )
        bins: int ; The number of equal-width bins to use for estimating the ECE.
    Returns:
        The estimated ECE.
    """

    assert labels.shape[0] == probs.shape[0], 'Label/probs shape mismatch'

    batch_size = probs.shape[0]

    # Get model scores, predicted labels, and outcomes
    scores, pred_labels = probs.max(dim=1)
    outcomes = (pred_labels == labels).float()

    # Bin scores
    counts, bin_edges = np.histogram(scores, bins=bins, range=[0., 1.])
    indices = np.digitize(scores, bin_edges, right=True)

    # Get confidence and accuracy in each bin
    confidences = np.array([torch.mean(scores[indices == i]).item()
                            for i in range(1, bins + 1)])
    accuracies = np.array([torch.mean(outcomes[indices == i]).item()
                           for i in range(1, bins + 1)])

    # Compute ECE
    ece = (1. / batch_size) * np.sum([counts[i] * np.abs(confidences[i] - accuracies[i])
                                      for i in range(bins) if counts[i] > 0])
    return ece
