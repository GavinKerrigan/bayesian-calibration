import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from torch.nn.functional import softmax

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


def classwise_ece(probs, labels, bins=15, detailed=False):
    """ Computes the classwise ECE (Kull et al, 2019)

    See: (Kull et al, 2019) "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities
                            with Dirichlet calibration"

    Args:
        probs: tensor ; shape (batch_size, num_classes)
        labels: tensor ; shape (batch_size, )
    """
    assert labels.shape[0] == probs.shape[0], 'Label/probs shape mismatch'

    batch_size, num_classes = probs.shape
    onehot_labels = torch.from_numpy(label_binarize(labels, classes=np.arange(num_classes))).float()

    ece_per_class = []
    for i in range(num_classes):
        # Probabilities / one-hot labels for the ith class
        class_probs = probs[:, i]
        class_labels = onehot_labels[:, i]

        ece_this_class = _one_class_ece(class_probs, class_labels, bins=bins)
        ece_per_class.append(ece_this_class)

    # cw-ECE is mean of all class-i ECEs
    cw_ece = np.mean(ece_per_class)

    if detailed:
        return cw_ece, ece_per_class
    else:
        return cw_ece


def _one_class_ece(probs, labels, bins=15):
    """  Helper function for classwise ECE -- computes the ECE for a single class.

    Args:
        probs: tensor ; shape (batch_size, )
        labels: tensor ; shape (batch_size )
            Assumed to be binary
    """
    batch_size = probs.shape[0]

    counts, bin_edges = np.histogram(probs, bins=bins, range=[0., 1.])
    indices = np.digitize(probs, bin_edges, right=True)

    bin_probs = np.array([torch.mean(probs[indices == j]).item() for j in range(1, bins + 1)])
    bin_proportions = np.array([torch.mean(labels[indices == i]).item()
                                for i in range(1, bins + 1)])

    this_class_ece = (1. / batch_size) * np.sum([counts[i] * np.abs(bin_probs[i] - bin_proportions[i])
                                                 for i in range(bins) if counts[i] > 0])

    return this_class_ece


def bayesian_ece_samples(logits, labels, beta_samples, bins=15):
    """ Computes a Bayesian estimate of the ECE from samples of our calibration parameters.

    Current implementation assumes just Bayesian TS
    """

    ece_samples = []
    for beta in beta_samples:
        probs = softmax(np.exp(-1. * beta) * logits, dim=1)
        ece = expected_calibration_error(probs, labels, bins=bins)
        ece_samples.append(ece)

    return ece_samples


def bayesian_ece_credible_interval(logits, labels, beta_samples, lower, upper, bins=15):
    """ Thin wrapper around bayesian_ece_samples that computes the given CIs.

    lower/upper are scalars in [0, 1] representing the lower/upper percentiles for the CI.
    """

    ece_samples = bayesian_ece_samples(logits, labels, beta_samples, bins=bins)
    lower_ece = np.quantile(ece_samples, lower)
    upper_ece = np.quantile(ece_samples, upper)

    return lower_ece, upper_ece

