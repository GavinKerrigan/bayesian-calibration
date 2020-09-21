import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import label_binarize

from utils.metrics import expected_calibration_error as ECE
from utils.metrics import classwise_ece as CWECE


def confidence_reliability_diagram(probs, labels, bins=15):
    assert labels.shape[0] == probs.shape[0], 'Label/probs shape mismatch'
    ece = ECE(probs, labels, bins)  # Re-doing some computation here but whatever

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

    # ---- Setting up figure
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # ----- Plotting data
    for i in range(bins):
        x = [bin_edges[i], bin_edges[i], bin_edges[i + 1], bin_edges[i + 1]]
        y = [0, accuracies[i], accuracies[i], 0]
        ax.fill(x, y, 'b', alpha=0.6, edgecolor='black')

    ax.text(0.01, .875, 'ECE: {:.2f}'.format(100. * ece), size=15)

    score_hist_ax = ax.twinx()
    score_hist_ax.hist(scores, density=True, label='Score distr.', color='orange', alpha=0.7, bins=20)
    score_hist_ax.set_ylim(0, 35)
    score_hist_ax.legend(loc='upper left')
    score_hist_ax.set_yticks([])

    return fig


def classwise_reliability_diagram(probs, labels, class_idx, bins=15):
    assert labels.shape[0] == probs.shape[0], 'Label/probs shape mismatch'

    batch_size, num_classes = probs.shape
    onehot_labels = torch.from_numpy(label_binarize(labels, classes=np.arange(num_classes))).float()

    # Predicted probabilities / one-hot labels for the given class
    class_probs = probs[:, class_idx]
    class_labels = onehot_labels[:, class_idx]

    counts, bin_edges = np.histogram(class_probs, bins=bins, range=[0., 1.])
    indices = np.digitize(class_probs, bin_edges, right=True)
    bin_probs = np.array([torch.mean(class_probs[indices == j]).item() for j in range(1, bins + 1)])
    bin_proportions = np.array([torch.mean(class_labels[indices == i]).item()
                                for i in range(1, bins + 1)])

    this_class_ece = (1. / batch_size) * np.sum([counts[i] * np.abs(bin_probs[i] - bin_proportions[i])
                                                 for i in range(bins) if counts[i] > 0])

    # ---- Setting up figure
    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('Class Score')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # ----- Plotting data
    for i in range(bins):
        x = [bin_edges[i], bin_edges[i], bin_edges[i + 1], bin_edges[i + 1]]
        y = [0, bin_proportions[i], bin_proportions[i], 0]
        ax.fill(x, y, 'b', alpha=0.6, edgecolor='black')

    ax.text(0.01, .875, 'Class {} CE: {:.3f}'.format(class_idx, this_class_ece), size=15)

    score_hist_ax = ax.twinx()
    score_hist_ax.hist(class_probs, density=True, label='Score distr.', color='orange', alpha=0.7, bins=20)
    score_hist_ax.set_ylim(0, 35)
    score_hist_ax.legend(loc='upper left')
    score_hist_ax.set_yticks([])

    return fig