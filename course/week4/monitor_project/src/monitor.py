import torch
import numpy as np
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression


def get_ks_score(tr_probs, te_probs):
  score = None
  tr_probs = tr_probs.numpy()
  te_probs = te_probs.numpy()
  statistic, score = ks_2samp(te_probs, tr_probs)
  return score


def get_hist_score(tr_probs, te_probs, bins=10):
  score = None
  # Convert tensors to NumPy arrays
  tr_probs_np = tr_probs.numpy()
  te_probs_np = te_probs.numpy()
  # Compute the histogram counts for the training set probabilities
  tr_counts, tr_edges = np.histogram(tr_probs_np, bins=bins)
  # Compute the histogram counts for the test set probabilities
  te_counts, te_edges = np.histogram(te_probs_np, bins=tr_edges)
  # Normalize the counts to obtain frequencies
  tr_freqs = tr_counts / np.sum(tr_counts)
  te_freqs = te_counts / np.sum(te_counts)
  # Compute the histogram score as the Euclidean distance between the normalized frequencies
  score = np.linalg.norm(tr_freqs - te_freqs)
  return score


def get_vocab_outlier(tr_vocab, te_vocab):
  score = None
  num_seen=0
  num_total=len(te_vocab)
  for key, value in dict1.items():
    if key in tr_vocab and te_vocab[key] == value:
        num_seen += 1
  score = 1 - (num_seen/num_total)
  return score


class MonitoringSystem:

  def __init__(self, tr_vocab, tr_probs, tr_labels):
    self.tr_vocab = tr_vocab
    self.tr_probs = tr_probs
    self.tr_labels = tr_labels

  def calibrate(self, tr_probs, tr_labels, te_probs):
    tr_probs_cal = None
    te_probs_cal = None
    # Collect the training probabilities and their corresponding labels.
    #Sort the probabilities in ascending order.
    # Apply isotonic regression on the sorted probabilities.
    # Use the resulting isotonic transformation function to transform the probabilities of new data points.

    sorted_probabilities, sorted_labels = zip(*sorted(zip(tr_probs, tr_labels)))
    ir = IsotonicRegression()
    tr_probs_cal = ir.fit_transform(sorted_probabilities, sorted_labels)
    te_probs_cal = ir.transform(te_probs)
    # ============================
    # FILL ME OUT
    # 
    # Calibrate probabilities with isotonic regression using 
    # the training probabilities and labels. 
    # 
    # Pseudocode:
    # --
    # use IsotonicRegression(out_of_bounds='clip')
    #   See documentation for `out_of_bounds` description.
    # tr_probs_cal = fit calibration model
    # te_probs_cal = evaluate using fitted model
    # 
    # Type:
    # --
    # `tr_probs_cal`: torch.Tensor. Note that sklearn
    # returns a NumPy array. You will need to cast 
    # it to a torch.Tensor.
    # 
    # `te_probs_cal`: torch.Tensor
    # ============================
    return tr_probs_cal, te_probs_cal

  def monitor(self, te_vocab, te_probs):
    tr_probs, te_probs = self.calibrate(self.tr_probs, self.tr_labels, te_probs)

    # compute metrics. 
    ks_score = get_ks_score(tr_probs, te_probs)
    hist_score = get_hist_score(tr_probs, te_probs)
    outlier_score = get_vocab_outlier(self.tr_vocab, te_vocab)

    metrics = {
      'ks_score': ks_score,
      'hist_score': hist_score,
      'outlier_score': outlier_score,
    }
    return metrics
