import numpy as np


def vector_diff_ranking_over(diff_vector, percentage):
    """Replaces high diff"""
    if percentage < 1: percentage *= 100
    percentile = np.percentile(diff_vector, 100 - percentage)
    return diff_vector >= percentile


def vector_diff_ranking_under(diff_vector, percentage):
    """Replaces low diff"""
    if percentage < 1: percentage *= 100
    percentile = np.percentile(diff_vector, percentage)
    return diff_vector <= percentile


def magnitude_ranking(weights_vector, percentage):
    if percentage < 1: percentage *= 100
    abs_weights = np.abs(weights_vector)
    percentile = np.percentile(abs_weights, percentage)
    print("{} percentile of magnitudes is {}".format(percentage, percentile))
    return abs_weights <= percentile