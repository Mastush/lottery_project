import numpy as np


def diff_ranking_over(diff_kernels, diff_bias, percentage):
    if percentage < 1: percentage *= 100

    diff_weights = np.concatenate((diff_kernels.flatten(), diff_bias.flatten()))
    percentile = np.percentile(diff_weights, 100 - percentage)
    return diff_weights >= percentile


def vector_diff_ranking_over(diff_vector, percentage):
    if percentage < 1: percentage *= 100
    percentile = np.percentile(diff_vector, 100 - percentage)
    return diff_vector >= percentile


def vector_diff_ranking_under(diff_vector, percentage):
    if percentage < 1: percentage *= 100
    percentile = np.percentile(diff_vector, 100 - percentage)
    return diff_vector <= percentile


def magnitude_ranking(weights_vector, percentage):
    if percentage < 1: percentage *= 100
    abs_weights = np.abs(weights_vector)
    percentile = np.percentile(abs_weights, 100 - percentage)
    return abs_weights <= percentile