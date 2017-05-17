#!/usr/bin/env python
""" Generate training and validation data from ground truth """
import numpy as np
import sys

from selam import prepdata
from selam.utils import io, img


def test_generate_bin(path="/home/batumon/github/selam/test/training/bin"):
    """ Generate datasets and write to path """
    positives, negatives = prepdata.extract_training(sys.argv[1], sys.argv[2])
    io.write_to_dir(positives, "{}/positives".format(path))
    io.write_to_dir(negatives, "{}/negatives".format(path))


def test_prepare_datasets(path="/home/batumon/github/selam/test/training/bin"):
    positives = img.get_jpgs('{}/positives'.format(path))
    negatives = img.get_jpgs('{}/negatives'.format(path))
    X = positives + negatives
    Y = ['bin'] * len(positives) + ['non_bin'] * len(negatives)
    X_train, Y_train, X_valid, Y_valid = prepdata.kfold(np.array(X), np.array(Y))


if __name__ == '__main__':
    test_prepare_datasets()
