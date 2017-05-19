#!/usr/bin/env python
""" Generate training and validation data from ground truth """
import os
import numpy as np
import sys

from selam import prepdata
from selam.utils import io, img


def test_generate_bin(path="{}/github/selam/test/training/bin".format(os.environ['HOME'])):
    """ Generate datasets and write to path """
    positives, negatives = prepdata.extract_training(sys.argv[1], sys.argv[2])
    io.write_to_dir(positives, "{}/positives".format(path))
    io.write_to_dir(negatives, "{}/negatives".format(path))


def test_prepare_datasets(path="{}/github/selam/test/training/bin".format(os.environ['HOME'])):
    positives = img.get_jpgs('{}/positives'.format(path))
    negatives = img.get_jpgs('{}/negatives'.format(path))
    X = np.array(positives + negatives)
    X = X.reshape((len(X), -1))
    Y = np.array([1] * len(positives) + [0] * len(negatives))
    return prepdata.kfold(X, Y)


def test_preprocess_datasets(X_train):
    X = prepdata.std_minmax(X_train)
    return prepdata.reduce_nmf(X, 100, 100)


def test_classifier_svm(X_train, Y_train):
    clf = prepdata.classify_svm(X_train, Y_train)


def test_classifier_rf(X_train, Y_train):
    clf = prepdata.classify_rf(X_train, Y_train)


def test_classifier_gp(X_train, Y_train):
    clf = prepdata.classify_gp(X_train, Y_train)


def test_classifier_xgb(X_train, Y_train):
    clf = prepdata.classify_xgb(X_train, Y_train)


if __name__ == '__main__':
    X_train, X_valid, Y_train, Y_valid = test_prepare_datasets()
    X_reduced = test_preprocess_datasets(X_train)
    clf = test_classifier_xgb(X_reduced, Y_train)
