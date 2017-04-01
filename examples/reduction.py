#!/usr/bin/env python
""" Component analysis and dimension reduction """
from __future__ import division
import cv2
import numpy as np

from sklearn import decomposition
from selam.utils import img


def faDecomposition(imgs, n=6, display=True):
    """ Factor Analysis decomposition
        :param imgs: centered grayscale images
        :param n: number of principal components
    """
    data = imgs.reshape(imgs.shape[0], -1)
    estimator = decomposition.FactorAnalysis(n_components=n, max_iter=2)
    estimator.fit(data)
    components_ = estimator.components_
    if display:
        for c in components_[:n]:
            eigenface = c.reshape(240, -1)
            cv2.imshow('eigen', np.uint8(img.normUnity(eigenface) * 255))
            cv2.waitKey(0)
    return components_


def icaDecomposition(imgs, n=6, display=True):
    """ ICA decomposition
        :param imgs: centered grayscale images
        :param n: number of principal components
    """
    data = imgs.reshape(imgs.shape[0], -1)
    estimator = decomposition.FastICA(n_components=n, whiten=True)
    estimator.fit(data)
    components_ = estimator.components_
    if display:
        for c in components_[:n]:
            eigenface = c.reshape(240, -1)
            cv2.imshow('eigen', np.uint8(img.normUnity(eigenface) * 255))
            cv2.waitKey(0)
    return components_


def pcaDecomposition(imgs, n=6, display=True):
    """ PCA decomposition
        :param imgs: centered grayscale images
        :param n: number of principal components
    """
    data = imgs.reshape(imgs.shape[0], -1)
    estimator = decomposition.RandomizedPCA(n_components=n, whiten=True)
    estimator.fit(data)
    components_ = estimator.components_
    if display:
        for c in components_[:n]:
            eigenface = c.reshape(240, -1)
            cv2.imshow('eigen', np.uint8(img.normUnity(eigenface) * 255))
            cv2.waitKey(0)
    return components_


def centerData(imgs, display=False):
    """ Perform global centering and local centering to the dataset
        :param grayscale images (n images, h, w)
    """
    mean = imgs.mean(axis=0)
    centered = imgs - mean
    if display:
        for i in centered:
            i -= np.mean(i)
            cv2.imshow('centred', np.uint8(img.normUnity(i) * 255))
            cv2.waitKey(0)
    return centered


if __name__ == '__main__':
    path = './examples/dataset/robosub16/buoy/1'
    imgs = img.get_jpgs(path, resize=2)
    grays = np.array([np.mean(i, axis=2) for i in imgs])
    centered = centerData(grays)
    faDecomposition(centered)
