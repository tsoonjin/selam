#!/usr/bin/env python
""" Explore different preprocessing transformation on data before training """
import cv2
import numpy as np
from selam.utils import img
from sklearn import pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale


def incrementalPreprocesser(data, method, batch_size=3):
    """ Fits standard scaling of data incrementally
        data: input data with too many samples
        method: scikit learn preprocessing algorithm
        batch_size: number of samples in one round of preprocessing
    """
    splitted = np.array_split(data, batch_size)
    res = np.empty(data.shape)
    prev = 0
    # Incremetally fitiing data
    for batch in splitted:
        method.partial_fit(batch)

    # Transform individual batch to prevent MemoryError
    for batch in splitted:
        next = prev + batch.shape[0]
        res[prev: next, :] = method.transform(batch)
        prev = next
    return res


def zmuvColorImgs(imgs):
    """ Performs zero mean unit variance on separate color channels """
    mean = np.mean(imgs, axis=0)
    std = np.std(imgs, axis=0)
    return (imgs - mean) / std


def rescaleColorImgs(imgs):
    return imgs / 255.0


def zcaWhitening(imgs):
    """ Performs ZCA Whitening on grayscale image
        row: n_samples
        column: n_features
    """
    # Zero-mean imgs
    imgs -= np.mean(imgs, axis=0)
    imgs = imgs.T
    # Computes covariance matrix. Typically, columns represents observations
    sigma = np.cov(imgs)
    U, S, _ = np.linalg.svd(sigma)
    epsilon = 1e-5
    pcaWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(imgs)
    return U.dot(pcaWhite)


def f_whitening(img, size=(300, 300)):
    """ Performs f-whitening which is more suitable for large natural images. Only work on square matrix.
        size: size of square matrix
    """
    img = np.float32(cv2.resize(img, size))
    whitened = []
    diff = np.rollaxis(img, 2)
    for chan in diff:
        mean = np.mean(chan)
        chan -= mean
        aa = np.fft.fft2(chan)
        spectr = np.sqrt(np.mean(np.dot(abs(aa), abs(aa))))
        out = np.fft.ifft2(np.dot(aa, 1./spectr))
        whitened.append(out)
    res = np.uint8(np.stack(whitened, axis=2))
    return res


def main():
    path = './examples/dataset/robosub16/FRONT/0-264_buoys'
    imgs = np.asarray(img.get_jpgs(path, resize=6), dtype=np.float32)
    # Scaling the pixels to fall within [0 - 1] for numerical stability
    minMaxScaled = rescaleColorImgs(imgs)
    # Standardizing the images
    standardized = zmuvColorImgs(minMaxScaled)
    red_channels = standardized[..., 0]
    new = red_channels.reshape(len(imgs), -1)
    zcaWhitening(new)


if __name__ == '__main__':
    main()
