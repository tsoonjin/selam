#!/usr/bin/env python
""" Image segmentation algorithms """
from __future__ import division
import cv2
import numpy as np

from matplotlib import pyplot as plt
from skimage import segmentation, color
from skimage.future import graph
from selam.utils import img
from selam import colorconstancy as cc


def felzenszwalb(im, scale=3.0, sigma=0.95, min_size=5, display=True):
    """ Felzenszwalb oversegmentation
    https://cs.brown.edu/~pff/papers/seg-ijcv.pdf
        :param scale: higher means larger clusters
        :param sigma: width of Gaussian kernel preprocessing
    """
    # Convert to rgb for skimage
    im = cc.shadegrey(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    labels1 = segmentation.felzenszwalb(im, scale=scale, sigma=sigma, min_size=min_size)
    out1 = color.label2rgb(labels1, im, kind='avg')

    if display:
        cv2.imshow('felzenszwalb', cv2.cvtColor(out1, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)


def normalizeCut(im, compactness=0.1, n_segments=300, convert2lab=False, display=True):
    """ Normalize cut on superpixels generated using SLIC
        :param compactness: balance color proximity and space proximity. Higher value better for space.
        :param n_segments: approximate number of labels
        :param convert2lab: whether should be converted to Lab before segmentation
    """
    # Convert to rgb for skimage
    im = cc.shadegrey(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    labels1 = segmentation.slic(im, n_segments, compactness, convert2lab=convert2lab)
    out1 = color.label2rgb(labels1, im, kind='avg')

    g = graph.rag_mean_color(im, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, im, kind='avg')
    if display:
        cv2.imshow('normalizecut', np.hstack((cv2.cvtColor(out1, cv2.COLOR_RGB2BGR),
                                              cv2.cvtColor(out2, cv2.COLOR_RGB2BGR))))
        cv2.waitKey(0)


if __name__ == '__main__':
    path = './examples/dataset/robosub16/buoy/1'
    imgs = img.get_jpgs(path, resize=2)
    for i in imgs:
        modified = i
        felzenszwalb(i)
        cv2.imshow('compare', np.hstack((i, modified)))
        cv2.waitKey(0)
