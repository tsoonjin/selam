#!/usr/bin/env python
""" Analyze image statistics for Robosub dataset """
import cv2
import numpy as np

from selam.utils import img
from selam import colorconstancy as cc


def analyzeColorSpaces(im, size=(1000, 200)):
    """ Generate stack of individual channels of listed color spaces
        :param size: size of stacked channels
        :return cspaces: image converted to different color spaces
    """
    Lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    XYZ = cv2.cvtColor(im, cv2.COLOR_BGR2XYZ)
    cspaces = {'BGR': im, 'Lab': Lab, 'HSV': HSV, 'YUV': YUV, 'XYZ': XYZ}
    return {k: cv2.resize(np.hstack((c[..., 0], c[..., 1], c[..., 2])), size) for k, c in cspaces.items()}


if __name__ == '__main__':
    path = './examples/dataset/robosub16/tower/3'
    imgs = img.get_jpgs(path)
    for i in imgs:
        sets = analyzeColorSpaces(cc.shadegrey(i))
        cv2.imshow('color spaces', np.vstack((sets['HSV'], sets['Lab'], sets['XYZ'])))
        cv2.waitKey(0)
