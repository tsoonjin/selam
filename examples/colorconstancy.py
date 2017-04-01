#!/usr/bin/env python
""" Performs color transformation """
import cv2
import numpy as np
from selam.utils import img
from lib.intrinsic import intrinsic


def colorRetinex(im):
    choices = intrinsic.ColorRetinexEstimator.param_choices()
    for j, params in enumerate(choices):
        estimator = intrinsic.ColorRetinexEstimator(**params)
        mask = np.ones_like(im[..., 0])
        est_shading, est_refl = estimator.estimate_shading_refl(im, mask)
        print(est_refl.shape)
        est_refl = img.normUnity(est_refl) * 255
        cv2.imshow('color retinex', np.hstack((np.uint8(est_shading), np.uint8(est_refl))))
        cv2.waitKey(0)


if __name__ == '__main__':
    path = './examples/dataset/robosub16/buoy/1'
    imgs = img.get_jpgs(path, resize=2)
    for i in imgs:
        colorRetinex(i)
