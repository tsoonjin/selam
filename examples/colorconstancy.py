#!/usr/bin/env python
""" Performs color transformation """
import cv2
import numpy as np
from selam.utils import img
from selam import colorconstancy as cc
from selam import preprocess as pre


if __name__ == '__main__':
    path = './examples/dataset/robosub16/FRONT/0-264_buoys'
    imgs = img.get_jpgs(path, resize=2)
    augmented = pre.norm_illum_color(cc.greyPixel(imgs[20]), gamma=1.5)
    cv2.imwrite('./examples/output/colorconstancy/greyPixel.jpg', augmented)
