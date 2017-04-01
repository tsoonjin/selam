#!/usr/bin/env python
""" Explore image enhancement techniques """
from __future__ import division
import cv2
import numpy as np

from selam.utils import img


def dcd(im, n=5, display=True):
    """ Dominant color descriptor using kmeans """
    # Convert to uniform space color space like LUV
    res, labels, centers = img.quantizeColor(cv2.cvtColor(im, cv2.COLOR_BGR2LUV), K=n)
    # Reshaped image
    Z = im.reshape((-1, 3))
    Z_b = Z[..., 0][:, np.newaxis]
    Z_g = Z[..., 1][:, np.newaxis]
    Z_r = Z[..., 2][:, np.newaxis]
    total_p = Z.shape[0]
    # Calculate normalized percentage of pixels for each dominant color
    p = np.array([Z_b[labels == i].size / total_p for i in range(n)])
    # Calculate color variance for each cluster
    v = [np.var(np.dstack((Z_b[labels == i], Z_g[labels == i], Z_r[labels == i])),
         axis=1) for i in range(n)]

    # Convert color space of centers
    c = [cv2.cvtColor(i[np.newaxis, np.newaxis, :], cv2.COLOR_LUV2BGR) for i in centers]
    if display:
        cv2.imshow('DCD', cv2.cvtColor(res, cv2.COLOR_LUV2BGR))
        cv2.waitKey(0)
    return c, p, v


def hmmd(im, display=True):
    """ Convert BGR to HMMD (Hue, Min, Max, Diff, Sum) color space """
    HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im = img.norm(im)
    Max = np.max(im, axis=2)
    Min = np.min(im, axis=2)
    Diff = Max - Min
    Sum = (Max + Min) / 2
    if display:
        cv2.imshow('HMMD', cv2.resize(np.hstack((HSV[..., 0],
                                                 np.uint8(img.normUnity(Max) * 255),
                                                 np.uint8(img.normUnity(Min) * 255),
                                                 np.uint8(img.normUnity(Diff) * 255),
                                                 np.uint8(img.normUnity(Sum) * 255))), (1000, 300)))
        cv2.waitKey(0)
    return img.norm(HSV[..., 0]), Max, Min, Diff, Sum


def logChromacity(im, display=True):
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    lc1 = np.log(R / G)
    lc2 = np.log(R / B)
    lc3 = np.log(G / B)
    if display:
        cv2.imshow('log chromacity', np.hstack((np.uint8(img.normUnity(lc1) * 255),
                                                np.uint8(img.normUnity(lc2) * 255),
                                                np.uint8(img.normUnity(lc3) * 255))))
        cv2.waitKey(0)
    return lc1, lc2, lc3


def luminosity(im, display=True):
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    denom = (R - G)**2 + (R - B)**2 + (G - B)**2
    l1 = (R - G)**2 / denom
    l2 = (R - B)**2 / denom
    l3 = (G - B)**2 / denom

    if display:
        cv2.imshow('rg', np.hstack((np.uint8(img.normUnity(l1) * 255),
                                    np.uint8(img.normUnity(l2) * 255),
                                    np.uint8(img.normUnity(l3) * 255))))
        cv2.waitKey(0)
    return l1, l2, l3


def chromacity(im, display=True):
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    c1 = np.arctan(R, np.max([G, B], axis=0))
    c2 = np.arctan(G, np.max([R, B], axis=0))
    c3 = np.arctan(B, np.max([R, G], axis=0))
    if display:
        cv2.imshow('rg', np.hstack((np.uint8(img.normUnity(c1) * 255),
                                    np.uint8(img.normUnity(c2) * 255),
                                    np.uint8(img.normUnity(c3) * 255))))
        cv2.waitKey(0)
    return c1, c2, c3


def WOpponent(im, display=True):
    O1, O2, O3 = opponent1(im)
    W1 = O1 / O3
    W2 = O2 / O3
    if display:
        cv2.imshow('W opponent', np.hstack((np.uint8(img.normUnity(W1) * 255),
                                            np.uint8(img.normUnity(W2) * 255))))
        cv2.waitKey(0)
    return W1, W2


def bgryOpponent(im, display=False):
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    Ro = R - ((G+B) / 2)
    Go = G - ((R+B) / 2)
    Bo = B - ((R+G) / 2)
    Yo = (R+G) / 2 - np.abs(R - G) - B
    if display:
        cv2.imshow('rg', np.hstack((np.uint8(img.normUnity(Bo) * 255),
                                    np.uint8(img.normUnity(Go) * 255),
                                    np.uint8(img.normUnity(Ro) * 255),
                                    np.uint8(img.normUnity(Ro) * 255))))
        cv2.waitKey(0)
    return Bo, Go, Ro, Yo


def normalizedBGR(im, display=True):
    """ Generate Opponent color space. O3 is just the intensity """
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    b = (B - np.mean(B)) / np.std(B)
    g = (G - np.mean(G)) / np.std(G)
    r = (R - np.mean(R)) / np.std(R)

    if display:
        cv2.imshow('rg', np.hstack((np.uint8(img.normUnity(b) * 255),
                                    np.uint8(img.normUnity(g) * 255),
                                    np.uint8(img.normUnity(r) * 255))))
        cv2.waitKey(0)
    return b, g, r


def opponent2(im, display=True):
    """ Generate Opponent color space """
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    O1 = (R - G) / 2
    O2 = (R + G) / (4 - B / 2)
    if display:
        cv2.imshow('rg', np.hstack((np.uint8(img.normUnity(O1) * 255),
                                    np.uint8(img.normUnity(O2) * 255))))
        cv2.waitKey(0)
    return O1, O2


def opponent1(im, display=False):
    """ Generate Opponent color space. O3 is just the intensity """
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2 * B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)
    if display:
        cv2.imshow('rg', np.hstack((np.uint8(img.normUnity(O1) * 255),
                                    np.uint8(img.normUnity(O2) * 255),
                                    np.uint8(img.normUnity(O3) * 255))))
        cv2.waitKey(0)
    return O1, O2, O3


def rgChromacity(im, display=False):
    im = img.norm(im)
    B, G, R = np.dsplit(im, 3)
    BGR = B + G + R
    # Technically only two channels needed since r + g + b = 1
    r = R / BGR
    g = G / BGR
    b = B / BGR
    if display:
        cv2.imshow('rg', np.hstack((np.uint8(b * 255), np.uint8(g * 255), np.uint8(r * 255))))
        cv2.waitKey(0)
    return r, g, b


if __name__ == '__main__':
    path = './examples/dataset/robosub16/buoy/1'
    imgs = img.get_jpgs(path, resize=2)
    for i in imgs:
        colorStructureDescriptor(i)
        print(i)
