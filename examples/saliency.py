#!/usr/bin/env python
""" Explore saliency approach for object proposals """
from __future__ import division
import math
import cv2
import numpy as np
from selam.utils import img
from selam import colorconstancy as cc
from selam import enhancement as en
from lib.pysaliency import saliency, binarise, saliency_mbd


def pysaliency(im):
    """ Performs saliency algorithms listed: robust background detection, frequency tuning,
    minimum barrier
    """
    rbdMap = saliency.get_saliency_rbd(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8)
    rbd = cv2.cvtColor(np.uint8(255 * binarise.binarise_saliency_map(rbdMap, method='adaptive')), cv2.COLOR_GRAY2BGR)
    ftMap = saliency.get_saliency_ft(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8)
    ft = cv2.cvtColor(np.uint8(255 * binarise.binarise_saliency_map(ftMap, method='adaptive')), cv2.COLOR_GRAY2BGR)
    mbdMap = saliency_mbd.get_saliency_mbd(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).astype(np.uint8)
    mbd = cv2.cvtColor(np.uint8(255 * binarise.binarise_saliency_map(mbdMap, method='adaptive')), cv2.COLOR_GRAY2BGR)
    out = np.hstack((im,  rbd, ft, mbd))
    cv2.imshow('sal', cv2.resize(out, (1000, 300)))
    cv2.waitKey(0)


def archantaSaliency(im):
    im = cc.shadegrey(im)
    saliencyMap = [en.get_salient(i) for i in cv2.split(im)]
    cv2.imshow('saliency', np.hstack(saliencyMap))
    cv2.waitKey(0)


def fasa(im, sigmac=16, histogramSize1D=8):
    """ Fast Accurate Size Aware saliency detection
    https://pdfs.semanticscholar.org/672d/c52a5c43714af2af49e02a79c8609afce07f.pdf
        :param sigmac: sigma value used to calculate color weight
        :param histsize: number of histogram bins per channel
    """
    # Initialization
    histogramSize2D = histogramSize1D**2
    histogramSize3D = histogramSize2D * histogramSize1D
    logSize = int(math.log(histogramSize1D, 2))
    logSize2 = 2 * logSize
    squares = (np.arange(10000.0).reshape(1, -1))**2
    modelMean = np.array([[0.5555],
                          [0.6449],
                          [0.0002],
                          [0.0063]])
    modelInverseCovariance= np.array([[43.3777,  1.7633, -0.4059,  1.0997],
                                      [1.7633,  40.7221, -0.0165,  0.0447],
                                      [-0.4059, -0.0165, 87.0455, -3.2744],
                                      [1.0997,   0.0447, -3.2744, 125.1503]])

    def calculateHistogram(im):
        rows, cols = im.shape[:-1]
        LL = []
        AA = []
        BB = []

        averageX = np.zeros((1, histogramSize3D), dtype=np.float32)
        averageY = np.zeros((1, histogramSize3D), dtype=np.float32)
        averageX2 = np.zeros((1, histogramSize3D), dtype=np.float32)
        averageY2 = np.zeros((1, histogramSize3D), dtype=np.float32)

        LAB = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2Lab))
        minL, maxL, _, _ = cv2.minMaxLoc(LAB[0])
        minA, maxA, _, _ = cv2.minMaxLoc(LAB[1])
        minB, maxB, _, _ = cv2.minMaxLoc(LAB[2])

        tempL = (255 - maxL + minL) / (maxL - minL + 1e-3)
        tempA = (255 - maxA + minA) / (maxA - minA + 1e-3)
        tempB = (255 - maxB + minB) / (maxB - minB + 1e-3)

        i = np.arange(256)
        Lshift = np.int32(tempL * (i - minL) - minL)
        Ashift = np.int32(tempA * (i - minA) - minA)
        Bshift = np.int32(tempB * (i - minB) - minB)

        # Calculate quantized LAB value
        minL = minL / 2.56
        maxL = maxL / 2.56
        minA = minA - 128
        maxA = maxA - 128
        minB = minB - 128
        maxB = maxB - 128

        tempL = float(maxL - minL) / histogramSize1D
        tempA = float(maxA - minA) / histogramSize1D
        tempB = float(maxB - minB) / histogramSize1D

        sL = float(maxL - minL) / histogramSize1D / 2 + minL
        sA = float(maxA - minA) / histogramSize1D / 2 + minA
        sB = float(maxB - minB) / histogramSize1D / 2 + minB

        for i in xrange(histogramSize3D):

            lpos = i % histogramSize1D
            apos = i % histogramSize2D / histogramSize1D
            bpos = i / histogramSize2D

            LL.append(lpos * tempL + sL)
            AA.append(apos * tempA + sA)
            BB.append(bpos * tempB + sB)

        # Calculates LAB histogram

        histogramIndex = np.zeros((rows, cols), dtype=np.int32)
        histogram = np.zeros((1, histogramSize3D), dtype=np.int32)
        histShift = 8 - logSize

        for y in xrange(rows):
            lPtr = LAB[0][y]
            aPtr = LAB[1][y]
            bPtr = LAB[2][y]
            for x in xrange(cols):
                lpos = lPtr[x] + Lshift[lPtr[x]] >> histShift
                apos = aPtr[x] + Ashift[aPtr[x]] >> histShift
                bpos = bPtr[x] + Bshift[bPtr[x]] >> histShift

                index = lpos + (apos << logSize) + (bpos << logSize2)
                histogramIndex[y, x] = index
                histogram[0, index] += 1

                averageX[0, index] += x
                averageY[0, index] += y
                averageX2[0, index] += squares[0, x]
                averageY2[0, index] += squares[0, y]

        return averageX, averageY, averageX2, averageY2, LL, AA, BB, histogram, histogramIndex

    averageX, averageY, averageX2, averageY2, LL, AA, BB, \
        histogram, histogramIndex = calculateHistogram(im)
    return im


if __name__ == '__main__':
    path = './examples/dataset/robosub16/buoy/9'
    imgs = img.get_jpgs(path, resize=4)
    for i in imgs:
        pysaliency(cc.shadegrey(i))
