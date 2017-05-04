#!/usr/bin/env python
""" Explore different object proposals algorithm """
import numpy as np
import cv2
from selam.utils import img
from selam import colorconstancy as cc
from selam import preprocess as pre
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_float
from lib.selectivesearch import selectivesearch as ss
from examples import config


""" Segmentation """


def selectiveSearch(im, display=True):
    i = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    row, col = i.shape[:-1]
    img_lbl, regions = ss.selective_search(i, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue
        x, y, w, h = r['rect']
        if w < col - 5 and h < row - 5:
            cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
            candidates.add(r['rect'])

    if display:
        cv2.imshow('selective search', im)
        cv2.waitKey(0)


def superpixelSLIC(img, n_segments=200, sigma=4, compactness=6, max_iter=3):
    """ Generates superpixels based on SLIC
        :param n_segments:    number of superpixels wished to be generated
        :param sigma:         smoothing used prior to segmentation
        :param compactness:   balance between color space proximity and image space proximity.
                              higher more weight to space
        :param max_iter      iterations for kmeans
        :return: image with labels, contours, segments
    """
    image = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    segments = slic(image, n_segments=n_segments, sigma=sigma, compactness=compactness,
                    max_iter=max_iter)

    contours = []
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        if len(cnts) >= 1:
            contours.append(cnts[0])

    out = mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), segments)
    out = cv2.cvtColor(np.uint8(out * 255), cv2.COLOR_RGB2BGR)
    cv2.imwrite('/home/batumon/github/selam/demo/figs/slic.png', out)
    return out, contours, segments


def superpixelSEED(im, n_superpixels=200, n_levels=8, prior=4, n_bins=5, seeds=None, n_iter=20):
    """ Perform SEED superpixel segmentation
        :param n_superpixels: number of superpixels
        :param n_levels: number of block levels, more accurate if higher
        :param prior: enable smoothing. must be [0, 5], higher more smooth
        :param n_bins: number of histogram bins
        :param n_iter: number of iterations
    """
    converted_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, w, c = converted_img.shape

    seeds = cv2.ximgproc.createSuperpixelSEEDS(w, h, c, n_superpixels, n_levels,
                                               prior, n_bins)
    color_img = np.zeros((h, w, 3), np.uint8)
    color_img[:] = (0, 0, 255)

    seeds.iterate(converted_img, n_iter)

    # retrieve the segmentation result
    labels = seeds.getLabels()

    num_label_bits = 2
    labels &= (1 << num_label_bits)-1
    labels *= 1 << (16-num_label_bits)

    mask = seeds.getLabelContourMask(False)

    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(im, im, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)
    cv2.imshow('SEEDS', result)
    cv2.waitKey(0)


def mserSearch(im, canvas):
    """ Using different thresholds on grayscale image to detect different components """
    mser = cv2.MSER_create()
    gray = im
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(canvas, hulls, 3, (0, 0, 255))
    return canvas


def edgeBoxCustom(im, w_diff=1.5):
    # Color space used for edge detection
    a, b, c = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2XYZ))
    a_edge = cv2.cvtColor(np.uint8(cv2.Canny(a, 0, 250)), cv2.COLOR_GRAY2BGR)
    b_edge = cv2.cvtColor(np.uint8(cv2.Canny(b, 0, 250)), cv2.COLOR_GRAY2BGR)
    c_edge = cv2.cvtColor(np.uint8(cv2.Canny(c, 0, 250)), cv2.COLOR_GRAY2BGR)
    cv2.imshow('img', cv2.resize(np.hstack((im, a_edge, b_edge, c_edge)), (1200, 400)))
    cv2.waitKey(0)


def main():
    path = './benchmark/datasets/buoy/size_change'
    imgs = img.get_jpgs(path, resize=2)
    for i in imgs:
        i = cc.shadegrey(i)
        Lab = cv2.cvtColor(i, cv2.COLOR_BGR2Lab)
        YUV = cv2.cvtColor(i, cv2.COLOR_BGR2YUV)
        r = mserSearch(i[..., 2], cv2.cvtColor(i[..., 2], cv2.COLOR_GRAY2BGR))
        a = mserSearch(Lab[..., 1], cv2.cvtColor(Lab[..., 1], cv2.COLOR_GRAY2BGR))
        u = mserSearch(YUV[..., 1], cv2.cvtColor(YUV[..., 1], cv2.COLOR_GRAY2BGR))
        res = np.hstack((i, r, a, u))
        cv2.imshow('final', res)
        cv2.waitKey(0)



if __name__ == '__main__':
    main()
