#!/usr/bin/env python
import skimage
import cv2
import numpy as np

from skimage.feature import blob_dog, blob_log, blob_doh, daisy
from selam import colorconstancy as cc
from lib.moments.zernike import Zernikemoment
from lib.moments import pseudozernike, pyefd
from lib.IDSC import describe
from selam.utils import img


# Feature Detector and Descriptor
# http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#orb

def daisyDetector(im, step=180, rad=58, rings=2, histograms=6, orientations=8, visualize=True):
    """ DAISY
    https://infoscience.epfl.ch/record/138785/files/tola_daisy_pami_1.pdf
        :param im: grayscale image
        :param step: distance between descriptor sampling point
        :param rad: radius of outermost ring
        :param rings: number of rings
        :param histograms: number of histograms per ring
        :param orientations: number of orientations per ring
    """
    descs, descs_img = daisy(im, step, rad, rings, histograms, orientations, visualize=visualize)
    if visualize:
        cv2.imshow('detected', descs_img)
        cv2.waitKey(0)
    return descs


def starDetector(im, canvas, color=(0, 0, 255), rad=2, draw=True):
    """ StarDetector
    http://snorriheim.dnsdojo.com/doku/lib/exe/fetch.php/en:engineering:slam:cslam:censure.pdf
    http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_censure.html#sphx-glr-auto-examples-features-detection-plot-censure-py
        :param im: grayscale image
    """
    detector = cv2.xfeatures2d.StarDetector_create()
    kps = detector.detect(im)
    if draw:
        cv2.drawKeypoints(canvas, kps, canvas, color, rad)
        cv2.imshow('detected', canvas)
        cv2.waitKey(0)
    return kps


def DoH(im, canvas, max_sigma=30, threshold=0.1, display=True):
    """ Difference of Hessian blob detector
        :param im: grayscale image
        :param max_sigma: maximum sigma of Gaussian kernel
        :param threshold: absolute lower bound Local maxima smaller than threshold ignore
    """
    blobs = blob_doh(im, max_sigma=30, threshold=.1)
    for blob in blobs:
        y, x, r = blob
        cv2.circle(canvas, (int(x), int(y)), int(r), (0, 0, 255), 2)

    if display:
        cv2.imshow('Difference of Hessian', canvas)
        cv2.waitKey(0)

    return blobs


def DoG(im, canvas, max_sigma=30, threshold=0.1, display=True):
    """ Difference of Gaussian blob detector
        :param im: grayscale image
        :param max_sigma: maximum sigma of Gaussian kernel
        :param threshold: absolute lower bound Local maxima smaller than threshold ignore
    """
    blobs = blob_dog(im, max_sigma=30, threshold=.1)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    for blob in blobs:
        y, x, r = blob
        cv2.circle(canvas, (int(x), int(y)), int(r), (0, 0, 255), 2)

    if display:
        cv2.imshow('Difference of Gaussian', canvas)
        cv2.waitKey(0)

    return blobs


def LoG(im, canvas, max_sigma=30, num_sigma=10, threshold=0.1, display=True):
    """ Laplacian of Gaussian blob detector
        :param im: grayscale image
        :param max_sigma: maximum sigma of Gaussian kernel
        :param num_sigma: number of sigma to consider between min sigma and max sigma
        :param threshold: absolute lower bound Local maxima smaller than threshold ignore
    """
    blobs = blob_log(im, max_sigma=30, num_sigma=10, threshold=.1)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    for blob in blobs:
        y, x, r = blob
        cv2.circle(canvas, (int(x), int(y)), int(r), (0, 0, 255), 2)

    if display:
        cv2.imshow('Laplacian of Gaussian', canvas)
        cv2.waitKey(0)

    return blobs


def lbp(im, n_points=24, rad=8):
    """ Calculate local binary pattern for a grayscale image
    http://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
        :param im: grayscale image
        :param n_points: number of points considered in a circular neighborhood
        :param rad: radius of neighborhood
        :return: histogram of local binary pattern
    """
    desc = skimage.feature.local_binary_pattern(im, n_points, rad, method='uniform')
    (hist, _) = np.histogram(desc.ravel(), bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def gabor(im, ksize=31, sigma=4.0, theta_step=np.pi / 16, lambd=10.0, gamma=0.5):
    """ Apply Gabor filter
    https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
        :param im: grayscale image
        :param ksize: size of filter
        :param sigma: standard deviation of gaussian
        :param theta: desired orientation for feature extraction
        :param lambd: wavelength of sinusoidal factor
        :param gamma: spatial aspect ratio
    """
    # Divide into theta_step equal parts
    thetas = np.arange(0, np.pi, theta_step)
    filters = []
    features = []

    for theta in thetas:
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0,
                                  ktype=cv2.CV_32F)
        # Normalization step
        kern /= kern.sum()
        filters.append(kern)

    features = [cv2.filter2D(im[..., 0], cv2.CV_8UC3, f) for f in filters]
    return features


def sift(im, canvas, mask=None, draw=True, color=(0, 0, 255), rad=2):
    """ SIFT
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        :param im: grayscale image
    """
    detector = cv2.xfeatures2d.SIFT_create(200)
    kps, descs = detector.detectAndCompute(im, mask=mask)
    if draw:
        cv2.drawKeypoints(canvas, kps, canvas, color, rad)
        cv2.imshow('detected', canvas)
        cv2.waitKey(0)


def surf(im, canvas, mask=None, draw=True, color=(0, 0, 255), rad=2):
    """ SURF
    http://www.vision.ee.ethz.ch/~surf/eccv06.pdf
        :param im: grayscale image
    """
    detector = cv2.xfeatures2d.SURF_create(200)
    kps, descs = detector.detectAndCompute(im, mask=mask)
    if draw:
        cv2.drawKeypoints(canvas, kps, canvas, color, rad)
        cv2.imshow('detected', canvas)
        cv2.waitKey(0)


def freak(im, canvas, mask=None, draw=True, color=(0, 0, 255), rad=2):
    """ FREAK
    https://infoscience.epfl.ch/record/175537/files/2069.pdf
        :param im: grayscale image
    """
    detector = cv2.xfeatures2d.SURF_create(200)
    kps, descs = detector.detectAndCompute(im, mask=mask)
    descriptor = cv2.xfeatures2d.FREAK_create()
    kps, descs = descriptor.compute(im, kps)
    if draw:
        cv2.drawKeypoints(canvas, kps, canvas, color, rad)
        cv2.imshow('detected', canvas)
        cv2.waitKey(0)


def orb(im, canvas, mask=None, draw=True, color=(0, 0, 255), rad=2):
    """ ORB
    http://www.willowgarage.com/sites/default/files/orb_final.pdf
        :param im: grayscale image
    """
    detector = cv2.ORB_create(200)
    kps, descs = detector.detectAndCompute(im, mask=mask)
    if draw:
        cv2.drawKeypoints(canvas, kps, canvas, color, rad)
        cv2.imshow('detected', canvas)
        cv2.waitKey(0)


def brisk(im, canvas, mask=None, draw=True, color=(0, 0, 255), rad=2):
    """ BRISK
    http://e-collection.library.ethz.ch/eserv/eth:7684/eth-7684-01.pdf
        :param im: grayscale image
    """
    detector = cv2.BRISK_create(thresh=10, octaves=1)
    kps, descs = detector.detectAndCompute(im, mask=mask)
    if draw:
        cv2.drawKeypoints(canvas, kps, canvas, color, rad)
        cv2.imshow('detected', canvas)
        cv2.waitKey(0)


def akaze(im, canvas, mask=None, draw=True, color=(0, 0, 255), rad=2):
    """ AKAZE
    http://www.robesafe.com/personal/pablo.alcantarilla/papers/Alcantarilla13bmvc.pdf
        :param im: grayscale image
    """
    detector = cv2.AKAZE_create()
    kps, descs = detector.detectAndCompute(im, mask=mask)
    if draw:
        cv2.drawKeypoints(canvas, kps, canvas, color, rad)
        cv2.imshow('detected', canvas)
        cv2.waitKey(0)


# Shape Descriptor

def imd(im):
    """ Invariant Multi-scale shape descriptor
    http://eeeweba.ntu.edu.sg/computervision/Research%20Papers/2016/Invariant%20Multi-Scale%20Shape%20Descriptor%20for%20Object%20Matching%20and%20Recognition.pdf
    """
    pass


def innerShapeContext(im):
    """ Describes contour using inner distance shape context
    https://www.cs.umd.edu/~djacobs/pubs_files/ID-pami-8.pdf
    https://github.com/brenden/opencv-plant-recognizer
    """
    descriptor = describe.describeLeaf(im)
    return descriptor


def ellipticFourier(im, n=10):
    """ Calculate Elliptic Fourier Descriptor
    http://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Kuhl-Giardina-CGIP1982.pdf
    https://github.com/hbldh/pyefd
        :param im: grayscale image thresholded
        :param n: order of coefficients
    """
    cnts = []
    _, cnts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Remember to convert contour into numpy array with (M x 2) shape
    chosen = np.squeeze(np.array(cnts[0]))
    coeffs = pyefd.elliptic_fourier_descriptors(chosen, order=n)
    print(coeffs)
    return coeffs


def huMoment(im):
    """ Calculate 7 HuMoments
        :param im: grayscale image
    """
    moments = cv2.HuMoments(cv2.moments(im)).flatten()
    return moments


def pseudoZernikeMoment(im, n=4, m=2):
    """ Calculate pseudo-zernike moment which is more robust to noise
    https://github.com/d-klein/image-hash/blob/master/PseudoZernike.py
    Very slow
        :param im: binarized image
        :param n: order of moment
        :param m: repetitions of Zernike moment
    """
    moment = pseudozernike.zmoment(np.float64(im), n, m)
    print(moment)
    return moment


def zernikeMoment(im, n=4, m=2):
    """ Calculate Zernike moment of the image
    To achieve scale and translation invariance, use opencv
    central normalized moment
    https://github.com/primetang/pyzernikemoment
    https://en.wikipedia.org/wiki/Zernike_polynomials
        :param im: binarized image
        :param n: order of moment
        :param m: repetitions of Zernike moment
    """
    Z, A, Phi = Zernikemoment(im, n, m)
    return A, Phi


def computeHOG(im):
    """ Compute Histogram of Oriented Gradient (HOG) """
    hog = cv2.HOGDescriptor()
    feat = hog.compute(im)
    return feat


if __name__ == '__main__':
    path = './benchmark/datasets/torpedo/illumination_change'
    imgs = img.get_jpgs(path, resize=2)
    for i in imgs:
        modified = cc.shadegrey(i)
        akaze(i[..., 0], i)
        akaze(modified[..., 0], modified)
