#!/usr/bin/env python
""" Explore image enhancement techniques """
from __future__ import division
import cv2
import numpy as np
import scipy.fftpack

from skimage import restoration
from examples import config
from selam.utils import img
from selam import colorconstancy as cc
from selam import enhancement as en
from lib.DCP import DarkChannelRecover as dcp


def darkChannelPrior(im, display=True):
    radiance = dcp.getRecoverScene(im)
    if display:
        cv2.imshow('dehazed', np.hstack((im, radiance)))
        cv2.waitKey(0)


def taiwanLightCompensation(im, theta=0.5, thresh=0.2, B=[10, 30]):
    """ Perform light compensation
    http://www.csie.ntu.edu.tw/~fuh/personal/LightCompensation.pdf
        :param theta: weight of corrected image in final output
        :return: light compensated image
    """

    def getOptimalB(im, thresh, B):
        """ Determine optimal B for logarithmic correction
            :param thresh: threshold to count dark pixels
            :param limit: [B_min, B_max]
        """
        HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        v = img.norm(HSV[..., 2])
        # Total number of pixels
        N_total = v.size
        N_dark = v[v < thresh].size
        B_optimal = (B[1] - B[0]) * (N_dark / N_total) + B[0]
        return B_optimal

    I_corrected = logCorrection(im, getOptimalB(im, thresh, B))
    final = I_corrected * (1 - theta) + im * theta
    return np.uint8(final)


def chenLightCompensation(im, scaler=15, power=0.5):
    """ Two stage brightness compensation by H.T.Chen National Taiwan University
        :param scaler: strength of brightening effect
        :param power:  strenght of darken effect
        :return: brighten image, darken image
    """

    def brighten(im, scaler):
        HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        v = img.norm(HSV[..., 2])
        brighter = img.normUnity(scaler * np.log1p(2 - v) * v)
        v_scaled = np.uint8(brighter * 255)
        return cv2.cvtColor(cv2.merge((HSV[..., 0], HSV[..., 1], v_scaled)), cv2.COLOR_HSV2BGR)

    def darken(im, power):
        HSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        s = img.norm(HSV[..., 1])
        v = img.norm(HSV[..., 2])
        avgV = cv2.blur(v, (5, 5))
        avgS = cv2.blur(s, (5, 5))
        D_saturation = (avgS + 1) / 2
        D_brightness = 1 / ((1 + avgV)**power)
        v_scaled = D_brightness * D_saturation
        v_scaled = np.uint8(v_scaled * v * 255)
        return cv2.cvtColor(cv2.merge((HSV[..., 0], HSV[..., 1], v_scaled)), cv2.COLOR_HSV2BGR)

    I_bright = brighten(im, scaler)
    I_darken = darken(im, power)
    return I_bright, I_darken


def logCorrection(im, B=3.0):
    """ Brightness correction using log curve which obeys Weber-Fechner
    law of JND response in human
    """
    im = img.norm(im)
    corrected = np.log(im * (B-1) + 1) / np.log(B)
    return np.uint8(corrected * 255)


def enhanceFusion(im):
    """ Effective Single Underwater Image Enhancement by Fusion
    http://www.jcomputers.us/vol8/jcp0804-10.pdf
    """

    def genImgPyramid(im, n_scales=6):
        """ Generate Gaussian and Laplacian pyramid
            :param n_scales: number of pyramid layers
            :return: Gaussian pyramid, Laplacian pyramid
        """
        G = im.copy()
        gp = [G]
        # Generate Gaussian Pyramid
        for i in range(n_scales):
            G = cv2.pyrDown(G)
            gp.append(G)

        lp = [gp[n_scales-1]]
        # Generate Laplacian Pyramid
        for i in range(n_scales - 1, 0, -1):
            size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
            GE = cv2.pyrUp(gp[i], dstsize=size)
            L = cv2.subtract(gp[i - 1], GE)
            lp.append(L)

        return gp, lp

    def saliencyMap(im):
        Lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
        Lab_blur = cv2.GaussianBlur(Lab, (5, 5), 20)
        Lab_blur = img.norm(Lab_blur)
        Lab = img.norm(Lab)
        I_mean = np.zeros_like(im)
        I_mean[..., 0] = np.mean(Lab[..., 0])
        I_mean[..., 1] = np.mean(Lab[..., 1])
        I_mean[..., 2] = np.mean(Lab[..., 1])
        saliencyMap = np.linalg.norm(I_mean - Lab_blur, axis=2)
        return saliencyMap

    def chromaticMap(im, sigma=0.3):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv = img.norm(hsv)
        s = hsv[..., 1]
        num = - (s - np.max(s))**2
        chromaticMap = np.exp(num / (2 * sigma * sigma))
        return chromaticMap

    def luminanceMap(im):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv = img.norm(hsv)
        bgr = img.norm(im)
        b, g, r = np.dsplit(bgr, 3)
        h, s, v = np.dsplit(hsv, 3)
        luminanceMap = np.std([b, g, r, v], axis=0)
        return np.squeeze(luminanceMap, axis=(2,))

    def generateMaps(im):
        lMap = img.normUnity(luminanceMap(im))
        cMap = img.normUnity(chromaticMap(im))
        sMap = img.normUnity(saliencyMap(im))
        maps = [cMap, lMap, sMap]
        out = np.hstack((np.uint8(cMap * 255), np.uint8(lMap * 255), np.uint8(sMap * 255)))
        return maps, out

    input1 = cc.shadegrey(im)
    input2 = en.claheColor(cv2.cvtColor(input1, cv2.COLOR_BGR2Lab))
    input2 = cv2.cvtColor(input2, cv2.COLOR_Lab2BGR)
    input1 = im

    maps1, out1 = generateMaps(input1)
    maps2, out2 = generateMaps(input2)
    sumCMap = maps1[0] + maps2[0] + 0.0001
    sumLMap = maps1[1] + maps2[1] + 0.0001
    sumSMap = maps1[2] + maps2[2] + 0.0001

    normCMap1 = maps1[0] / sumCMap
    normLMap1 = maps1[1] / sumLMap
    normSMap1 = maps1[2] / sumSMap

    normCMap2 = maps2[0] / sumCMap
    normLMap2 = maps2[1] / sumLMap
    normSMap2 = maps2[2] / sumSMap

    finalMap1 = img.normUnity(np.sum([normCMap1, normLMap1, normSMap1], axis=0))
    finalMap1 = np.repeat(finalMap1[:, :, np.newaxis], 3, axis=2)
    finalMap2 = img.normUnity(np.sum([normCMap2, normLMap2, normSMap2], axis=0))
    finalMap2 = np.repeat(finalMap2[:, :, np.newaxis], 3, axis=2)
    gp1, _ = genImgPyramid(finalMap1)
    gp2, _ = genImgPyramid(finalMap2)
    _, lp1 = genImgPyramid(input1)
    _, lp2 = genImgPyramid(input2)
    f = []
    for i in xrange(6):
        f1 = gp1[5 - i] * lp1[i]
        f2 = gp2[5 - i] * lp2[i]
        res = np.uint8(0.5 * f1 + 0.5 * f2)
        f.append(res)

    ls_ = f[0]
    for i in xrange(1, 6):
        size = (f[i].shape[1], f[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, f[i])
    return maps1, maps2, finalMap1, finalMap2, ls_


def homomorphicFilterColor(im):
    """ Perform homomorphic filter on color image """
    a, b, c = cv2.split(im)
    a_filt = homomorphicFilter(a)
    b_filt = homomorphicFilter(b)
    c_filt = homomorphicFilter(c)
    return cv2.merge((a_filt, b_filt, c_filt))


def homomorphicFilter(im):
    """ Homomorphic filtering on single channel image
    http://stackoverflow.com/questions/24731810/segmenting-license-plate-characters
    """
    # Normalizes image to [0, 1]
    im = img.norm(im)
    rows, cols = im.shape
    # Convert to log(1 + I)
    imgLog = np.log1p(im)

    # Create Gaussian mask of sigma = 10
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1*Ioutlow[0:rows, 0:cols] + gamma2*Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")

    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2.copy()
    Ithresh[Ithresh < 65] = 255

    # Clear off the border.  Choose a border radius of 5 pixels
    Iclear = img.clearborder(Ithresh, 5)

    # Eliminate regions that have areas below 120 pixels
    Iopen = img.bwareaopen(Iclear, 120)

    return Ihmf2


def anisodiffColor(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.),
                   option=1):
    """ Anisotropic diffusion on color image by performing diffusion
    on independent color channels
    """
    a, b, c = cv2.split(img)
    a_diff = anisodiff(a, niter, kappa, gamma, step, option)
    b_diff = anisodiff(b, niter, kappa, gamma, step, option)
    c_diff = anisodiff(c, niter, kappa, gamma, step, option)
    return np.uint8(cv2.merge((a_diff, b_diff, c_diff)))


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.),
              option=1):
    """
    Anisotropic diffusion on single channel image

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in xrange(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
                gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
                gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma*(NS+EW)

    return imgout


if __name__ == '__main__':
    path = './examples/dataset/robosub16/torpedo/2'
    imgs = img.get_jpgs(path, resize=2)
    for i in imgs:
        darkChannelPrior(i)
