#!/usr/bin/env python
""" Color constancy algorithm to achieve illuminance-invariance """
import math
import random
import cv2
import numpy as np

from selam.utils import img
from scipy.ndimage import filters


def colorRabbit(im, N=5, n=100, upperBound=1.0, rowsStep=50, colsStep=50, ksize=(9, 9), f=1/3, d=0.05):
    """ Color Rabbit:
    http://www.fer.unizg.hr/_download/repository/Color_Rabbit_-_Guiding_the_Distance_of_Local_Maximums_in_Illumination_Estimation.pdf
        :param N:          Number of per pixel estimations.
        :param n:          Number of circle checking points.
        :param upperBound: Maximal value for a pixel channel.
        :param rowsStep:   Rows counting step.
        :param colsStep:   Columns counting step.
        :param ksize:      Size of the averaging kernel to be used.
        :param R:          Maximal radius length.
        :param d:          Distribution control parameter.
        :return:           normalized estimated illumination
    """

    def isInBound(x, y, xlim, ylim):
        return x >= 0 and x < xlim and y >= 0 and y < ylim

    def getNextPoint(row, col, r, j, n):
        newRow = int(row + r * np.sin(2 * j * np.pi / n))
        newCol = int(col + r * np.sin(2 * j * np.pi / n))
        return [newRow, newCol]

    def angularError(pixel):
        """ Calculate angular error of a (b, g, r) vector """
        angle = np.clip(np.sum(pixel) / (3 * (pixel[0]**2 + pixel[1]**2 + pixel[2]**2)), -1.0, 1.0)
        return np.arccos(angle)

    im = img.norm(im)
    rows, cols = im.shape[:-1]
    outRows = int(rows / rowsStep)
    outCols = int(cols / colsStep)
    resized = np.zeros((outRows, outCols, 3), dtype=np.float64)
    dest = np.zeros((outRows, outCols, 3), dtype=np.float64)
    R = np.sqrt(rows**2 + cols**2) * f
    errors = []

    for outRow in xrange(0, outRows):
        for outCol in xrange(0, outCols):
            row = outRow * rowsStep
            col = outCol * colsStep
            current_pt = im[row, col]
            final_pt = np.zeros((1, 3), dtype=np.float64)
            for ti in xrange(N):
                p = random.uniform(0.0, 1.0)
                p = p**d
                r = p * R
                new_indices = [getNextPoint(row, col, r, j, n) for j in xrange(n)]
                new_pixels = np.array([im[y, x] for y, x in new_indices if isInBound(y, x, rows, cols)])
                if new_pixels.size is not 0:
                    max_pixel = np.max(new_pixels, axis=0)
                    errors.append(max_pixel)

            # Sort pixels by error
            sorted_pixels = sorted(errors, key=lambda e: angularError(e))
            max_pixel = sorted_pixels[int(N / 2)]
            max_pixel[max_pixel == 0] = 1.0
            final_pt += (current_pt / max_pixel)
            final_pt[(final_pt > 1.0)] = 1.0
            resized[outRow, outCol] = current_pt
            dest[outRow, outCol] = final_pt
            if (final_pt == 0).any():
                final_pt = 1.0
                resized = 0.0

    # Apply average filter
    # resized = filters.generic_filter(resized, function=np.mean, size=ksize)
    # dest = filters.generic_filter(dest, function=np.mean, size=ksize)
    illumination = resized / dest
    b_est = np.mean(illumination[..., 0])
    g_est = np.mean(illumination[..., 1])
    r_est = np.mean(illumination[..., 2])

    b = im[..., 0]
    g = im[..., 1]
    r = im[..., 2]

    b_corrected = np.uint8(b / (b_est * np.sqrt(3)) * 255)
    g_corrected = np.uint8(g / (g_est * np.sqrt(3)) * 255)
    r_corrected = np.uint8(r / (r_est * np.sqrt(3)) * 255)
    return cv2.merge((b_corrected, g_corrected, r_corrected))


def bgr2ill(im):
    """ Converts BGR image to illumination invariant
    https://www.cs.harvard.edu/~sjg/papers/cspace.pdf
    """
    # Converts to XYZ color space
    XYZ = cv2.cvtColor(im, cv2.COLOR_BGR2XYZ)
    X = img.norm(XYZ[..., 0])
    Y = img.norm(XYZ[..., 1])
    Z = img.norm(XYZ[..., 2])
    denom = (X + Y + Z + 0.0001)
    # Converts to xyz color space
    x = X / denom
    y = Y / denom
    z = 1 - x - y
    xyz = np.dstack((x, y, z))
    # xyz to ill conversion
    B = np.array([[ 0.9465229,   0.2946927,  -0.1313419],
                  [-0.1179179,   0.9929960,   0.007371554],
                  [0.09230461,  -0.04645794,  0.9946464]])

    A = np.array([[ 27.07439,  -22.80783,  -1.806681],
                  [-5.646736,  -7.722125,  12.86503],
                  [-4.163133,  -4.579428,  -4.576049]])

    xyzVec = xyz.reshape(xyz.shape[0] * xyz.shape[1], 3).T
    illVec = np.dot(A, np.log(np.dot(B, xyzVec)))
    ill = (illVec.T).reshape(xyz.shape[0], xyz.shape[1], 3)
    # Normalizes to 255
    norm = np.uint8((ill - np.min(ill)) / (np.max(ill) - np.min(ill)) * 255)
    norm0 = norm[..., 0]
    norm1 = norm[..., 1]
    norm2 = norm[..., 1]
    return norm


def greyPixel(im, prc=0.0001):
    """ Grey pixel illumination estimation framework
    http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yang_Efficient_Illuminant_Estimation_2015_CVPR_paper.pdf
        :param prc: percentage of n pixels chosen for estimation
        :return:    estimated illumination
    """

    def avgFilter(im, ksize=(7, 7)):
        return filters.generic_filter(im, function=np.mean, size=ksize)

    def localStdIIM(im, ksize=(3, 3)):
        """ Calculates local standard deviation of a single channel image for illumination invariant measure (IIM)
        :param ksize: patch size
        """
        return filters.generic_filter(im, function=np.std, size=ksize)

    im = img.norm(im)
    b = im[..., 0]
    g = im[..., 1]
    r = im[..., 2]
    L = np.mean(im, axis=2)
    log_im = np.log(im)
    # Luminance
    # Calculates local standard deviation for each channels
    b_std = localStdIIM(log_im[..., 0])
    g_std = localStdIIM(log_im[..., 1])
    r_std = localStdIIM(log_im[..., 2])
    # Calculate relative standard deviation
    p = np.std([b_std, g_std, r_std], axis=0)
    # Grey index map
    GI = avgFilter(p / L).flatten()
    N = GI.shape[0]
    sorted_indices = np.argsort(GI)
    selected_indices = sorted_indices[0:int(math.ceil(prc * N))]
    b_est = np.mean((b.flatten())[selected_indices])
    g_est = np.mean((g.flatten())[selected_indices])
    r_est = np.mean((r.flatten())[selected_indices])
    b_corrected = b / b_est
    g_corrected = g / g_est
    r_corrected = r / r_est
    sum = b_est + g_est + r_corrected
    b_corrected = np.uint8(b_corrected / sum * 255)
    g_corrected = np.uint8(g_corrected / sum * 255)
    r_corrected = np.uint8(r_corrected / sum * 255)
    return cv2.merge((b_corrected, g_corrected, r_corrected))


def bgr2HSy(im):
    """ Derive HSy color space that is illumination invariant
    https://ai2-s2-pdfs.s3.amazonaws.com/a8fc/277c217c2b5e8287591d852042f4b132cf65.pdf
    """
    B, G, R = cv2.split(im)
    im = img.norm(im)
    B = im[..., 0]
    G = im[..., 1]
    R = im[..., 2]
    H = (R - 0.5*G - 0.5*B) / np.sqrt(R**2 + G**2 + B**2 - (R*G) - (R*B) - (B*G))
    H = H.clip(min=-1.0, max=1.0)
    H = np.uint8(np.arccos(H) / np.arccos(-1.0) * 255)
    denom = B + G + R
    S = np.uint8(1 - (3*(np.min(im, axis=2)) / denom) * 255)
    Y = 0.2125 * R + 0.7154 * G + 0.0721 * R
    y = np.uint8(Y / denom * 255)
    return H, S, y


def normalizedBGR(im):
    """ Converts linear BGR to normalized BGR """
    im = img.norm(im)
    b, g, r = cv2.split(im)
    denom = np.sqrt(b**2 + g**2 + r**2)
    b_corrected = np.uint8(b / denom * 255)
    g_corrected = np.uint8(g / denom * 255)
    r_corrected = np.uint8(r / denom * 255)
    return cv2.merge((b_corrected, g_corrected, r_corrected))


def greyEdge(im):
    """ Performs grey-edge color constancy
    http://lear.inrialpes.fr/people/vandeweijer/papers/ColorConstancyIP.pdf
    """

    def norm_derivative(im, sigma=2):
        x = filters.gaussian_filter(im, sigma=sigma, order=[1, 0])
        y = filters.gaussian_filter(im, sigma=sigma, order=[0, 1])
        mag = np.sqrt(x**2 + y**2)
        return mag

    im = img.norm(im)
    b, g, r = cv2.split(im)
    size = im.shape[0] * im.shape[1]
    db = norm_derivative(b)
    dg = norm_derivative(g)
    dr = norm_derivative(r)
    b_est = ((np.sum(db**(6)))**(1/6.0)) / size
    g_est = ((np.sum(dg**(6)))**(1/6.0)) / size
    r_est = ((np.sum(dr**(6)))**(1/6.0)) / size
    som = np.sqrt(b_est**2 + g_est**2 + r_est**2)
    b_est /= som
    g_est /= som
    r_est /= som
    b_corrected = np.uint8(b / (b_est * np.sqrt(3)) * 255)
    g_corrected = np.uint8(g / (g_est * np.sqrt(3)) * 255)
    r_corrected = np.uint8(r / (r_est * np.sqrt(3)) * 255)
    return cv2.merge((b_corrected, g_corrected, r_corrected))


def shadegrey(img):
    """ Minkowski P-Norm Shades of Grey """

    def shade_grey_est(grayimg):
        size = grayimg.size
        power = np.power(np.float32(grayimg), 6)
        normalized_p_norm = np.power(np.sum(power) / size, 1 / 6.0)
        return normalized_p_norm

    b, g, r = cv2.split(img)
    illumination_est = np.mean([shade_grey_est(x) for x in [b, g, r]]) + 0.0001
    b_corrected = illumination_est / float(np.mean(b) + 0.001) * b
    b_corrected = b_corrected.clip(max=240)
    g_corrected = illumination_est / float(np.mean(g) + 0.001) * g
    g_corrected = g_corrected.clip(max=240)
    r_corrected = illumination_est / float(np.mean(r) + 0.001) * r
    r_corrected = r_corrected.clip(max=240)
    return cv2.merge((np.uint8(b_corrected), np.uint8(g_corrected), np.uint8(r_corrected)))


def logChromaNorm(img):
    """ Performs log-chromacity non-iterative normalization """
    b, g, r = cv2.split(img)
    b = np.float32(b)
    g = np.float32(g)
    r = np.float32(r)
    log_b = cv2.log(b)
    log_g = cv2.log(g)
    log_r = cv2.log(r)
    b = cv2.exp(log_b - cv2.mean(log_b)[0])
    g = cv2.exp(log_g - cv2.mean(log_g)[0])
    r = cv2.exp(log_r - cv2.mean(log_r)[0])
    b = cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255
    g = cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255
    r = cv2.normalize(r, 0, 255, cv2.NORM_MINMAX) * 255
    b = b.clip(max=255)
    g = g.clip(max=255)
    r = r.clip(max=255)
    return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))


def finlayNorm(img, cycle=2):
    """ Performs non-iterative comprehensive normalization by Finlaysson """
    b, g, r = cv2.split(img)
    b = np.float32(b)
    g = np.float32(g)
    r = np.float32(r)
    # Prevent division by 0
    b = b / (b + g + r + 0.001) * 255
    g = g / (b + g + r + 0.001) * 255
    r = r / (b + g + r + 0.001) * 255
    bmean = np.mean(b)
    gmean = np.mean(g)
    rmean = np.mean(r)
    b = b / bmean
    g = g / gmean
    r = r / rmean
    b = cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255
    g = cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255
    r = cv2.normalize(r, 0, 255, cv2.NORM_MINMAX) * 255
    b = b.clip(max=255)
    g = g.clip(max=255)
    r = r.clip(max=255)
    out = cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
    return out


def chromaIterNorm(img, cycle=2):
    """ Performs chromatic iterative comprehensive normalization """
    b, g, r = cv2.split(img)
    b = np.float32(b)
    g = np.float32(g)
    r = np.float32(r)
    for i in xrange(cycle):
        b = b / (b + g + r) * 255
        g = g / (b + g + r) * 255
        r = r / (b + g + r) * 255
    out = cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
    return out


def sodMinkowski(img):
    """ Minkowski P-Norm Shades of Grey """
    b, g, r = cv2.split(img)
    gray = np.mean([np.mean(b), np.mean(g), np.mean(r)])
    gray = np.power(gray, 1 / 6.0)
    r = gray / np.mean(r) * r
    r = np.uint8(cv2.normalize(r, 0, 255, cv2.NORM_MINMAX) * 255)
    g = gray / np.mean(g) * g
    g = np.uint8(cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255)
    b = gray / np.mean(b) * b
    b = np.uint8(cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255)
    return cv2.merge((b, g, r))


def sodNorm1(img):
    """ Shades of gray norm 1 """
    b, g, r = cv2.split(img)
    gray = np.max([np.mean(b), np.mean(g), np.mean(r)])
    r = cv2.normalize(gray / np.mean(r) * r, 0, 255, cv2.NORM_MINMAX) * 255
    b = cv2.normalize(gray / np.mean(b) * b, 0, 255, cv2.NORM_MINMAX) * 255
    g = cv2.normalize(gray / np.mean(g) * g, 0, 255, cv2.NORM_MINMAX) * 255
    return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))


def greyWorld(img):
    """ Calculate grey world corrected image """
    b, g, r = cv2.split(img)
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    gray = np.mean([r_mean, b_mean, g_mean])
    gray = 0.5 + 0.2 * gray
    b = gray / b_mean * b
    g = gray / g_mean * g
    r = gray / r_mean * r
    b = b.clip(max=255)
    g = g.clip(max=255)
    r = r.clip(max=255)
    return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))


def logChromacity(img):
    """ Calculates log-chromacity image """
    b, g, r = cv2.split(img)
    b = np.float32(b)
    g = np.float32(g)
    r = np.float32(r)
    sum = cv2.pow(b + g + r + 0.1, 1 / 3.0)
    b = b / sum
    g = g / sum
    r = r / sum
    b = cv2.log(b)
    g = cv2.log(g)
    r = cv2.log(r)
    b = cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255
    g = cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255
    r = cv2.normalize(r, 0, 255, cv2.NORM_MINMAX) * 255
    out = cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
    return out


def illumInvariantExtraction(im, alpha=0.7):
    """ Extracts illumination invariant image representation
    http://www.robots.ox.ac.uk/~mobile/Papers/2014ICRA_maddern.pdf
        :param alpha: camera dependent parameter based on spectral response
        :return:      grayscale illumination invariant image
    """
    im = img.norm(im)
    log_im = np.log(im)
    ii_image = 0.5 + log_im[..., 1] - (alpha * log_im[..., 0]) - ((1 - alpha) * log_im[..., 2])
    ii_image = ii_image.clip(min=0, max=1)
    cv2.imshow('ii', np.uint8(ii_image * 255))
    cv2.waitKey(0)


def spatialColorConstancy(im):
    """ Use spatial domain method for illumination estimation """

    def illuminantEstimator(im, prc=0.035):
        """ Calculates estimated illumination
        http://www.comp.nus.edu.sg/~whitebal/illuminant/files/illuminantEstimator.m
            :return: estimated illumination, corrected image
        """
        # Number of pixels
        n = im.shape[0] * im.shape[1]
        data = im.reshape(-1, 3)
        # Grey world estimation
        l = np.mean(data, axis=0)
        l /= np.linalg.norm(l)
        # Projection on grey world estimation
        data_p = np.sum(data * l, axis=1)
        sorted_indices = np.argsort(data_p)
        bot_idx = sorted_indices[0: int(math.ceil(n * prc))]
        top_idx = sorted_indices[int(math.ceil(n * (1 - prc))):]
        bot_pixels = data[bot_idx]
        top_pixels = data[top_idx]
        data_selected = np.concatenate((bot_pixels, top_pixels), axis=0)
        sigma = np.dot(data_selected.T, data_selected)
        _, _, V = np.linalg.svd(sigma, full_matrices=True)
        ei = np.abs(V[0, :])
        return ei

    im = img.norm(im)
    b = im[..., 0]
    g = im[..., 1]
    r = im[..., 2]
    ei = illuminantEstimator(im)
    b /= ei[0]
    g /= ei[1]
    r /= ei[2]
    sum = b + g + r
    b_corrected = np.uint8(b / sum * 255.0)
    g_corrected = np.uint8(g / sum * 255.0)
    r_corrected = np.uint8(r / sum * 255.0)
    corrected = cv2.merge((b_corrected, g_corrected, r_corrected))
    return ei, corrected


def whitePatchRetinex(im):
    """ Color correct image based on Retinex algorithm by Land and McCann """
    b, g, r = cv2.split(im)
    # Estimated white point for each channels
    b_max = np.max(b)
    g_max = np.max(g)
    r_max = np.max(r)

    b_corrected = np.uint8(np.float32(b) / b_max * 255.0)
    g_corrected = np.uint8(np.float32(g) / g_max * 255.0)
    r_corrected = np.uint8(np.float32(r) / r_max * 255.0)
    return cv2.merge((b_corrected, g_corrected, r_corrected))


def colorConstancyGamma(im):
    """ Performs color constancy and gamma correction
    http://www.aacademica.org/jcepedanegrete/3.pdf
        :return: retinex corrected, grey world corrected
    """

    def greyWorld(img):
        b, g, r = cv2.split(img)
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        gray = np.mean([r_mean, b_mean, g_mean])
        gray = 0.5 + 0.2 * gray
        b = gray / b_mean * b
        g = gray / g_mean * g
        r = gray / r_mean * r
        b = b.clip(max=255)
        g = g.clip(max=255)
        r = r.clip(max=255)
        return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))

    def sRGB(im):
        """ Converts Linear RGB to sRGB with gamma correction """
        im = img.norm(im)
        corrected = np.where(im > 0.0031308, 1.055 * im**(1/2.4) - 0.055, im * 12.92)
        return np.uint8(corrected * 255)

    return sRGB(whitePatchRetinex(im)), sRGB(greyWorld(im))


def chromaticAdaptationLab(im):
    """ Transform pixels to estimated white point in Lab color space
    https://www.researchgate.net/profile/Guy_Kloss/publication/228945215_Colour_Constancy_using_von_Kries_Transformations_Colour_Constancy_goes_to_the_Lab/links/0c96051f322e2bef6d000000.pdf?origin=publication_detail
        :return: white patch retinex corrected image, grey world corrected image
    """

    def whitePatchLab(im, percentage=0.04):
        """ Performs white patch retinex in Lab color space
            :return: a estimate, b estimate
        """
        l, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2Lab))
        n_pixels = im.shape[0] * im.shape[1]
        cutoff_index = int((1 - percentage) * n_pixels)
        # Sort image by lightness
        sorted_img = img.sort_by_chan(im, 0)
        # Number of pixels taken into calculation
        selected = sorted_img[cutoff_index:, :]
        # Estimated illumination
        l_est = np.mean(selected[..., 0])
        a_est = np.mean(selected[..., 1])
        b_est = np.mean(selected[..., 2])
        return l_est, a_est, b_est

    def greyWorldLab(im, percentage=0.02):
        """ Performs grey world illumination estimation in Lab color space
            :return: l estimate, a estimate, b estimate
        """
        l, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2Lab))
        average_l = np.mean(l)
        average_a = np.mean(a)
        average_b = np.mean(b)
        n_pixels = im.shape[0] * im.shape[1]
        cutoff_index = int((1 - percentage) * n_pixels)
        # Sort image by lightness
        sorted_img = img.sort_by_chan(im, 0)
        max_l = sorted_img[cutoff_index, 0]
        return max_l, average_a * max_l / average_l, average_b * max_l / average_l

    def transform(im, est):
        """ Transforms Lab color image to corrected image in BGR based on illuminant estimation """
        l, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2Lab))

        l_est, a_est, b_est = est
        max_l = np.max(l)
        scale_l = max_l / l_est
        ratio = l / l_est

        a_corrected = (a - (ratio * a_est)).clip(min=0, max=255)
        b_corrected = (b - (ratio * b_est)).clip(min=0, max=255)
        l_corrected = (l * scale_l).clip(min=0, max=255)
        lab_corrected = cv2.merge((np.uint8(l_corrected), np.uint8(a_corrected), np.uint8(b_corrected)))
        return cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)

    return transform(im, whitePatchLab(im)), transform(im, greyWorldLab(im))


def labCorrection(im):
    """ Peforms color correction in lab color space
    http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XL-5-W5/25/2015/isprsarchives-XL-5-W5-25-2015.pdf
    """
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    l, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2Lab))
    # Hue correction by shifting
    a = a - np.median(a)
    b = b - np.median(b)
    a = (a - np.min(a)) / (np.max(a) - np.min(a)) * 255.0
    b = (b - np.min(b)) / (np.max(b) - np.min(b)) * 255.0
    # Luminanc constrast enhancement
    l = clahe.apply(l)
    final = cv2.merge((np.uint8(l), np.uint8(a), np.uint8(b)))
    return cv2.cvtColor(final, cv2.COLOR_Lab2BGR)


def doubleOpponency(im):
    """ Implementation of A Color Constancy Model with Double Opponency Mechanicsm
        http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Gao_A_Color_Constancy_2013_ICCV_paper.pdf
    """
    def getSO(O, sigma, ksize=(5, 5)):
        """ Converts Single Opponent cell to Double opponent cell """
        return cv2.GaussianBlur(O, ksize, sigma)

    def getConeLayer(im):
        """ Derives R, G, B, Y, L channels from color image
            :return: [r, g, b, y, l]
        """
        b = im[..., 0]
        g = im[..., 1]
        r = im[..., 2]
        y = (r + g) / 2
        l = r + g + b
        return [r, g, b, y, l]

    def getLGNLayer(im):
        r, g, b, y, l = getConeLayer(im)
        O_rg = (r - g) / (math.sqrt(2))
        O_yb = (y - 2*b) / (math.sqrt(6))
        O_lplus = l / (math.sqrt(3))
        lgn = [O_rg, O_yb, O_lplus]
        return lgn

    def getV1Layer(im, k=0.8, scale=3, sigma=2.5):
        """ Generates DO cells that are chromatically and spatially opponent
            :param k: relative cone weight that controls contribution of RF surround
            :param scale: size of receptive field. default 3
            :return do: list of Double opponent channels
        """
        O_rg, O_yb, O_lplus = getLGNLayer(im)
        DO_rg = getSO(O_rg, sigma) + k * getSO(-O_rg, scale*sigma)
        DO_by = getSO(-O_yb, sigma) + k * getSO(O_yb, scale*sigma)
        DO_l = getSO(O_lplus, sigma) + k * getSO(-O_lplus, scale*sigma)
        do = [DO_rg, DO_by, DO_l]
        return do

    def rgbFromDO(im):
        """ Convert from double opponent space to RGB space
            :return: true illuminant, corrected image
        """
        do = getV1Layer(im)
        # Transform function
        a = 1 / math.sqrt(2)
        b = 1 / math.sqrt(6)
        c = 1 / math.sqrt(3)
        tf = np.linalg.inv(np.array([[a, -a, 0],
                                     [b,  b, -2*b],
                                     [c,  c, c]]))
        R = (tf[0][0] * do[0]) + (tf[0][1] * do[1]) + (tf[0][2] * do[2])
        maxR = np.max(R)
        G = (tf[1][0] * do[0]) + (tf[1][1] * do[1]) + (tf[1][2] * do[2])
        maxG = np.max(G)
        B = (tf[2][0] * do[0]) + (tf[2][1] * do[1]) + (tf[2][2] * do[2])
        maxB = np.max(B)
        # Illumination
        coef = maxR + maxG + maxB
        e = (maxB / coef, maxG / coef, maxR / coef)
        B = np.uint8(B * 255)
        G = np.uint8(G * 255)
        R = np.uint8(R * 255)
        return e, cv2.merge((B, G, R))

    im = img.norm(im)
    e, dt = rgbFromDO(im)
    return e, dt
