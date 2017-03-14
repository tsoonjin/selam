#!/usr/bin/python2.7
import cv2
import numpy as np
import preprocess

from selam.utils import img


def novel_color_correct(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    a = clahe.apply(a)
    b = clahe.apply(b)
    return cv2.cvtColor(cv2.merge((l, np.uint8(a), np.uint8(b))), cv2.COLOR_LAB2BGR)


def frenchPreprocess(img):
    """ Automatic Underwater Image Pre-processing
    https://www.ensta-bretagne.fr/jaulin/paper_bazeille_CMM06.pdf
    """
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
    homo = preprocess.deilluminate_single(y)
    ansio = cv2.GaussianBlur(homo, (5, 5), 1)
    bgr = cv2.cvtColor(cv2.merge((ansio, cr, cb)), cv2.COLOR_YCR_CB2BGR)
    b, g, r = cv2.split(bgr)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    out = cv2.merge((b, g, r))
    return out


def iace(img):
    """ Performs Image Adaptive Contrast Enhancement """

    def util_iace(channel):
        min__val, max__val, min_loc, max_loc = cv2.minMaxLoc(channel)
        min_val, max_val = hist_info(channel)
        channel_ = (channel - min__val) / (max__val - min__val) * 255.0
        return channel_

    b, g, r = cv2.split(img)
    b_ = util_iace(b)
    g_ = util_iace(g)
    r_ = util_iace(r)
    out = cv2.merge((np.uint8(b_), np.uint8(g_), np.uint8(r_)))  # scale up to 255 range
    return out


def enhanceTan(img):
    """ Tan's method to enhance image """
    gamma = preprocess.gamma_correct(img)
    b, g, r = cv2.split(gamma)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    out = cv2.merge((b, g, r))
    return out


def meanFilter(chan):
    y, x = chan.shape[:2]
    chan = cv2.resize(chan, (x / 2, y / 2))
    return np.uint8(img.blockiter(chan, np.mean, blksize=(10, 10)))


def hybrid_clahe(img):
    img = cv2.medianBlur(img, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    h, l, s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    s = clahe.apply(s)
    l = clahe.apply(l)
    hls2bgr = cv2.cvtColor(cv2.merge((h, l, s)), cv2.COLOR_HLS2BGR)
    b_, g_, r_ = cv2.split(hls2bgr)
    b, g, r = cv2.split(img)
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)
    rgb = cv2.merge((b, g, r))
    out = cv2.addWeighted(hls2bgr, 0.4, rgb, 0.4, 0)
    return out


def dark_channel(img):
    """Dark Channel Prior"""
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    darkMap = np.zeros(img.shape[:2], dtype=np.uint8)
    tMap = np.zeros(img.shape[:2], dtype=np.float32)
    h, w, _ = img.shape
    w /= 40
    h /= 40
    b, g, r = cv2.split(img)
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
    x = 0
    y = 0
    for i in xrange(40):
        for j in xrange(40):
            bmin, _, _, _ = cv2.minMaxLoc(b[y:y + h - 1, x:x + w - 1])
            gmin, _, _, _ = cv2.minMaxLoc(g[y:y + h - 1, x:x + w - 1])
            rmin, _, _, _ = cv2.minMaxLoc(r[y:y + h - 1, x:x + w - 1])
            dark = min(gmin, rmin)
            darkMap[y:y + h - 1, x:x + w - 1] = dark
            x += w
        x = 0
        y += h
    _, ambient_max, _, ambient_loc = cv2.minMaxLoc(darkMap)
    x = 0
    y = 0
    bmax = b[ambient_loc[1], ambient_loc[0]]
    gmax = g[ambient_loc[1], ambient_loc[0]]
    rmax = r[ambient_loc[1], ambient_loc[0]]
    for i in xrange(40):
        for j in xrange(40):
            bmin, _, _, _ = cv2.minMaxLoc(b[y:y + h - 1, x:x + w - 1])
            gmin, _, _, _ = cv2.minMaxLoc(g[y:y + h - 1, x:x + w - 1])
            rmin, _, _, _ = cv2.minMaxLoc(r[y:y + h - 1, x:x + w - 1])
            t = min(gmin / gmax, rmin / rmax)
            tMap[y:y + h - 1, x:x + w - 1] = max((1 - t), 0.1)
            x += w
        x = 0
        y += h
    tMap = cv2.dilate(tMap, kern, 3)
    tMap = cv2.erode(tMap, kern, 3)
    b = np.float32(b)
    g = np.float32(g)
    r = np.float32(r)
    b = (b - bmax) / tMap + bmax
    b = cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255 * 255
    g = (g - gmax) / tMap + gmax
    g = cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255 * 255
    # r = (r - rmax)/tMap+rmax
    return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
    # return cv2.cvtColor(np.uint8(tMap*255),cv2.COLOR_GRAY2BGR)


def redchannelprior(img):
    img = cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2))
    b, g, r = cv2.split(img)
    waterEst = cv2.GaussianBlur(r, (5, 5), 0)
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(waterEst)
    A = img[maxloc[1], maxloc[0]]
    A = [i / 255.0 for i in A]
    b = np.float32(b) / 255.0
    g = np.float32(g) / 255.0
    r = np.float32(r) / 255.0
    t_bound = np.full(img.shape[:2], 1)
    r_min = img.blockiter(1 - r, np.min) / float(1 - A[2])
    g_min = img.blockiter(g, np.min) / float(A[1])
    b_min = img.blockiter(b, np.min) / float(A[0])
    tMap = t_bound - np.min([r_min, b_min, g_min], axis=0)
    tMap = cv2.GaussianBlur(tMap, (11, 11), 0)
    # return VUtil.toBGR(np.uint8(tMap*255), 'gray')
    return redchannel_util(img, A, tMap)


def redchannel_util(img, A, t):
    bgr = cv2.split(img)
    bgr = [np.float32(i / 255.0) for i in bgr]
    t_bound = np.full(img.shape[:2], 0.1)
    additive = [(1 - i) * i for i in A]
    J = [(i - A[x]) / np.maximum(t, t_bound) + additive[x] for x, i in enumerate(bgr)]
    J = [np.uint8(z_norm(j) * 255) for j in J]
    print(np.max(J[0]))
    print(np.max(J[1]))
    print(np.max(J[2]))
    return cv2.merge(tuple(J))


def naive_fusionmap(img):
    """Merge weight maps without multiple scale fusion"""
    img = cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2))
    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    maps = [getExposedness(img), cv2.cvtColor(get_salient(gray), cv2.COLOR_GRAY2BGR),
            getLuminance(img), getChromatic(img)]
    maps = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in maps]
    mean = np.mean(maps, axis=0) / 255.0 * 8
    mean.clip(max=1.0, out=mean)
    b = np.uint8(mean * b)
    g = np.uint8(mean * g)
    r = np.uint8(mean * r)
    return cv2.merge((b, g, r))


def showWMaps(img):
    """Debug all 6 weight maps before fusion"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplace = laplacian(img)
    local_contrast = getLocalContrast(img)
    exposedness = getExposedness(img)
    chromatic = getChromatic(img)
    salient = cv2.cvtColor(get_salient(gray), cv2.COLOR_GRAY2BGR)
    luminance = getLuminance(img)
    h1 = np.hstack((laplace, local_contrast, exposedness))
    h2 = np.hstack((chromatic, salient, luminance))
    return np.vstack((h1, h2))


def getLocalContrast(img):  # can find also std between channel and saturation
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    blur = cv2.GaussianBlur(v, (5, 5), 0)
    final = np.std([v, blur], axis=0)
    final = cv2.normalize(final, 0, 255, cv2.NORM_MINMAX) * 255
    return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)


def getExposedness(img):  # can find also std between channel and saturation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = 2 * (0.25**2)
    final = np.power(img - 0.5, 2) / sigma
    final = cv2.normalize(final, 0, 255, cv2.NORM_MINMAX)
    final = np.exp(-1 * final) * 255
    return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)


def getChromatic(img):  # can find also std between channel and saturation
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    final = np.std([b, g, r, s], axis=0)
    return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)


def get_salient(chan):
    empty = np.ones_like(chan)
    mean = np.mean(chan)
    mean = empty * mean
    blur = cv2.GaussianBlur(chan, (21, 21), 1)
    final = mean - blur
    final = final.clip(min=0)
    final = np.uint8(final)
    return final


def get_salient_aggregate(img):
    """ Returns saliency map by combining each channels in the img """
    a, b, c = cv2.split(img)
    a = cv2.cvtColor(get_salient(a), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(get_salient(b), cv2.COLOR_GRAY2BGR)
    c = cv2.cvtColor(get_salient(c), cv2.COLOR_GRAY2BGR)
    return np.uint8(np.mean([a, b, c], axis=0))


def get_salient_color(img):
    a, b, c = cv2.split(img)
    a = cv2.cvtColor(get_salient(a), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(get_salient(b), cv2.COLOR_GRAY2BGR)
    c = cv2.cvtColor(get_salient(c), cv2.COLOR_GRAY2BGR)
    return np.hstack((a, b, c))


def getLuminance(img):
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    final = np.std([b, g, r, v], axis=0)
    return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)


def laplacian(img):
    return cv2.cvtColor(np.uint8(
        cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)), cv2.COLOR_GRAY2BGR)
