#!/usr/bin/python2.7
import cv2
import numpy as np
import enhancement as en
import preprocess
import threshold as th
import core


def thresh_orange2(img):
    imgg = en.finlaynorm(img)
    b, g, r = cv2.split(imgg)
    hsv = core.to_hsv(imgg)
    loThresh1 = (0, 20, 0)
    hiThresh1 = (13, 200, 235)
    loThresh2 = (170, 20, 0)
    hiThresh2 = (180, 200, 235)
    mask = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh1, hiThresh1)
    mask2 = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh2, hiThresh2)
    orange = mask | mask2
    min = np.min(g)
    max = np.max(g)
    diff = max - min
    thresh = min + diff / 2
    common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    final = common & orange
    final = cv2.erode(final, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    final = cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return final


def threshChan(chan, lo=None, hi=None):
    if not lo and not hi:
        lo = np.mean(chan) - 5
        hi = np.mean(chan) + 5
    return cv2.inRange(chan, lo, hi)


def contourRect(thresh):
    outImg = core.to_bgr(thresh, 'gray')
    contours, hierr = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        hierr = hierr[0]
        for cnt in contours:
            cnt = cv2.convexHull(cnt)
            if cv2.contourArea(cnt) > 500:
                rect = cv2.minAreaRect(cnt)
                cv2.drawContours(outImg, [cnt], -1, (0, 0, 255), 2)
                cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255, 0, 255), 2)
                break
    return outImg


def cannyThresh(chan, offset=8):
    chan = cv2.GaussianBlur(chan, (5, 5), 0)
    min, max = np.min(chan), np.max(chan)
    min += (max - min) / offset
    out = np.uint8(cv2.Canny(chan, min / 2, min, apertureSize=5))
    return out


def filterBlack(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    return cv2.inRange(v, 0, 110)


def filterWhite(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    mask = cv2.inRange(v, 150, 255)
    return cv2.bitwise_not(mask)


def getPatternOrange(img):
    img = en.finlaynorm(img)
    b, g, r = cv2.split(img)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 201, 13)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return cv2.add(r, b)


def getPatternValue(img, mode=1):
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    v = preprocess.norm_illum_color(v, 2.5)
    v = th.adaptive_thresh(v, 15, 10)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    v = cv2.erode(v, kern2, iterations=2)
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, kern2, iterations=2)
    return v


def adaptive_thresh(gray, block=31, offset=5):
    res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                block, offset)
    return res


def getPattern1(img, mode=1):
    img = en.finlaynorm(img)
    b, g, r = cv2.split(img)
    b = th.adaptive_thresh(b, 41, 5)
    b = cv2.bitwise_not(b)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    b = cv2.erode(b, kern, iterations=1)
    b = cv2.dilate(b, kern2, iterations=3)
    return b


def getPatternHue(img, mode=1):
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    h = th.adaptive_thresh(h, 61, 3)
    h = cv2.bitwise_not(h)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    #h = cv2.dilate(h, kern2,iterations=1)
    return h


def getPatternWhite(img):
    img = en.finlaynorm(img)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    a = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 3)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return a


def thresh_orange(img, hsv=0):
    imgg = en.finlaynorm(en.iace(img))
    #imgg = VUtil.iace(img)
    b, g, r = cv2.split(imgg)
    # HSV threshold
    h, s, v = cv2.split(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV))
    loThresh1 = (0, 0, 150)
    hiThresh1 = (20, 255, 255)
    loThresh2 = (170, 0, 150)
    hiThresh2 = (180, 255, 255)
    '''
    #Nice pool
    loThresh1 = (0,100,0)
    hiThresh1 = (12,230,150)
    loThresh2 = (172,100,0)
    hiThresh2 = (180,230,150)
    '''
    mask = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh1, hiThresh1)
    mask2 = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh2, hiThresh2)
    orange = mask | mask2
    erodeEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    min_pix = np.amin(b)
    max_pix = np.amax(b)
    '''
    thresh = min_pix + (max_pix-min_pix)/3.5
    thresh = thresh if thresh <= 90 else 90
    '''
    thresh = 90
    common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]

    if hsv == 0:
        return common & orange
    if hsv == 1:
        return common
    if hsv == 2:
        return orange


def thresh_yellow(img, hsv=0, thresh_val=90):
    imgg = en.finlaynorm(en.iace(img))
    # imgg = VUtil.iace(img)
    b, g, r = cv2.split(imgg)
    b, g, r = cv2.split(imgg)
    '''HSV Thresh'''
    h, s, v = cv2.split(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV))
    s_status = (np.amin(s), np.mean(s), np.amax(s))
    v_status = (np.amin(v), np.mean(v), np.amax(v))

    loThresh = (15, 0, 150)
    hiThresh = (50, 255, 255)

    yellow = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh, hiThresh)
    erodeEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    min_pix = np.amin(b)
    max_pix = np.amax(b)
    thresh = thresh_val
    # thresh = thresh if thresh <= 90 else 90
    common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]

    if hsv == 0:
        return common & yellow
    if hsv == 1:
        return common
    if hsv == 2:
        return yellow


def thresh_yellow2(img, hsv=0):
    imgg = en.finlaynorm(en.iace(img))
    # imgg = VUtil.iace(img)
    b, g, r = cv2.split(imgg)
    b, g, r = cv2.split(imgg)
    '''HSV Thresh'''
    h, s, v = cv2.split(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV))
    s_status = (np.amin(s), np.mean(s), np.amax(s))
    v_status = (np.amin(v), np.mean(v), np.amax(v))

    loThresh = (20, 0, 150)
    hiThresh = (50, 255, 255)

    yellow = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh, hiThresh)
    erodeEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    min_pix = np.amin(b)
    max_pix = np.amax(b)
    thresh = 90
    # thresh = thresh if thresh <= 90 else 90
    common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]

    if hsv == 0:
        return common & yellow
    if hsv == 1:
        return common
    if hsv == 2:
        return yellow


def thresh_yellow_orange(img):
    imgg = en.finlaynorm(img)
    b, g, r = cv2.split(imgg)
    loThresh = (0, 0, 0)
    hiThresh = (50, 230, 230)
    loThresh2 = (172, 0, 0)
    hiThresh2 = (180, 230, 230)
    mask = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh, hiThresh)
    mask2 = cv2.inRange(cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV), loThresh2, hiThresh2)
    hsv = mask | mask2
    erodeEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    min_pix = np.amin(b)
    max_pix = np.amax(b)
    thresh = min_pix + (max_pix - min_pix) / 3.5
    thresh = thresh if thresh < 95 else 255
    common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    return common & hsv
