#!/usr/bin/env python
import numpy as np
import cv2
import core


def DoG(img, kern1=(3, 3), kern2=(5, 5)):
    """Difference of Gaussian using diff kernel size"""
    smooth1 = cv2.GaussianBlur(img, kern1, 0)
    smooth2 = cv2.GaussianBlur(img, kern2, 0)
    final = smooth1 - smooth2
    return final


def norm_illum_color(img, gamma=2.2):
    """ Normalizes illumination for colored image """
    img = np.float32(img)
    img /= 255.0
    img = cv2.pow(img, 1 / gamma) * 255
    img = np.uint8(img)
    return img


def gamma_correct(img):
    gamma = 2.2
    inverse_gamma = 1.0 / gamma
    b, g, r = cv2.split(img)
    b = np.uint8(cv2.pow(b / 255.0, inverse_gamma) * 255)
    g = np.uint8(cv2.pow(g / 255.0, inverse_gamma) * 255)
    r = np.uint8(cv2.pow(r / 255.0, inverse_gamma) * 255)
    return cv2.merge((b, g, r))


def sharpen(img):
    blur = cv2.GaussianBlur(img, (5, 5), 5)
    res = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return res


def deilluminate(img):
    h, s, gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    blur = cv2.GaussianBlur(gray, (63, 63), 41)
    gray = cv2.log(np.float32(gray))
    blur = cv2.log(np.float32(blur))
    res = np.exp(gray - blur)
    res = cv2.normalize(res, 0, 255, cv2.NORM_MINMAX) * 255
    v = np.uint8(res)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def homomorphic(img):
    h, s, gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    gray = cv2.log(np.float32(gray))
    blur = cv2.GaussianBlur(gray, (63, 63), 41)
    res = np.exp(gray - blur)
    res = cv2.normalize(res, 0, 255, cv2.NORM_MINMAX) * 255
    v = np.uint8(res)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def deilluminate_single(gray):
    blur = cv2.GaussianBlur(gray, (63, 63), 41)
    gray = cv2.log(np.float32(gray))
    blur = cv2.log(np.float32(blur))
    res = np.exp(gray - blur)
    res = cv2.normalize(res, 0, 255, cv2.NORM_MINMAX) * 255
    gray = np.uint8(res)
    return gray


def motion_deflicker(frames, img):
    log_median = cv2.log(np.float32(np.median(frames, axis=0)))
    log_img = cv2.log(np.float32(img))
    diff = cv2.GaussianBlur(log_img - log_median, (21, 21), 0)
    res = img / np.exp(diff)
    res = res.clip(max=255)
    blur = cv2.GaussianBlur(np.uint8(res), (5, 5), 0)
    res = cv2.addWeighted(np.uint8(res), 1.5, blur, -0.5, 0)
    return res


def deilluminate2(img):
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    log_v = cv2.log(np.float32(v))
    blur_v = cv2.log(np.float32(cv2.GaussianBlur(v, (63, 63), 41)))
    res = np.exp(log_v - blur_v)
    return cv2.cvtColor(np.uint8(res * 255), cv2.COLOR_GRAY2BGR)


def gamma1(gray):
    gray = np.float32(gray)
    gray /= 255.0
    gray = 0.3 * ((cv2.log(2 * gray + 0.1)) + abs(np.log(0.1)))
    return np.uint8(gray * 255)


def gamma2(gray):
    gray = np.float32(gray)
    gray /= 255.0
    gray = 0.8 * (cv2.pow(gray, 2))
    return np.uint8(gray * 255)


def gamma3(gray):
    gray = np.float32(gray)
    gray /= 255.0
    total = 1 / (np.exp(8 * (gray - 0.5)) + 1) * 255
    return np.uint8(total)


def gamma1color(img):
    b, g, r = cv2.split(img)
    b = gamma1(b)
    g = gamma1(g)
    r = gamma1(r)
    return cv2.merge((b, g, r))


def gamma2color(img):
    b, g, r = cv2.split(img)
    b = gamma2(b)
    g = gamma2(g)
    r = gamma2(r)
    return cv2.merge((b, g, r))


def gamma3color(img):
    b, g, r = cv2.split(img)
    b = gamma3(b)
    g = gamma3(g)
    r = gamma3(r)
    return cv2.merge((b, g, r))


def single_deflicker(grayimgs):
    logimgs = [cv2.log(np.float32(x)) for x in grayimgs]
    median = np.median(logimgs, axis=0)
    diff = np.abs(logimgs[-1] - median)
    blur = cv2.GaussianBlur(diff, (3, 3), 1, 1)
    illumination_est = np.exp(blur)
    output = grayimgs[-1] / (illumination_est)
    return output


def motion_deflicker2(imgs):
    """A motion compensated approach to remove sunlight flicker
    Choice of low-pass filter could be changed or used a different standard deviation
    """
    b = [x[:, :, 0] for x in imgs]
    g = [x[:, :, 1] for x in imgs]
    r = [x[:, :, 2] for x in imgs]
    b_corrected = single_deflicker(b)
    g_corrected = single_deflicker(g)
    r_corrected = single_deflicker(r)
    return cv2.merge((np.uint8(b_corrected), np.uint8(g_corrected), np.uint8(r_corrected)))


def single_homomorphic_filter(grayimg):
    log = np.nan_to_num(np.log(np.float32(grayimg)))
    blur = np.nan_to_num(cv2.GaussianBlur(log, (21, 21), 21))
    output = np.exp(cv2.addWeighted(log, 0.5, blur, 0.5, 0))
    return output


def homomorphic_filter(img):
    """Homomorphic filtering"""
    b, g, r = cv2.split(img)
    b_corrected = single_homomorphic_filter(b)
    g_corrected = single_homomorphic_filter(g)
    r_corrected = single_homomorphic_filter(r)
    return cv2.merge((np.uint8(b_corrected), np.uint8(g_corrected), np.uint8(r_corrected)))
