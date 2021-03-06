#!/usr/bin/env python
import cv2
import os
import numpy as np


# Input / Output

def normUnity(img):
    """ Normalizes image to unity """
    min = np.min(img)
    max = np.max(img)
    return (img - min) / (max - min)


def norm(img, max=255.0):
    """ Normalizes image to 0 - 1 assuming max to be 255 """
    return img / max


def get_jpgs(dirpath, skip=0, resize=None):
    """ Returns all images located in given dirpath
        skip : number of frames skip to reduce computation time
        resize: scale factor for resize

    """
    filenames = os.listdir(dirpath)
    # Only attempt to parse and sort files that end with .jpg
    filenames = [filename for filename in filenames
                 if filename.endswith(".jpg") or filename.endswith(".png")]
    filenames.sort(key=lambda x: int(x.split('.', 1)[0]))
    frames = [cv2.imread('{}/{}'.format(dirpath, filename))
              for filename in filenames]
    out = frames[0::skip] if skip > 0 else frames
    if resize:
        new_size = (out[0].shape[1] / resize, out[0].shape[0] / resize)
        return map(lambda x: cv2.resize(x, new_size), out)
    return out


def write_jpgs(dirpath, jpgs):
    if os.path.exists(os.path.abspath(dirpath)):
        for i in range(len(jpgs)):
            filename = dirpath + str(i) + ".jpg"
            cv2.imwrite(filename, jpgs[i])
        print('Wrote {} images to {}'.format(len(jpgs), dirpath))


# Image Formatting

def quantizeColor(img, K=8):
    """ Return quantized image and centers """
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2, label, center


def cv2others(img):
    """ Returns RGB channel image given BGR OpenCV image """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                thickness=2)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))


def workon_frames(frames, func, wait=100):
    """ Wait for n milliseconds before moving on to next frame """
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.namedWindow(func.__name__, cv2.WINDOW_NORMAL)
    for frame in frames:
        cv2.imshow('original', frame)
        cv2.imshow(func.__name__, func(frame))
        k = cv2.waitKey(wait)
        if k == 27:
            break
    cv2.destroyAllWindows()


def display_channels(frames, wait=100):
    " Displayes BGR, HSV and LAB channels information given frames """
    r, c = frames[0].shape[:2]
    for f in frames:
        out = cv2.resize(fuse_channels(f), (c * 2, r * 2))
        cv2.imshow('channels', out)
        k = cv2.waitKey(wait)
        if k == 27:
            break
    cv2.destroyAllWindows()


def get_channel(img, channel):
    """ Get specific color channel given an image """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = {'bgr_b': img[..., 0], 'bgr_green': img[..., 1], 'bgr_r': img[..., 2],
                'hsv_h': hsv[..., 0], 'hsv_s': hsv[..., 1], 'hsv_v': hsv[..., 2],
                'lab_l': lab[..., 0], 'lab_a': lab[..., 1], 'lab_b': lab[..., 2],
                'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)}
    return channels[channel]


def fuse_channels(img):
    """ Returns bgr, hsv and lab channels of image in order """
    return np.vstack((get_bgr_stack(img), get_hsv_stack(img),
                      get_lab_stack(img), get_salient_stack(img)))


def get_bgr_stack(img):
    """ Returns horizontal stack of BGR channels """
    b, g, r = cv2.split(img)
    b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    return np.hstack((b, g, r))


def get_hsv_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))


def get_luv_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2LUV))


def get_ycb_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))


def get_lab_stack(img):
    return get_bgr_stack(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))


def get_salient_stack(img):
    """ Return saliency map for each channels in given image colorspace """
    a, b, c = cv2.split(img)
    a = cv2.cvtColor(get_salient(a), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(get_salient(b), cv2.COLOR_GRAY2BGR)
    c = cv2.cvtColor(get_salient(c), cv2.COLOR_GRAY2BGR)
    return np.hstack((a, b, c))


def get_salient(chan):
    empty = np.ones_like(chan)
    mean = np.mean(chan)
    mean = empty * mean
    blur = cv2.GaussianBlur(chan, (21, 21), 1)
    final = mean - blur
    final = final.clip(min=0)
    final = np.uint8(final)
    return final


def get_roi(img, top_left, bot_right):
    """ Returns region of interest of an img given bounding box points """
    y = [max(top_left[1], 0), min(bot_right[1], img.shape[0] - 1)]
    x = [max(top_left[0], 0), min(bot_right[0], img.shape[1] - 1)]
    return img[y[0]:y[1], x[0]:x[1]]


def hist_info(chan):  # For iace
    done_low = True
    hist, bins = np.histogram(chan, 256, [0, 256])
    cdf = hist.cumsum()
    low = int(chan.size * 0.04)
    hi = int(chan.size * 0.96)
    for h, i in enumerate(cdf):
        if i > low and done_low:
            low_thresh = h
            done_low = False
        if i > hi:
            high_thresh = h
            break
    return (low_thresh, high_thresh)


def blockiter(img, func, blksize=(30, 30)):
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    y, x = img.shape[:2]
    for i in xrange(0, y, 5):
        dy = blksize[1] - 1 if i + blksize[1] < y else y - i - 1
        for j in xrange(0, x, 5):
            dx = blksize[0] - 1 if j + blksize[0] < x else x - j - 1
            view = img[i:i + dy, j:j + dx]
            mask[i:i + dy, j:j + dx] = func(view)
    return mask


def sort_by_chan(image, chan):
    """ Sorts by the specified channel
        Returns the sorted image.
    """
    # Reshape to 2D array
    flat = np.reshape(image, (-1, 1, image.shape[2])).squeeze()
    # Get specified Channel
    channel = image[:, :, chan].flatten()
    # Calculate sort indicies then sort by them
    sort_indices = np.argsort(channel)
    return flat[sort_indices]
    # return flat[sort_indices].reshape(image.shape)


def clearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows, imgCols = imgBW.shape

    contourList = []  # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

    return imgBWcopy


def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)

    return imgBWcopy
