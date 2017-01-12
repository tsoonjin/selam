#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
import math
import random
import multiprocessing as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from skimage import img_as_float

'''COLOR CODES'''
RED = (0, 0, 255)
BLUE = (255, 128, 0)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 69, 255)
PURPLE = (204, 0, 204)
WHITE = (255, 255, 255)

"""Info Class

Holds information about object of interest
"""


class Info:

    def __init__(self):
        self.data = {'area': 0.0, 'dxy': [0.2, 0.2], 'angle': 666,
                     'centroid': (0, 0), 'heading': 0, 'detected': False,
                     'object': 0}

    def draw(self, img):
        center_x, center_y = int(img.shape[1] / 2), int(img.shape[0] / 2)
        cv2.putText(img, "detected: " + str(self.data['detected']), (30, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, RED)
        cv2.putText(img, "a: %.2f" % self.data['area'], (200, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, BLUE)
        cv2.putText(
            img,
            "dx: %.2f" %
            self.data['dxy'][0] +
            " dy: %.2f" %
            self.data['dxy'][1],
            (300,
             20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            GREEN)
        cv2.putText(img, "t: %d" % int(self.data['angle']), (530, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, ORANGE)
        cv2.putText(img, "h: %d" % int(self.data['heading']), (620, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, PURPLE)
        VUtil.draw_rect(img, (center_x - 5, center_y + 5), (center_x + 5, center_y - 5), WHITE, 2)
        VUtil.draw_circle(img, self.data['centroid'], 4, YELLOW, -1)

"""Vision Queue Class

Holds n frames of images and corresponding info
"""


class VQueue:

    def __init__(self):
        self.pipeline = []
        self.currFrame = None
        self.frame_limit = 10
        self.corrected = []

    def add(self, img):
        self.pipeline.append(img)
        if len(self.pipeline) > self.frame_limit:
            self.pipeline = []

"""Vision VUtils Class

Core vision algorithms, utility functions and experimental setup
"""


class VUtil:

    bridge = CvBridge()

    """Image I/O"""

    @staticmethod
    def readCompressed(rosimg):
        np_arr = np.fromstring(rosimg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)

    @staticmethod
    def writeCompressed(img):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpeg', img)[1]).tostring()
        return msg

    @staticmethod
    def sk2cv(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2sk(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rosimg2cv(ros_img):
        try:
            frame = VUtil.bridge.imgmsg_to_cv2(ros_img, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        return frame

    @staticmethod
    def cv2rosimg(cv_img):
        try:
            return VUtil.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    """Image Conversion"""

    @staticmethod
    def toBGR(img, flag):
        if flag is 'gray':
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if flag is 'hsv':
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if flag is 'lab':
            return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    @staticmethod
    def toHSV(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    @staticmethod
    def toLAB(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    """Basic OpenCV functions"""

    @staticmethod
    def rectify(img, rect):
        h = np.int0(np.around(cv2.cv.BoxPoints(rect)))
        h = h.reshape((4, 2))
        hnew = np.zeros((4, 2), dtype=np.float32)
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
        diff = np.diff(h, axis=1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
        return img[hnew[0][1]:hnew[2][1], hnew[0][0]:hnew[2][0]]

    @staticmethod
    def z_norm(arr):
        minX = np.min(arr)
        maxX = np.max(arr)
        return arr - minX / (maxX - minX)

    @staticmethod
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

    @staticmethod
    def calc_hist(src):
        h = np.zeros((src.shape[0], 256, 3))
        bins = np.arange(256).reshape(256, 1)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for ch, col in enumerate(color):
            hist_item = cv2.calcHist([src], [ch], None, [256], [0, 255])
            cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
            hist = np.int32(np.around(hist_item))
            pts = np.column_stack((bins, hist))
            cv2.polylines(h, [pts], False, col)
        h = np.flipud(h)
        return np.uint8(h)

    @staticmethod
    def resize(img, x, y, flagg=cv2.cv.CV_INTER_NN):
        return cv2.resize(img, fx=x, fy=y, flag=flagg)

    @staticmethod
    def blend(img1, a, img2, b, g):
        return cv2.addWeighted(img1, a, img2, b, g)

    @staticmethod
    def genGPyramid(img, level=6):
        """Generate Gaussian Pyramid of 6 levels"""
        G = img.copy()
        gp = [img]
        for i in xrange(level):
            G = cv2.pyrDown(G)
            gp.append(G)
        return gp

    @staticmethod
    def genLPyramid(gp):
        """Generate Laplacian Pyramid"""
        size = len(gp) - 2
        lp = [gp[size]]
        for i in xrange(size, 0, -1):
            blur = gp[i - 1]
            GE = cv2.pyrUp(gp[i])
            GE = cv2.resize(GE, (blur.shape[1], blur.shape[0]))
            L = cv2.subtract(blur, GE)
            lp.append(L)
        return lp

    @staticmethod
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

    @staticmethod
    def getHist(chan, color=(0, 0, 255)):  # For iace
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        mask = np.zeros((300, 256, 3))
        for x, y in enumerate(np.int32(np.around(hist))):
            cv2.line(mask, (x, 299), (x, y), color)
        return np.uint8(mask)

    @staticmethod
    def integral(img):
        res = cv2.integral(img)
        return img

    @staticmethod
    def confidenceMask(img, offset=0.5):
        cent_x = img.shape[1] / 2.0
        cent_y = img.shape[0] / 2.0
        mask = np.zeros_like(img)
        cv2.circle(mask, (int(cent_x), int(cent_y)), int(offset * cent_y * 2), (255, 255, 255), -1)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def euclid_dist(ls):
        sum = 0
        for x, y in ls:
            sum += np.power(x - y, 2)
        return np.sqrt(sum)

    """Drawing functions"""

    @staticmethod
    def draw_rect(canvas, top_left, bot_right, color, thickness):
        cv2.rectangle(canvas, top_left, bot_right, color, thickness)

    @staticmethod
    def draw_circle(src, center, rad, color, thickness):
        cv2.circle(src, center, rad, color, thickness)

    @staticmethod
    def text(canvas, text, org, fontScale, color):
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color)

    """Contour related"""

    @staticmethod
    def getAspRatio(cnt):
        rect = cv2.minAreaRect(cnt)
        w, l = rect[1]
        if l > w:
            ratio = l / w
        else:
            ratio = w / l
        print(ratio)
        return ratio

    @staticmethod
    def box2D(rect):
        box = cv2.cv.BoxPoints(rect)
        return np.int0(box)

    @staticmethod
    def groupContours(chosen_cnt, outImg, info):
        hull = cv2.convexHull(np.vstack(chosen_cnt))
        info['area'] = VUtil.getRectArea(hull) / float(outImg.shape[0] * outImg.shape[1])
        VUtil.getDOA(hull, outImg, info)

    @staticmethod
    def groupContoursPickup(chosen_cnt, outImg, info):
        hull = cv2.convexHull(np.vstack(chosen_cnt))
        info['area'] = VUtil.getRectArea(hull)
        print(info['area'])
        VUtil.getDOA(hull, outImg, info)

    @staticmethod
    def groupContoursAlign(chosen_cnt, outImg, info, blank):
        hull = cv2.convexHull(np.vstack(chosen_cnt))
        info['area'] = VUtil.getRectArea(hull) / float(outImg.shape[0] * outImg.shape[1])
        VUtil.getRailDOA(hull, outImg, info, blank)

    @staticmethod
    def getDOA(cnt, outImg, info):
        rect = cv2.minAreaRect(cnt)
        points = np.int32(cv2.cv.BoxPoints(rect))
        edge1 = points[1] - points[0]
        edge2 = points[2] - points[1]
        if cv2.norm(edge1) > cv2.norm(edge2):
            rectAngle = math.degrees(math.atan2(edge1[1], edge1[0]))
        else:
            rectAngle = math.degrees(math.atan2(edge2[1], edge2[0]))
        startpt = info['centroid']
        gradient = np.deg2rad(rectAngle)
        endpt = (int(startpt[0] + 200 * math.cos(gradient)),
                 int(startpt[1] + 200 * math.sin(gradient)))
        startpt = (int(startpt[0]), int(startpt[1]))
        cv2.line(outImg, startpt, endpt, (0, 255, 0), 2)
        cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255, 0, 0), 2)
        info['angle'] = 90 - abs(rectAngle) if rectAngle >= -90 else 90 - abs(rectAngle)
        info['angle'] = (info['angle'] + 90) % 360

    @staticmethod
    def getRailDOA(cnt, outImg, info, blank):
        rect = cv2.minAreaRect(cnt)
        points = np.int32(cv2.cv.BoxPoints(rect))
        edge1 = points[1] - points[0]
        edge2 = points[2] - points[1]
        if cv2.norm(edge1) > cv2.norm(edge2):
            rectAngle = math.degrees(math.atan2(edge1[1], edge1[0]))
        else:
            rectAngle = math.degrees(math.atan2(edge2[1], edge2[0]))
        startpt = info['centroid']
        gradient = np.deg2rad(rectAngle)
        endpt = (int(startpt[0] + 200 * math.cos(gradient)),
                 int(startpt[1] + 200 * math.sin(gradient)))
        startpt = (int(startpt[0]), int(startpt[1]))
        cv2.line(outImg, startpt, endpt, (0, 255, 0), 2)
        cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255, 0, 0), 2)
        cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, BLUE, 2)
        info['angle'] = 90 - abs(rectAngle) if rectAngle >= -90 else 90 - abs(rectAngle)
        #info['angle'] = (info['angle']-90)%360

    @staticmethod
    def averageCentroids(centroids):
        x = int(sum(c[0] for c in centroids) / float(len(centroids)))
        y = int(sum(c[1] for c in centroids) / float(len(centroids)))
        return (x, y)

    @staticmethod
    def getRectArea(cnt):
        rect = cv2.minAreaRect(cnt)
        return int(rect[1][0] * rect[1][1])

    @staticmethod
    def getCorner(box):
        x = [i[0] for i in box]
        y = [i[1] for i in box]
        top_left = (min(x), max(y))
        top_right = (max(x), max(y))
        bot_right = (max(x), min(y))
        bot_left = (min(x), min(y))
        return [top_left, top_right, bot_left, bot_right]

    @staticmethod
    def approxCnt(cnt, offset=0.05):
        """lower offset yields better approximation"""
        epsilon = offset * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return approx

    @staticmethod
    def getCentroid(cnt):
        mom = cv2.moments(cnt)
        centroid_x = int((mom['m10'] + 0.0001) / (mom['m00'] + 0.0001))
        centroid_y = int((mom['m01'] + 0.0001) / (mom['m00'] + 0.0001))
        return (centroid_x, centroid_y)

    @staticmethod
    def getCovexity(cnts):
        P = cv2.arcLength(cnts, True)
        P_convex = cv2.arcLength(cv2.convexHull(cnts), True)
        return P_convex / P

    @staticmethod
    def getHu(cnts):
        return cv2.HuMoments(cv2.moments(cnts))

    @staticmethod
    def getCompactness(cnts):
        P_circle = ((cv2.contourArea(cnts) * math.pi)**0.5) * 2
        P = cv2.arcLength(cnts, True)
        return P_circle / P

    @staticmethod
    def detectRailBlack(img, info, blank):
        cent = (-1, -1)
        mask = VUtil.filterBlack(img)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        outImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, hierr = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=cv2.contourArea, reverse=True)

        if len(contours) >= 1:
            for currCnt in contours:
                rect = cv2.minAreaRect(currCnt)
                if cv2.contourArea(currCnt) > 5000:
                    info['centroid'] = VUtil.getCentroid(currCnt)
                    VUtil.drawInfo(outImg, info)
                    VUtil.getRailDOA(currCnt, outImg, info, blank)
                    break
        return outImg

    @staticmethod
    def detectRail(gray, info, blank):
        chosen_cnt = []
        chosen_cntx = []
        cent = (-1, -1)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        #gray = cv2.GaussianBlur(gray, (9,9),2)
        area = gray.shape[0] * gray.shape[1]
        min = np.amin(gray)
        max = np.amax(gray)
        thresh = min + (max - min) / 1.5
        mask = np.uint8(cv2.Canny(gray, thresh / 2, thresh))
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kern, iterations=1)
        #mask = cv2.erode(mask, kern, iterations=1)
        outImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, hierr = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=cv2.contourArea, reverse=True)

        info['detected'] = False
        if len(contours) >= 1:
            for currCnt in contours:
                rect = cv2.minAreaRect(currCnt)

                if 300 < cv2.contourArea(currCnt) < 3000 and VUtil.checkRectangle(currCnt):
                    chosen_cnt.append(VUtil.getCentroid(currCnt))
                    chosen_cntx.append(currCnt)
                    info['centroid'] = VUtil.getCentroid(currCnt)
                    VUtil.drawInfo(outImg, info)
                    # VUtil.getRailDOA(currCnt,outImg,info,blank)
            if len(chosen_cnt) > 1:
                info['detected'] = True
                info['centroid'] = VUtil.averageCentroids(chosen_cnt)
                VUtil.groupContoursAlign(chosen_cntx, outImg, info, blank)
                VUtil.drawInfo(outImg, info)

            print(info['detected'])
        return outImg

    @staticmethod
    def detectSmallSquare(gray, info, blank, sm=50):
        chosen_cnt = []
        chosen_cntx = []
        cent = (-1, -1)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        #gray = cv2.GaussianBlur(gray, (9,9),2)
        area = gray.shape[0] * gray.shape[1]
        min = np.amin(gray)
        max = np.amax(gray)
        thresh = min + (max - min) / 1.5
        mask = np.uint8(cv2.Canny(gray, thresh / 2, thresh))
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kern, iterations=1)
        #mask = cv2.erode(mask, kern, iterations=1)
        outImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, hierr = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=cv2.contourArea, reverse=True)

        if len(contours) >= 1:
            for currCnt in contours:
                rect = cv2.minAreaRect(currCnt)
                ellipse = cv2.fitEllipse(currCnt)
                # if cv2.contourArea(currCnt) > 5000 and (ellipse[1][1]/ellipse[1][0]) >= 3:
                if cv2.contourArea(currCnt) > 1000 and VUtil.checkRectangle(currCnt):
                    info['detected'] = True
                    cent = VUtil.getCentroid(currCnt)
                    cent = (cent[0], cent[1] - 70)
                    chosen_cnt.append(cent)
                    chosen_cntx.append(currCnt)
                    info['centroid'] = cent
                    VUtil.drawInfo(outImg, info)
                    VUtil.getRailDOA(currCnt, outImg, info, blank)

            if len(chosen_cnt) > 1:
                info['detected'] = True
                info['centroid'] = VUtil.averageCentroids(chosen_cnt)
                VUtil.groupContoursAlign(chosen_cntx, outImg, info, blank)
                chosen_cnt.sort(key=lambda x: x[0], reverse=True)
                chosen_cntx.sort(key=cv2.contourArea, reverse=True)
                chosen_cntx.sort(key=lambda x: VUtil.getCentroid(x)[0], reverse=True)
                rect_r = cv2.minAreaRect(chosen_cntx[0])
                rect_l = cv2.minAreaRect(chosen_cntx[-1])
                cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect_l))], -1, PURPLE, 3)
                cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect_r))], -1, YELLOW, 3)
                cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect_l))], -1, PURPLE, 2)
                cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect_r))], -1, YELLOW, 2)
                if sm < 0:
                    info['centroid'] = chosen_cnt[-1]
                else:
                    info['centroid'] = chosen_cnt[0]
                VUtil.drawInfo(outImg, info)
        return outImg

    @staticmethod
    def detectEdge(gray):
        blur1 = cv2.GaussianBlur(gray, (5, 5), 2)
        blur2 = cv2.GaussianBlur(gray, (5, 5), 5)
        laplacian = VUtil.toBGR(blur1 - blur2, 'gray')
        min = np.min(gray)
        max = np.max(gray)
        thresh = min + (max - min)
        canny = VUtil.toBGR(np.uint8(cv2.Canny(gray, thresh / 2, thresh)), 'gray')
        return np.hstack((laplacian, canny))

    @staticmethod
    def getRailBox(img, info, blank, sm=50):
        chosen_cnt = []
        chosen_cntx = []
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        v = cv2.GaussianBlur(v, (5, 5), 0)
        v = cv2.GaussianBlur(v, (9, 9), 0)
        threshImg = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        threshImg = cv2.erode(threshImg, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        outImg = VUtil.toBGR(threshImg, 'gray')
        contours, hierr = cv2.findContours(threshImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 1:
            hierr = hierr[0]
            for component in zip(contours, hierr):
                currCnt = component[0]
                parent = component[1][3]
                rect = cv2.minAreaRect(currCnt)
                currCnt = cv2.convexHull(currCnt)
                if cv2.contourArea(currCnt) > 2000 and VUtil.checkRectangle(currCnt):
                    chosen_cnt.append(VUtil.getCentroid(currCnt))
                    chosen_cntx.append(currCnt)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 3)
                    cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, RED, 2)

            if chosen_cnt:
                chosen_cnt.sort(reverse=True)
                chosen_cntx.sort(key=cv2.contourArea, reverse=True)
                chosen_cntx.sort(key=lambda x: VUtil.getCentroid(x)[0], reverse=True)
                rect_l = cv2.minAreaRect(chosen_cntx[0])
                rect_r = cv2.minAreaRect(chosen_cntx[-1])
                cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect_l))], -1, BLUE, 3)
                cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect_r))], -1, BLUE, 3)
                cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect_l))], -1, BLUE, 2)
                cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect_r))], -1, BLUE, 2)
                cent = VUtil.averageCentroids(chosen_cnt)
                info['centroid'] = (cent[0], cent[1])
                VUtil.drawInfo(outImg, info)
                # VUtil.getDOA(chosen_cntx[0],outImg,info)
        return outImg

    @staticmethod
    def getRailBox2(img, info, blank, sm=50):
        chosen_cnt = []
        chosen_cntx = []
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        threshImg = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
        threshImg = cv2.erode(threshImg, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        outImg = VUtil.toBGR(threshImg, 'gray')
        contours, hierr = cv2.findContours(threshImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 1:
            hierr = hierr[0]
            for component in zip(contours, hierr):
                currCnt = component[0]
                parent = component[1][3]
                rect = cv2.minAreaRect(currCnt)
                if cv2.contourArea(currCnt) > 1000 and VUtil.checkRectangle(currCnt):
                    chosen_cnt.append(VUtil.getCentroid(currCnt))
                    chosen_cntx.append(currCnt)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 3)
                    cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, RED, 2)
            if chosen_cnt:
                chosen_cnt.sort(key=lambda x: x[0], reverse=True)
                chosen_cntx.sort(key=cv2.contourArea, reverse=True)
                if sm > 0:
                    cent = chosen_cnt[1]
                else:
                    cent = chosen_cnt[-2]
                info['centroid'] = (cent[0], cent[1] - 50)
                VUtil.drawInfo(outImg, info)
        return outImg

    @staticmethod
    def getRail(img):
        info = dict()
        chosen_cnt = []
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        threshImg = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
        threshImg = cv2.erode(threshImg, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        outImg = VUtil.toBGR(threshImg, 'gray')
        contours, hierr = cv2.findContours(threshImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >= 1:
            hierr = hierr[0]
            for component in zip(contours, hierr):
                currCnt = component[0]
                parent = component[1][3]
                rect = cv2.minAreaRect(currCnt)
                if parent != -1 and cv2.contourArea(currCnt) > 1500:
                    chosen_cnt.append(VUtil.getCentroid(currCnt))
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 3)
            if chosen_cnt:
                #cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0,0,255),3)
                cent = VUtil.averageCentroids(chosen_cnt)
                info['centroid'] = (cent[0], cent[1])
                VUtil.drawInfo(outImg, info)
        return outImg, info

    @staticmethod
    def distanceTransform(gray):
        dt = cv2.distanceTransform(gray, cv2.cv.CV_DIST_L2, 5)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    """Underwater Image Enhancement"""

    @staticmethod
    def meanFilter(chan):
        y, x = chan.shape[:2]
        chan = cv2.resize(chan, (x / 2, y / 2))
        return np.uint8(VUtil.blockiter(chan, np.mean, blksize=(10, 10)))

    @staticmethod
    def log_chroma(img):
        """Log-chromacity"""
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

    @staticmethod
    def enhance_tan(img):
        """Tan's method to enhance image"""
        gamma = VUtil.gamma_correct(img)
        b, g, r = cv2.split(gamma)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        out = cv2.merge((b, g, r))
        return out

    @staticmethod
    def util_iace(channel):
        min__val, max__val, min_loc, max_loc = cv2.minMaxLoc(channel)
        min_val, max_val = VUtil.hist_info(channel)
        channel_ = (channel - min__val) / (max__val - min__val) * 255.0
        #channel_ = (channel - min_val)/(max_val-min_val)*255.0
        return channel_

    @staticmethod
    def iace(img):
        b, g, r = cv2.split(img)
        b_ = VUtil.util_iace(b)
        g_ = VUtil.util_iace(g)
        r_ = VUtil.util_iace(r)
        out = cv2.merge((np.uint8(b_), np.uint8(g_), np.uint8(r_)))  # scale up to 255 range
        return out

    @staticmethod
    def french_preprocess(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
        homo = VUtil.deilluminate_single(y)
        ansio = cv2.GaussianBlur(homo, (5, 5), 1)
        bgr = cv2.cvtColor(cv2.merge((ansio, cr, cb)), cv2.COLOR_YCR_CB2BGR)
        b, g, r = cv2.split(bgr)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        out = cv2.merge((b, g, r))
        return out

    @staticmethod
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

    @staticmethod
    def grayworld_modified(img):
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

    @staticmethod
    def sod_minkowski(img):
        """Minkowski P-Norm Shades of Grey"""
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

    @staticmethod
    def sodnorm1(img):
        """Shades of gray norm 1"""
        b, g, r = cv2.split(img)
        gray = np.max([np.mean(b), np.mean(g), np.mean(r)])
        r = cv2.normalize(gray / np.mean(r) * r, 0, 255, cv2.NORM_MINMAX) * 255
        b = cv2.normalize(gray / np.mean(b) * b, 0, 255, cv2.NORM_MINMAX) * 255
        g = cv2.normalize(gray / np.mean(g) * g, 0, 255, cv2.NORM_MINMAX) * 255
        return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))

    @staticmethod
    def dark_channel(img):
        """Dark Channel Prior"""
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        darkMap = np.zeros(img.shape[:2], dtype=np.uint8)
        tMap = np.zeros(img.shape[:2], dtype=np.float32)
        h, w, _ = img.shape
        w /= 40
        h /= 40
        b, g, r = cv2.split(img)
        y, cb, cr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
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
        b = cv2.normalize(b, cv2.NORM_MINMAX) * 255 * 255
        g = (g - gmax) / tMap + gmax
        g = cv2.normalize(g, cv2.NORM_MINMAX) * 255 * 255
        #r = (r - rmax)/tMap+rmax
        return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
        # return cv2.cvtColor(np.uint8(tMap*255),cv2.COLOR_GRAY2BGR)

    @staticmethod
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
        r_min = VUtil.blockiter(1 - r, np.min) / float(1 - A[2])
        g_min = VUtil.blockiter(g, np.min) / float(A[1])
        b_min = VUtil.blockiter(b, np.min) / float(A[0])
        tMap = t_bound - np.min([r_min, b_min, g_min], axis=0)
        tMap = cv2.GaussianBlur(tMap, (11, 11), 0)
        # return VUtil.toBGR(np.uint8(tMap*255), 'gray')
        return VUtil.redchannel_util(img, A, tMap)

    @staticmethod
    def redchannel_util(img, A, t):
        bgr = cv2.split(img)
        bgr = [np.float32(i / 255.0) for i in bgr]
        t_bound = np.full(img.shape[:2], 0.1)
        additive = [(1 - i) * i for i in A]
        J = [(i - A[x]) / np.maximum(t, t_bound) + additive[x] for x, i in enumerate(bgr)]
        J = [np.uint8(VUtil.z_norm(j) * 255) for j in J]
        print(np.max(J[0]))
        print(np.max(J[1]))
        print(np.max(J[2]))
        return cv2.merge(tuple(J))

    @staticmethod
    def naive_fusionmap(img):
        """Merge weight maps without multiple scale fusion"""
        img = cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2))
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        maps = [VUtil.getExposedness(img), VUtil.getSalient(gray), VUtil.getLuminance(img)]
        maps = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in maps]
        mean = np.mean(maps, axis=0) / 255.0
        b = np.uint8(mean * b)
        g = np.uint8(mean * g)
        r = np.uint8(mean * r)
        return cv2.merge((b, g, r))

    @staticmethod
    def showWMaps(img):
        """Debug all 6 weight maps before fusion"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplace = VUtil.Laplacian(img)
        local_contrast = VUtil.getLocalContrast(img)
        exposedness = VUtil.getExposedness(img)
        chromatic = VUtil.getChromatic(img)
        salient = VUtil.getSalient(gray)
        luminance = VUtil.getLuminance(img)
        h1 = np.hstack((laplace, local_contrast, exposedness))
        h2 = np.hstack((chromatic, salient, luminance))
        return np.vstack((h1, h2))

    @staticmethod
    def getLocalContrast(img):  # can find also std between channel and saturation
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        blur = cv2.GaussianBlur(v, (5, 5), 0)
        final = np.std([v, blur], axis=0)
        final = cv2.normalize(final, 0, 255, cv2.NORM_MINMAX) * 255
        return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def getExposedness(img):  # can find also std between channel and saturation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sigma = 2 * (0.25**2)
        final = np.power(img - 0.5, 2) / sigma
        final = cv2.normalize(final, 0, 255, cv2.NORM_MINMAX)
        final = np.exp(-1 * final) * 255
        return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def getChromatic(img):  # can find also std between channel and saturation
        b, g, r = cv2.split(img)
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        final = np.std([b, g, r, s], axis=0)
        return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def getSalient(chan):
        empty = np.ones_like(chan)
        mean = np.mean(chan)
        mean = empty * mean
        blur = cv2.GaussianBlur(chan, (21, 21), 1)
        final = mean - blur
        final = final.clip(min=0)
        final = np.uint8(final)
        print(np.std(final))
        return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def getSalientColor(img):
        a, b, c = cv2.split(img)
        a = VUtil.getSalient(a)
        b = VUtil.getSalient(b)
        c = VUtil.getSalient(c)
        return np.hstack((a, b, c))

    @staticmethod
    def getLuminance(img):
        b, g, r = cv2.split(img)
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        final = np.std([b, g, r, v], axis=0)
        return cv2.cvtColor(np.uint8(final), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def Laplacian(img):
        return cv2.cvtColor(
            np.uint8(
                cv2.Laplacian(
                    cv2.cvtColor(
                        img,
                        cv2.COLOR_BGR2GRAY),
                    cv2.CV_64F)),
            cv2.COLOR_GRAY2BGR)

    @staticmethod
    def chromaiter(img, cycle=2):
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

    @staticmethod
    def finlayiter(img, cycle=2):
        b, g, r = cv2.split(img)
        b = np.float32(b)
        g = np.float32(g)
        r = np.float32(r)
        for i in xrange(cycle):
            b = b / (b + g + r) * 255
            g = g / (b + g + r) * 255
            r = r / (b + g + r) * 255
            bmean = np.mean(b)
            gmean = np.mean(g)
            rmean = np.mean(r)
            b = b / bmean
            g = g / gmean
            r = r / rmean
            b = cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255
            g = cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255
            r = cv2.normalize(r, 0, 255, cv2.NORM_MINMAX) * 255
        out = cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
        return out

    @staticmethod
    def finlaynorm(img, cycle=2):
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

    @staticmethod
    def perfectnorm(img, cycle=2):
        b, g, r = cv2.split(img)
        b = np.float32(b)
        g = np.float32(g)
        r = np.float32(r)
        b = b / (b + g + r) * 255
        g = g / (b + g + r) * 255
        r = r / (b + g + r) * 255
        bmean = np.mean(b)
        gmean = np.mean(g)
        rmean = np.mean(r)
        b = b / bmean
        g = g / gmean
        r = r / rmean
        b = cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255
        g = cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255
        r = cv2.normalize(r, 0, 255, cv2.NORM_MINMAX) * 255
        b = np.float32(b)
        g = np.float32(g)
        r = np.float32(r)
        b = b / (b + g + r) * 255
        g = g / (b + g + r) * 255
        r = r / (b + g + r) * 255
        out = cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
        return out

    @staticmethod
    def chromanorm(img):
        b, g, r = cv2.split(img)
        b = np.float32(b)
        g = np.float32(g)
        r = np.float32(r)
        b = b / (b + g + r) * 255
        g = g / (b + g + r) * 255
        r = r / (b + g + r) * 255
        out = cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
        return out

    @staticmethod
    def noniternorm(img):
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
        return cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))

    """Clustering algorithms"""

    @staticmethod
    def kmeans(img):
        Z = np.float32(img.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2

    @staticmethod
    def slic(img):
        img = cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2))
        image = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        segments = slic(image, n_segments=100, sigma=10, compactness=30)
        img = mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), segments)
        return cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR), segments

    """Preprocessing"""

    @staticmethod
    def DoG(img, kern1=(3, 3), kern2=(5, 5)):
        """Difference of Gaussian using diff kernel size"""
        smooth1 = cv2.GaussianBlur(img, kern1, 0)
        smooth2 = cv2.GaussianBlur(img, kern2, 0)
        final = smooth1 - smooth2
        return final

    @staticmethod
    def normIllumColor(img, gamma=2.2):
        img = np.float32(img)
        img /= 255.0
        img = cv2.pow(img, 1 / gamma) * 255
        img = np.uint8(img)
        return img

    @staticmethod
    def gamma_correct(img):
        gamma = 2.2
        inverse_gamma = 1.0 / gamma
        b, g, r = cv2.split(img)
        b = np.uint8(cv2.pow(b / 255.0, inverse_gamma) * 255)
        g = np.uint8(cv2.pow(g / 255.0, inverse_gamma) * 255)
        r = np.uint8(cv2.pow(r / 255.0, inverse_gamma) * 255)
        return cv2.merge((b, g, r))

    @staticmethod
    def sharpen(img):
        blur = cv2.GaussianBlur(img, (5, 5), 5)
        res = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        return res

    @staticmethod
    def deilluminate(img):
        h, s, gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        blur = cv2.GaussianBlur(gray, (63, 63), 41)
        gray = cv2.log(np.float32(gray))
        blur = cv2.log(np.float32(blur))
        res = np.exp(gray - blur)
        res = cv2.normalize(res, 0, 255, cv2.NORM_MINMAX) * 255
        v = np.uint8(res)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    @staticmethod
    def homomorphic(img):
        h, s, gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        gray = cv2.log(np.float32(gray))
        blur = cv2.GaussianBlur(gray, (63, 63), 41)
        res = np.exp(gray - blur)
        res = cv2.normalize(res, 0, 255, cv2.NORM_MINMAX) * 255
        v = np.uint8(res)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    @staticmethod
    def deilluminate_single(gray):
        blur = cv2.GaussianBlur(gray, (63, 63), 41)
        gray = cv2.log(np.float32(gray))
        blur = cv2.log(np.float32(blur))
        res = np.exp(gray - blur)
        res = cv2.normalize(res, 0, 255, cv2.NORM_MINMAX) * 255
        gray = np.uint8(res)
        return gray

    @staticmethod
    def motiondeflicker(frames, img):
        log_median = cv2.log(np.float32(np.median(frames, axis=0)))
        log_img = cv2.log(np.float32(img))
        diff = cv2.GaussianBlur(log_img - log_median, (21, 21), 0)
        res = img / np.exp(diff)
        res = res.clip(max=255)
        blur = cv2.GaussianBlur(np.uint8(res), (5, 5), 0)
        res = cv2.addWeighted(np.uint8(res), 1.5, blur, -0.5, 0)
        return res

    @staticmethod
    def deilluminate2(img):
        b, g, r = cv2.split(img)
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        log_v = cv2.log(np.float32(v))
        blur_v = cv2.log(np.float32(cv2.GaussianBlur(v, (63, 63), 41)))
        res = np.exp(log_v - blur_v)
        return cv2.cvtColor(np.uint8(res * 255), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def gamma1(gray):
        gray = np.float32(gray)
        gray /= 255.0
        gray = 0.3 * ((cv2.log(2 * gray + 0.1)) + abs(np.log(0.1)))
        return np.uint8(gray * 255)

    @staticmethod
    def gamma2(gray):
        gray = np.float32(gray)
        gray /= 255.0
        gray = 0.8 * (cv2.pow(gray, 2))
        return np.uint8(gray * 255)

    @staticmethod
    def gamma3(gray):
        gray = np.float32(gray)
        gray /= 255.0
        total = 1 / (np.exp(8 * (gray - 0.5)) + 1) * 255
        return np.uint8(total)

    @staticmethod
    def gamma1color(img):
        b, g, r = cv2.split(img)
        b = VUtil.gamma1(b)
        g = VUtil.gamma1(g)
        r = VUtil.gamma1(r)
        return cv2.merge((b, g, r))

    @staticmethod
    def gamma2color(img):
        b, g, r = cv2.split(img)
        b = VUtil.gamma2(b)
        g = VUtil.gamma2(g)
        r = VUtil.gamma2(r)
        return cv2.merge((b, g, r))

    @staticmethod
    def gamma3color(img):
        b, g, r = cv2.split(img)
        b = VUtil.gamma3(b)
        g = VUtil.gamma3(g)
        r = VUtil.gamma3(r)
        return cv2.merge((b, g, r))

    """Features"""
    @staticmethod
    def rg(img):
        b, g, r = cv2.split(img)
        return cv2.absdiff(r, g)

    @staticmethod
    def conspicuityMaps(img):
        """Generate conspicutiy maps from intensity"""
        b, g, r = cv2.split(img)
        b = np.float32(b)
        g = np.float32(g)
        r = np.float32(r)
        intensity = np.mean(np.array([b, g, r]), axis=0)
        b /= intensity
        g /= intensity
        r /= intensity
        b = cv2.normalize(b, 0, 255, cv2.NORM_MINMAX) * 255
        g = cv2.normalize(g, 0, 255, cv2.NORM_MINMAX) * 255
        r = cv2.normalize(r, 0, 255, cv2.NORM_MINMAX) * 255
        normBGR = cv2.merge((np.uint8(b), np.uint8(g), np.uint8(r)))
        R = r - (g + b) / 2
        G = g - (r + b) / 2
        B = b - (r + g) / 2
        Y = (r + g) / 2 - abs(r - g) / 2
        Y = Y.clip(min=0)
        #out = cv2.cvtColor(np.uint8(intensity), cv2.COLOR_GRAY2BGR)
        R = cv2.cvtColor(np.uint8(R), cv2.COLOR_GRAY2BGR)
        B = cv2.cvtColor(np.uint8(B), cv2.COLOR_GRAY2BGR)
        G = cv2.cvtColor(np.uint8(G), cv2.COLOR_GRAY2BGR)
        Y = cv2.cvtColor(np.uint8(Y), cv2.COLOR_GRAY2BGR)
        return np.hstack((B, G, R, Y))

    @staticmethod
    def orb(img):
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        out = None
        out = cv2.drawKeypoints(img, kp, out, color=(0, 0, 255), flags=0)
        return out

    @staticmethod
    def harris(img, block=21, aperture=11, param=0.2):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = VUtil.finlaynorm(img)
        gray, g, r = cv2.split(img)
        corner = cv2.cornerHarris(np.float32(gray), block, aperture, param)
        corner = cv2.dilate(corner, None)
        img[corner > 0.01 * corner.max()] = (0, 0, 255)
        return img

    @staticmethod
    def fastDetector(img):
        fast = cv2.FastFeatureDetector()
        kp = fast.detect(img, None)
        img = cv2.drawKeypoints(img, kp, color=(255, 0, 0))
        return img

    @staticmethod
    def shiDetector(img):
        corners = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 25, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, 255, -1)
        return img

    @staticmethod
    def briefDetector(img):
        star = cv2.FeatureDetector_create('STAR')
        brief = cv2.DescriptorExtractor_create('BRIEF')
        kp = star.detect(img, None)
        kp, desc = brief.compute(img, kp)
        img = cv2.drawKeypoints(img, kp, color=(255, 0, 0))
        return img

    @staticmethod
    def surfDetector(img):
        surf = cv2.SURF(400)
        kp, desc = surf.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, kp, color=(255, 0, 0))
        return img

    @staticmethod
    def mserDetector(img):
        mser = cv2.FeatureDetector_create('MSER')
        kp = mser.detect(img)
        orb = cv2.DescriptorExtractor_create('ORB')
        kp, desc = orb.compute(img, kp)
        img = cv2.drawKeypoints(img, kp, color=(255, 0, 0))
        return img

    @staticmethod
    def generateSalientLAB(img):
        blur = cv2.GaussianBlur(img, (9, 9), 2)
        l, a, b = cv2.split(cv2.cvtColor(blur, cv2.COLOR_LBGR2LAB))
        mean_l = np.full(img.shape[:2], np.mean(l))
        mean_a = np.full(img.shape[:2], np.mean(a))
        mean_b = np.full(img.shape[:2], np.mean(b))
        total = VUtil.euclid_dist([(l, mean_l), (a, mean_a), (b, mean_b)])
        total = np.uint8(total)
        return VUtil.toBGR(total, 'gray')

    @staticmethod
    def generateNewColor(img):
        b, g, r = cv2.split(img)
        c1 = VUtil.toBGR(np.uint8(np.arctan2(r, np.maximum(b, g)) * 255), 'gray')
        c2 = VUtil.toBGR(np.uint8(np.arctan2(g, np.maximum(r, b)) * 255), 'gray')
        c3 = VUtil.toBGR(np.uint8(np.arctan2(b, np.maximum(r, g)) * 255), 'gray')
        denominator = cv2.pow(r - g, 2) + cv2.pow(r - b, 2) + cv2.pow(g - b, 2)
        l1 = VUtil.toBGR(cv2.pow(r - g, 2) / denominator, 'gray')
        l2 = VUtil.toBGR(cv2.pow(r - b, 2) / denominator, 'gray')
        l3 = VUtil.toBGR(cv2.pow(g - b, 2) / denominator, 'gray')
        return np.vstack((np.hstack((c1, c2, c3)), np.hstack((l1, l2, l3))))

    """Analysis"""

    @staticmethod
    def analyzeChan(chan):
        outImg = VUtil.toBGR(chan, 'gray')
        min, mean, mid, max = (np.min(chan), np.mean(chan), np.median(chan), np.max(chan))
        var, std = (np.var(chan), np.std(chan))
        median_img = cv2.threshold(chan, mid, 255, cv2.THRESH_BINARY)[1]
        mean_img = cv2.threshold(chan, 90, 255, cv2.THRESH_BINARY)[1]
        print("Min:%f, Mean:%f, Median:%f, Max:%f" % (min, mean, mid, max))
        print("Var:%f, Std:%f" % (var, std))
        return VUtil.toBGR(chan, 'gray')

    @staticmethod
    def analyzeSalient(chan):
        empty = np.ones_like(chan)
        mean = np.mean(chan)
        mean = empty * mean
        blur = cv2.GaussianBlur(chan, (21, 21), 1)
        final = mean - blur
        final = final.clip(min=0)
        final = np.uint8(final)
        return np.std(final)

    @staticmethod
    def analyze(img):
        hsv = VUtil.getHSV(img)
        bgr = VUtil.getRGB(img)
        luv = VUtil.getLUV(img)
        lab = VUtil.getLAB(img)
        return np.vstack((bgr, hsv, luv, lab))

    @staticmethod
    def getRGB(img):
        b, g, r = cv2.split(img)
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        return np.hstack((b, g, r))

    @staticmethod
    def getHSV(img):
        return VUtil.getRGB(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    @staticmethod
    def getLUV(img):
        return VUtil.getRGB(cv2.cvtColor(img, cv2.COLOR_BGR2LUV))

    @staticmethod
    def getYCB(img):
        return VUtil.getRGB(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))

    @staticmethod
    def getLAB(img):
        return VUtil.getRGB(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

    @staticmethod
    def bgrstd(img):
        b, g, r = cv2.split(img)
        std = np.std([b, g, r])
        return std

    @staticmethod
    def labvar(img):
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        var = np.var([l, a, b])
        return var

    @staticmethod
    def vc(img):
        h, l, s = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
        roughness = np.std(l) / np.mean(l)
        return roughness

    @staticmethod
    def channelThresh(img, block=51, offset=5):
        a, b, c = cv2.split(img)
        a = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, offset)
        b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, offset)
        c = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, offset)
        return np.hstack((VUtil.toBGR(a, 'gray'), VUtil.toBGR(b, 'gray'), VUtil.toBGR(c, 'gray')))

    @staticmethod
    def channelThreshColor(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a = VUtil.channelThresh(img)
        b = VUtil.channelThresh(hsv)
        c = VUtil.channelThresh(lab)
        return np.vstack((a, b, c))

    """Remove false positives"""

    @staticmethod
    def getBlobs(threshImg):
        contours, hierr = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            return contours, hierr[0]
        else:
            return None

    @staticmethod
    def numOfBlobs(contours):
        return len(contours)

    @staticmethod
    def meanBlobsSize(contours):
        area = [cv2.contourArea(i) for i in contours]
        return np.mean(area)

    @staticmethod
    def filterBlobSize(contours, minSize=200, maxSize=100000):
        return [lambda cnt:minSize < cv2.contourArea(cnt) < maxSize for cnt in contours]

    @staticmethod
    def feature(img, channel, lv=9):
        comm_x = img.shape[1] / (2**(lv / 2 - 1))
        comm_y = img.shape[0] / (2**(lv / 2 - 1))
        scaled = [channel(img)]
        # Create n level feature maps
        for i in xrange(lv - 1):
            scaled.append(cv2.pyrDown(scaled[-1]))

        features = []
        for x in xrange(1, lv - 5):
            big = scaled[x]
            for y in (3, 4):
                small = scaled[x + y]
                dstsize = big.shape[1], big.shape[0]
                small = cv2.resize(small, dstsize)
                features.append(cv2.absdiff(big, small))
        features = [cv2.resize(i, (comm_x, comm_y)) for i in features]
        return VUtil.toBGR(np.uint8(np.mean(features, axis=0)), 'gray')

    @staticmethod
    def goodThresh(img, mask, limit=0.7):
        white = cv2.countNonZero(mask)
        size = img.shape[1] * img.shape[0]
        ratio = white / float(size)
        print(ratio)
        return ratio < limit

    """Threshold"""

    @staticmethod
    def thresh_orange2(img):
        imgg = VUtil.finlaynorm(img)
        b, g, r = cv2.split(imgg)
        hsv = VUtil.toHSV(imgg)
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

    @staticmethod
    def threshChan(chan, lo=None, hi=None):
        if not lo and not hi:
            lo = np.mean(chan) - 5
            hi = np.mean(chan) + 5
        return cv2.inRange(chan, lo, hi)

    @staticmethod
    def contourRect(thresh):
        outImg = VUtil.toBGR(thresh, 'gray')
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

    @staticmethod
    def cannyThresh(chan, offset=8):
        chan = cv2.GaussianBlur(chan, (5, 5), 0)
        min, max, mean = np.min(chan), np.max(chan), np.mean(chan)
        min += (max - min) / offset
        out = np.uint8(cv2.Canny(chan, min / 2, min, apertureSize=5))
        return out

    @staticmethod
    def filterBlack(img):
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        return cv2.inRange(v, 0, 110)

    @staticmethod
    def filterWhite(img):
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        mask = cv2.inRange(v, 150, 255)
        return cv2.bitwise_not(mask)

    @staticmethod
    def getPatternOrange(img):
        img = VUtil.finlaynorm(img)
        b, g, r = cv2.split(img)
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 201, 13)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.add(r, b)

    @staticmethod
    def getPatternValue(img, mode=1):
        b, g, r = cv2.split(img)
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        v = VUtil.normIllumColor(v, 2.5)
        v = VUtil.adaptThresh(v, 15, 10)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        v = cv2.erode(v, kern2, iterations=2)
        v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, kern2, iterations=2)
        return v

    @staticmethod
    def adaptThresh(gray, block=31, offset=5):
        res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                    block, offset)
        return res

    @staticmethod
    def getPattern1(img, mode=1):
        img = VUtil.finlaynorm(img)
        b, g, r = cv2.split(img)
        b = VUtil.adaptThresh(b, 41, 5)
        b = cv2.bitwise_not(b)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        b = cv2.erode(b, kern, iterations=1)
        b = cv2.dilate(b, kern2, iterations=3)
        #b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kern,iterations=1)
        return b

    @staticmethod
    def getPatternHue(img, mode=1):
        b, g, r = cv2.split(img)
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        h = VUtil.adaptThresh(h, 61, 3)
        h = cv2.bitwise_not(h)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        #h = cv2.dilate(h, kern2,iterations=1)
        return h

    @staticmethod
    def getPatternWhite(img):
        img = VUtil.finlaynorm(img)
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        a = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 3)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return a

    @staticmethod
    def thresh_orange(img, hsv=0):
        imgg = VUtil.finlaynorm(VUtil.iace(img))
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

    @staticmethod
    def thresh_yellow(img, hsv=0, thresh_val=90):
        imgg = VUtil.finlaynorm(VUtil.iace(img))
        #imgg = VUtil.iace(img)
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
        #thresh = thresh if thresh <= 90 else 90
        common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        if hsv == 0:
            return common & yellow
        if hsv == 1:
            return common
        if hsv == 2:
            return yellow

    @staticmethod
    def thresh_yellow2(img, hsv=0):
        imgg = VUtil.finlaynorm(VUtil.iace(img))
        #imgg = VUtil.iace(img)
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
        #thresh = thresh if thresh <= 90 else 90
        common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        if hsv == 0:
            return common & yellow
        if hsv == 1:
            return common
        if hsv == 2:
            return yellow

    @staticmethod
    def thresh_yellow_orange(img):
        imgg = VUtil.finlaynorm(img)
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

    """Robosub VUtils"""

    @staticmethod
    def drawInfo(img, outData):
        y, x = img.shape[:2]
        center_y = y / 2
        center_x = x / 2
        outData['detected'] = True
        outData['dxy'][0] = (outData['centroid'][0] - center_x) / float(x)
        outData['dxy'][1] = (center_y - outData['centroid'][1]) / float(y)
        VUtil.draw_circle(img, outData['centroid'], 4, (0, 244, 255), -1)
        VUtil.draw_rect(img, (center_x - 5, center_y + 5), (center_x + 5, center_y - 5), (0, 255, 0), 2)

    @staticmethod
    def getBlack2(img):
        max_area = img.shape[0] * img.shape[1]
        white = np.zeros_like(img)
        cent = (-1, -1)
        h, s, gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        gray = VUtil.normIllumColor(gray, 0.5)
        max_pix = np.amax(gray)
        min_pix = np.amin(gray)
        diff = max_pix - min_pix
        thresh = min_pix + diff / 3.5
        black = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)[1]
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        mask = black
        mask = cv2.erode(mask, kern, iterations=1)
        #mask = cv2.bitwise_not(mask)
        outImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, hierr = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(contours) >= 1:
            hierr = hierr[0]
            for component in zip(contours, hierr):
                currCnt = component[0]
                currHierr = component[1]
                parent = currHierr[3]
                currCnt = cv2.convexHull(currCnt)
                if 500 < VUtil.getRectArea(currCnt) < max_area / 2 and VUtil.checkBins(currCnt, 1.1):
                    rect = cv2.minAreaRect(currCnt)
                    cv2.drawContours(outImg, [currCnt], -1, (255, 0, 0), 4)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 4)
                    cv2.drawContours(white, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255, 255, 255), -1)

        return white

    @staticmethod
    def findLane(img, info, blank):
        threshImg = VUtil.thresh_orange(img)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.erode(threshImg, kern, iterations=1)
        threshImg = cv2.dilate(threshImg, kern, iterations=2)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) >= 1:
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                points = np.int32(cv2.cv.BoxPoints(rect))
                edge1 = points[1] - points[0]
                edge2 = points[2] - points[1]
                # Remove false positive by limiting area
                cnt = cv2.convexHull(cnt)
                if cv2.contourArea(cnt) > 700 and VUtil.checkRectangle(cnt):
                    # Draw bounding rect
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255, 0, 0), 3)
                    cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, ORANGE, 2)
                    if cv2.norm(edge1) > cv2.norm(edge2):
                        rectAngle = math.degrees(math.atan2(edge1[1], edge1[0]))
                    else:
                        rectAngle = math.degrees(math.atan2(edge2[1], edge2[0]))
                    info['detected'] = True
                    info['angle'] = 90 - abs(rectAngle) if rectAngle >= -90 else 90 - abs(rectAngle)
                    info['centroid'] = VUtil.getCentroid(cnt)

                    # Draw angle
                    startpt = info['centroid']
                    gradient = np.deg2rad(rectAngle)
                    endpt = (int(startpt[0] + 200 * math.cos(gradient)),
                             int(startpt[1] + 200 * math.sin(gradient)))
                    startpt = (int(startpt[0]), int(startpt[1]))
                    cv2.line(outImg, startpt, endpt, (0, 255, 0), 3)
                    cv2.line(blank, startpt, endpt, GREEN, 2)
                    info['centroid'] = VUtil.getCentroid(cnt)
                    VUtil.drawInfo(outImg, info)
                    break
        return outImg

    @staticmethod
    def checkRectangle(cnt, ratio_limit=1.2):
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        ratio = rect_area / (cv2.contourArea(cnt) + 0.001)
        return ratio < ratio_limit

    @staticmethod
    def checkBins(cnt, ratio_limit=1.2):
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > rect[1][1]:
            asp_rat = rect[1][0] / rect[1][1]
        else:
            asp_rat = rect[1][1] / rect[1][0]
        rect_area = rect[1][0] * rect[1][1]
        ratio = rect_area / (cv2.contourArea(cnt) + 0.001)
        return ratio < ratio_limit and asp_rat < 3

    @staticmethod
    def checkRectangle2(cnt):
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        ratio = rect_area / (cv2.contourArea(cnt) + 0.001)
        return ratio

    @staticmethod
    def checkCircle(cnt):
        circle = cv2.minEnclosingCircle(cnt)
        circle_area = (circle[1]**2) * math.pi
        ratio = circle_area / (cv2.contourArea(cnt) + 0.001)
        return ratio < 1.5

    @staticmethod
    def checkCircle2(cnt):
        circle = cv2.minEnclosingCircle(cnt)
        circle_area = (circle[1]**2) * math.pi
        ratio = circle_area / (cv2.contourArea(cnt) + 0.001)
        return ratio

    @staticmethod
    def checkThunder(cnt):
        rect = cv2.minAreaRect(cnt)
        if rect[1][0] > rect[1][1]:
            asp_rat = rect[1][0] / rect[1][1]
        else:
            asp_rat = rect[1][1] / rect[1][0]
        return -85 < rect[2] < -50 or asp_rat > 2

    @staticmethod
    def checkBanana(cnt):
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        ratio = rect_area / (cv2.contourArea(cnt) + 0.001)
        return 1.5 < ratio < 3.0

    @staticmethod
    def getTrain(threshImg, info, blank):
        info['detected'] = False
        data = []
        chosen_cnt = []
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.dilate(threshImg, kern, iterations=1)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=VUtil.getRectArea, reverse=True)
        if len(contours) >= 1:
            info['heading'] = 0
            for x, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > 500:
                    data.append(VUtil.getCentroid(cnt))
                    chosen_cnt.append(cnt)
                    if x == 0:
                        info['centroid'] = VUtil.getCentroid(cnt)
                        info['area'] = VUtil.getRectArea(cnt) / float(outImg.shape[0] * outImg.shape[1])
                        VUtil.drawInfo(outImg, info)
                    rect = cv2.minAreaRect(cnt)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 1)
                    cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, YELLOW, 2)

            if len(data) > 1:
                info['detected'] = True
                info['heading'] = data[0][0] - data[len(data) - 1][0]
                info['heading'] = -1 if info['heading'] > 0 else 1
                deltaX = 20 if info['heading'] > 0 else -20
                info['centroid'] = VUtil.averageCentroids(data)
                cv2.circle(outImg, info['centroid'], 5, (0, 0, 255), -1)
                x = info['centroid'][0]
                y = info['centroid'][1] + 15
                cv2.circle(outImg, (x, y), 10, (255, 0, 255), 2)
                VUtil.groupContours(chosen_cnt, outImg, info)
                info['centroid'] = (x, y)
                VUtil.drawInfo(outImg, info)

        return outImg

    @staticmethod
    def getDelorean(threshImg, info, blank):
        info['detected'] = False
        data = []
        chosen_cnt = []
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.dilate(threshImg, kern, iterations=1)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=VUtil.getRectArea, reverse=True)
        if len(contours) >= 1:
            info['heading'] = 0
            for x, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > 500:
                    data.append(cnt)
                    if VUtil.getAspRatio(cnt) < 2.5 and x <= 1:
                        info['centroid'] = VUtil.getCentroid(cnt)
                        #info['area'] = VUtil.getRectArea(cnt)
                        info['area'] = VUtil.getRectArea(cnt) / float(outImg.shape[0] * outImg.shape[1])
                        # VUtil.getDOA(cnt,outImg,info)
                        VUtil.drawInfo(outImg, info)
                    rect = cv2.minAreaRect(cnt)
                    #cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0,0,255), 1)
                    cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, ORANGE, 2)

            if len(data) > 1:
                info['detected'] = True
                chosen_cnt = [VUtil.getCentroid(x) for x in data]
                combine = zip(data, chosen_cnt)
                if len(data) > 2:
                    combine.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
                    for i in combine:
                        cnt = i[0]
                        if VUtil.getAspRatio(cnt) < 2.5:
                            rect = cv2.minAreaRect(cnt)
                            cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, PURPLE, 2)
                            biggest = VUtil.getCentroid(cnt)
                            break
                    combine.sort(key=lambda x: cv2.norm(biggest, x[1]))
                    data = [x[0] for x in combine[:2]]
                    chosen_cnt = [x[1] for x in combine[:2]]
                info['heading'] = chosen_cnt[0][0] - chosen_cnt[1][0]
                info['heading'] = -1 if info['heading'] > 0 else 1
                deltaX = 40 if info['heading'] < 0 else -40
                info['centroid'] = VUtil.averageCentroids(chosen_cnt)
                cv2.circle(outImg, info['centroid'], 5, (0, 0, 255), -1)
                x = info['centroid'][0]
                y = info['centroid'][1] + 10
                cv2.circle(outImg, (x, y), 10, (255, 0, 255), 2)
                VUtil.groupContours(data, outImg, info)
                info['centroid'] = (x, y)
                VUtil.drawInfo(outImg, info)

        return outImg

    @staticmethod
    def getBinsShape(mask, info, blank):
        info['pattern'] = [-1, -1, -1, -1]
        data = []
        chosen_cnt = []
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        outImg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, hierr = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(contours) >= 1:
            hierr = hierr[0]
            for cnt, hirr in zip(contours, hierr):
                parent = hirr[3]
                child = hirr[2]
                if parent != -1 and cv2.contourArea(cnt) > 100 and VUtil.checkRectangle(contours[parent]):
                    # data.append(VUtil.getCentroid(cnt))
                    chosen_cnt.append(cnt)
                    rect = cv2.minAreaRect(cnt)
                    #cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0,0,255), 2)
                    info['centroid'] = VUtil.getCentroid(cnt)
                    VUtil.drawInfo(outImg, info)
                    # VUtil.getDOA(cnt,outImg,info)

            if len(data) > 1:
                info['centroid'] = VUtil.averageCentroids(data)
                cv2.circle(outImg, info['centroid'], 5, (0, 0, 255), -1)
                cv2.circle(outImg, info['centroid'], 10, (0, 0, 255), -1)
                VUtil.drawInfo(outImg, info)
            VUtil.classify_cnt(outImg, info, blank, chosen_cnt)
        return outImg

    @staticmethod
    def getBinsBlack(threshImg):
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        y, x = threshImg.shape[:2]
        black = np.zeros((y, x, 3), dtype=np.uint8)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, hierr = cv2.findContours(threshImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(contours) >= 1:
            hierr = hierr[0]
            for cnt, hirr in zip(contours, hierr):
                rect = cv2.minAreaRect(cnt)
                parent = hirr[3]
                if VUtil.checkRectangle(cnt) and parent != -1 and cv2.contourArea(cnt) > 400:
                    cv2.drawContours(black, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255, 255, 255), -1)
                    #cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255,0,255), 2)
        return black

    @staticmethod
    def getBins(threshImg, info, blank):
        info['pattern'] = [-1, -1, -1, -1]
        data = []
        chosen_cnt = []
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.dilate(threshImg, kern, iterations=1)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=VUtil.getRectArea, reverse=True)
        if len(contours) >= 1:
            info['heading'] = 0
            for x, cnt in enumerate(contours):
                cnt = cv2.convexHull(cnt)
                cnt = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
                rect = cv2.minAreaRect(cnt)
                area = rect[1][0] * rect[1][1]
                if cv2.contourArea(cnt) > 300 and x < 4:
                    data.append(VUtil.getCentroid(cnt))
                    chosen_cnt.append(cnt)
                    if x == 0:
                        info['centroid'] = VUtil.getCentroid(cnt)
                        info['area'] = cv2.minAreaRect(cnt)[2] / float(threshImg.shape[0] * threshImg.shape[1])
                        VUtil.drawInfo(outImg, info)
                        VUtil.getDOA(cnt, outImg, info)
                    rect = cv2.minAreaRect(cnt)
                    cv2.drawContours(outImg, [cnt], -1, (255, 0, 255), 2)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 2)

            if len(data) > 1:
                info['heading'] = data[0][0] - data[len(data) - 1][0]
                deltaX = 40 if info['heading'] < 0 else -40
                info['centroid'] = VUtil.averageCentroids(data)
                cv2.circle(outImg, info['centroid'], 5, (0, 0, 255), -1)
                cv2.circle(outImg, info['centroid'], 10, (0, 0, 255), -1)
                VUtil.groupContours(chosen_cnt, outImg, info)
                VUtil.drawInfo(outImg, info)

            VUtil.classify_cnt(outImg, info, blank, chosen_cnt)
        return outImg

    @staticmethod
    def getOverallBins(threshImg, info, blank):
        data = []
        chosen_cnt = []
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.dilate(threshImg, kern, iterations=1)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        if len(contours) >= 1:
            info['heading'] = 0
            for x, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > 300 and x < 4:
                    data.append(VUtil.getCentroid(cnt))
                    chosen_cnt.append(cnt)
                    if x == 0:
                        info['centroid'] = VUtil.getCentroid(cnt)
                        info['area'] = cv2.minAreaRect(cnt)[2] / float(threshImg.shape[0] * threshImg.shape[1])
                        VUtil.drawInfo(outImg, info)
                        VUtil.getDOA(cnt, outImg, info)
                    rect = cv2.minAreaRect(cnt)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 1)
                    cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, YELLOW, 2)

            if len(data) > 1:
                info['centroid'] = VUtil.averageCentroids(data)
                cv2.circle(outImg, info['centroid'], 5, (0, 0, 255), -1)
                cv2.circle(outImg, info['centroid'], 10, (0, 0, 255), -1)
                VUtil.groupContoursAlign(chosen_cnt, outImg, info, blank)
                VUtil.drawInfo(outImg, info)

        return outImg

    @staticmethod
    def getCover(threshImg, info, blank):
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.erode(threshImg, kern, iterations=1)
        threshImg = cv2.morphologyEx(threshImg, cv2.MORPH_CLOSE, kern, iterations=1)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=VUtil.getRectArea, reverse=True)
        if len(contours) >= 1:
            for x, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > 300:
                    rect = cv2.minAreaRect(cnt)
                    x, y = VUtil.getCentroid(cnt)
                    info['centroid'] = (x, y)
                    info['area'] = rect[1][1] * rect[1][0] / float(threshImg.shape[0] * threshImg.shape[1])
                    VUtil.drawInfo(outImg, info)
                    VUtil.getDOA(cnt, outImg, info)
                    rect = cv2.minAreaRect(cnt)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 1)
                    cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, ORANGE, 2)
                    break

        return outImg

    @staticmethod
    def classify_cnt(img, info, blank, chosen_cnt):
        y, x = img.shape[:2]
        center_y = y / 2
        center_x = x / 2
        info['pattern'] = [-1, -1, -1, -1]
        '''
        0  = mr.fusion, 1 = banana, 2 = cola, 3 = thunder
        '''
        chosen_cnt.sort(key=VUtil.getRectArea, reverse=True)

        for cnt in chosen_cnt:

            cent = VUtil.getCentroid(cnt)
            deltaX = (cent[0] - center_x) / float(x)
            deltaY = (center_y - cent[1]) / float(y)
            rect = cv2.minAreaRect(cnt)
            area = (rect[1][1] * rect[1][0]) / float(y * x)

            if VUtil.checkCircle(cnt) and info['pattern'][0] == -1:
                info['pattern'][0] = (deltaX, deltaY, area)
                cv2.circle(img, cent, 5, PURPLE, -1)
                cv2.circle(blank, cent, 5, PURPLE, -1)
                cv2.drawContours(blank, [cnt], -1, PURPLE, 2)

            if VUtil.checkThunder(cnt) and info['pattern'][3] == -1:
                info['pattern'][3] = (deltaX, deltaY, area)
                cv2.circle(img, cent, 5, BLUE, -1)
                cv2.circle(blank, cent, 5, BLUE, -1)
                cv2.drawContours(blank, [cnt], -1, BLUE, 2)

            if VUtil.checkRectangle(cnt) and info['pattern'][2] == -1:
                info['pattern'][2] = (deltaX, deltaY, area)
                cv2.circle(img, cent, 5, RED, -1)
                cv2.circle(blank, cent, 5, RED, -1)
                cv2.drawContours(blank, [cnt], -1, RED, 2)

            if VUtil.checkBanana(cnt) and info['pattern'][1] == -1:
                info['pattern'][1] = (deltaX, deltaY, area)
                cv2.circle(img, cent, 5, GREEN, -1)
                cv2.circle(blank, cent, 5, GREEN, -1)
                cv2.drawContours(blank, [cnt], -1, GREEN, 2)

    @staticmethod
    def getDeltaArea(cnt, x, y):
        center_x = x / 2
        center_y = y / 2
        cent = VUtil.getCentroid(cnt)
        deltaX = (cent[0] - center_x) / float(x)
        deltaY = (center_y - cent[1]) / float(y)
        rect = cv2.minAreaRect(cnt)
        area = (rect[1][1] * rect[1][0]) / float(y * x)
        return (deltaX, deltaY, area)

    @staticmethod
    def findTrain(img, info, blank):
        mask = VUtil.thresh_yellow(img)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return VUtil.getTrain(mask, info, blank)

    @staticmethod
    def findDelorean(img, info, blank):
        mask = VUtil.thresh_orange(img)
        return VUtil.getDelorean(mask, info, blank)

    @staticmethod
    def findOverallBins(img, info, blank, mode=1):
        white = VUtil.getBlack2(img)
        mask = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        if mode != 1:
            mask = mask & VUtil.thresh_yellow(img)
        return VUtil.getOverallBins(mask, info, blank)

    @staticmethod
    def findBins3(img, info, blank, mode=1):
        h, s, v = cv2.split(VUtil.toHSV(img))
        v = cv2.GaussianBlur(v, (3, 3), 0)
        black = VUtil.getBinsBlack(VUtil.adaptThresh(v, 11, 10))
        yellow = VUtil.getPatternHue(VUtil.iace(img))
        orange = cv2.bitwise_not(VUtil.thresh_orange2(img))
        mask = yellow & cv2.cvtColor(black, cv2.COLOR_BGR2GRAY) & orange
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        return VUtil.getBins(mask, info, blank)

    @staticmethod
    def findBins2(img, info, blank, mode=1):
        white = VUtil.getBlack2(img)
        mask = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        mask = mask & VUtil.getPatternHue(cv2.GaussianBlur(VUtil.iace(img), (5, 5), 0))
        orange = VUtil.thresh_orange2(img)
        yellow = VUtil.thresh_yellow(img, 1, 110)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                         iterations=2) & cv2.bitwise_not(orange) & yellow
        return VUtil.getBins(mask, info, blank)

    @staticmethod
    def findBins(img, info, blank, mode=1):
        img = VUtil.finlaynorm(VUtil.iace(img))
        black = cv2.cvtColor(VUtil.getBinsBlack(cv2.bitwise_not(VUtil.filterBlack(img))), cv2.COLOR_BGR2GRAY)
        black = cv2.cvtColor(VUtil.getBinsBlack(VUtil.filterBlack(img)), cv2.COLOR_BGR2GRAY)
        return VUtil.toBGR(black, 'gray')
        mask = VUtil.getPatternHue(img)
        #mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
        return VUtil.getBinsShape(mask, info, blank)

    @staticmethod
    def findCover(img, info, blank):
        mask = VUtil.thresh_orange2(img)
        return VUtil.getCover(mask, info, blank)

    @staticmethod
    def findRail(img, info, blank):
        h, s, v = cv2.split(VUtil.toHSV(img))
        return VUtil.detectRail(v, info, blank)

    @staticmethod
    def findTracks(img):
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        return VUtil.cannyChild(v)

    @staticmethod
    def identifyObject(img, info, blank):
        threshImg = VUtil.thresh_yellow(img)
        info['object'] = 0
        data = []
        chosen_cnt = []
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.dilate(threshImg, kern, iterations=1)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=VUtil.getRectArea, reverse=True)
        if len(contours) >= 1:
            for x, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > 600:
                    data.append(VUtil.getCentroid(cnt))
                    chosen_cnt.append(cnt)
                    rect = cv2.minAreaRect(cnt)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 1)

            if len(data) > 1:
                info['detected'] = True
                hull = cv2.convexHull(np.vstack(chosen_cnt))
                rect = cv2.minAreaRect(hull)
                ellipse = cv2.fitEllipse(hull)
                cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (255, 0, 255), 3)
                cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))], -1, PURPLE, 2)
                aspect_ratio = ellipse[1][1] / ellipse[1][0]
                print("Aspect ratio: " + str(aspect_ratio))
                if aspect_ratio > 1.7:
                    info['object'] = 1
                elif aspect_ratio < 1.5:
                    info['object'] = -1

        return outImg

    @staticmethod
    def detectGeneral(img, info, blank):
        threshImg = VUtil.thresh_yellow2(img)
        threshImg = cv2.erode(threshImg, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        data = []
        chosen_cnt = []
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        threshImg = cv2.dilate(threshImg, kern, iterations=1)
        outImg = cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=VUtil.getRectArea, reverse=True)
        if len(contours) >= 1:
            for x, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > 600 and VUtil.checkRectangle(cnt):
                    info['detected'] = True
                    data.append(VUtil.getCentroid(cnt))
                    chosen_cnt.append(cnt)
                    rect = cv2.minAreaRect(cnt)
                    cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))], -1, (0, 0, 255), 3)
                    info['object'] = 1
                    break

            '''
            if len(data) > 1:
                hull = cv2.convexHull(np.vstack(chosen_cnt))
                rect = cv2.minAreaRect(hull)
                ellipse = cv2.fitEllipse(hull)
                cv2.drawContours(outImg, [np.int0(cv2.cv.BoxPoints(rect))],-1,(255,0,255),3)
                cv2.drawContours(blank, [np.int0(cv2.cv.BoxPoints(rect))],-1,PURPLE,2)
                aspect_ratio = ellipse[1][1]/ellipse[1][0]
                print("Aspect ratio: " + str(aspect_ratio))
                if aspect_ratio > 1.7:
                    info['object'] = 1
                elif aspect_ratio < 1.5:
                    info['object'] = -1
                    '''

        return outImg

    @staticmethod
    def binSequence(img, info, blank):
        overall = VUtil.findOverallBins(img, info, blank)
        cover = VUtil.findCover(img, info, blank)
        bins = VUtil.findBins(img, info, blank)
        return np.hstack((overall, cover, bins))

    @staticmethod
    def homeSequence(img, info, blank):
        train = VUtil.findTrain(img, info, blank)
        delorean = VUtil.findDelorean(img, info, blank)
        general = VUtil.detectGeneral(img, info, blank)
        return np.hstack((general, train, delorean))
