#!/usr/bin/env python
import cv2
import math
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_float
# Custom class
import constant as c
from constant import Color
from preprocess import norm_illum_color
from camera import rect_to_distance, circle_to_distance
from config import Config
from vision.msg import DetectedObject
from core import to_hsv, to_lab, to_bgr
from enhancement import finlaynorm, iace, get_salient, shadegrey, chromanorm

# Red, Orange, Green to calculate hue distance
HUE_RANGE = [0, 15, 60]

# Red to Green range in HSV color space
RED_GREEN = [(0, 0, 0), (80, 255, 255)]

# Area in pixels bound for each object
AREA = {'bin': [1200, 10000], 'cover': [1300, 30000], 'horizontal_coin': [50, 5000],
        'vertical_coin': [50, 5000], 'red_coin': [50, 5000], 'green_coin': [50, 5000], 'xmark': [2000, 20000]}

# Maximum number of objects present in the competition
MAX_OBJ = {'coin': 2, 'xmark': 2, 'bin': 2}

# Color range for each object
COLOR_OBJ = {'cover': (20, 128, 128, 128), 'bin': (0, 0, 0, 0),
             'red_coin': (0, 40, 255, 255), 'green_coin': (60, 40, 0, 255),
             'red_xmark': (0, 60, 255, 255), 'green_xmark': (40, 219, 82, 255)}


''' Main objects to be detected for Robosub 2016 '''


def generic(img, data):
    mask = cv2.adaptiveThreshold(cv2.split(to_lab(img))[1], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 401, -20)
    return to_bgr(mask, 'gray'), [DetectedObject()]


def bin(img, data):
    thresh_limit = get_config(data)['salient_thresh']
    # mask = thresh_saturation_binary(img, thresh_limit) | thresh_salient(img, thresh_limit)
    # mask = cv2.bitwise_not(adaptive_thresh(cv2.split(img))[1])
    img = norm_illum_color(img, 0.5)
    mask = thresh_bin(img, thresh_limit)
    '''
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kern, iterations=1)
    mask = cv2.dilate(mask, kern, iterations=2)
    '''
    # return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_bin(img, mask, data)


def cover(img, data):
    thresh_limit = get_config(data)['salient_thresh']
    img = iace(finlaynorm(img))
    mask = thresh_salient(img, 40)
    return analyze_cover(img, mask, data)


def overall_bin(img, data):
    thresh_limit = get_config(data)['salient_thresh']
    img = norm_illum_color(img, 0.5)
    mask = thresh_bin(img, thresh_limit)
    return analyze_overall_bin(img, mask, data)


def vertical_coin(img, data):
    '''
    thresh_limit = get_config(data)['salient_thresh']
    mask = thresh_salient(img, 20)
    mask = adaptive_thresh(cv2.split(to_lab(img))[1])
    mask = adaptive_thresh(cv2.split(to_hsv(img))[2])
    mask = thresh_hsv(img, (60, 0, 100), (100, 255, 254))
    '''
    mask = thresh_red(img)
    # return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_vertical_coin(img, mask, data)


def horizontal_coin(img, data):
    '''
    thresh_limit = get_config(data)['salient_thresh']
    mask = thresh_salient(img, 50)
    mask = adaptive_thresh(cv2.split(to_hsv(img))[2])
    '''
    mask = thresh_green(img)
    # return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_vertical_coin(img, mask, data)


def red_coin(img, data):
    mask = thresh_red(img)
    # return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_vertical_coin(img, mask, data, 'red_coin', 1)


def green_coin(img, data):
    mask = thresh_green(img)
    # return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_vertical_coin(img, mask, data, 'green_coin', 1)


def tower(img, data):
    '''
    thresh_limit = get_config(data)['salient_thresh']
    mask = thresh_salient(img, thresh_limit)
    mask = adaptive_thresh(cv2.split(to_hsv(img))[2])
    '''
    mask = thresh_red(img) | thresh_green(img)
    # return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_tower(img, mask, data)


def xmark(img, data):
    '''
    mask = cv2.bitwise_not(thresh_table(img))
    thresh_limit = get_config(data)['salient_thresh']
    mask = thresh_salient(img, thresh_limit)
    mask = thresh_saturation(img)
    '''
    mask = thresh_table2(img)
    # return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_xmark(img, mask, data)


def red_x(img, data):
    mask = cv2.inRange(to_lab(img), (0, 140, 125), (255, 255, 255))
    return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_table(img, mask, data)


def green_x(img, data):
    mask = cv2.inRange(to_lab(img), (20, 0, 100), (255, 125, 255))
    return to_bgr(mask, 'gray'), [DetectedObject(detected=True)]
    return analyze_table(img, mask, data)


def table(img, data):
    mask = cv2.bitwise_not(thresh_table(img))
    return analyze_table(img, mask, data)


''' Sub objects '''


def black_bin(img, gamma=0.5, w_diff=3.5):
    """ Detect black square @feature.getBlack2
        gamma   < 1 turns img darker > 1 turns img lighter
        w_diff  weightage of difference between darkest and lightest pixel for thresholding
        returns thresholded image
    """
    # Normalizes illumination
    h, s, v = cv2.split(to_hsv(img))
    v = norm_illum_color(v, gamma)

    # Thresholding and morphological transformation
    max_pix = np.amax(v)
    min_pix = np.amin(v)
    diff = max_pix - min_pix
    thresh = min_pix + diff / w_diff
    mask = cv2.threshold(v, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    return mask

''' Shape validation '''


def is_circle(cnt, circle_limit=0.6):
    return get_circularity(cnt) >= circle_limit


def is_horizontal_coin(cnt, rectangularity_limit=0.5, ratio_limit=(1.5, 7)):
    """ Checks if rectangle with certain ratio """
    rect = cv2.minAreaRect(cnt)
    if rect[1][0] > 0 and rect[1][1] > 0:
        if rect[1][0] > rect[1][1]:
            asp_rat = rect[1][0] / float(rect[1][1])
        else:
            asp_rat = rect[1][1] / float(rect[1][0])
        rect_area = rect[1][0] * rect[1][1]
        rectanglularity = (cv2.contourArea(cnt) + 0.001) / rect_area
        return rectanglularity > rectangularity_limit and ratio_limit[0] < asp_rat < ratio_limit[1]
    return False


def get_rectangularity(cnt):
    return cv2.contourArea(cnt) / (get_rect_area(cnt) + 0.001)


def get_circularity(cnt):
    return cv2.contourArea(cnt) / (get_circle_area(cnt) + 0.001)


def is_rect(cnt, ratio_limit=0.6):
    """ Checks if rectangle with certain ratio """
    rect = cv2.minAreaRect(cnt)
    if rect[1][0] > 0 and rect[1][1] > 0:
        if rect[1][0] > rect[1][1]:
            asp_rat = rect[1][0] / float(rect[1][1])
        else:
            asp_rat = rect[1][1] / float(rect[1][0])
        return get_rectangularity(cnt) > ratio_limit and asp_rat < 3
    return False


def is_correct_area(area, name):
    return AREA[name][0] <= area <= AREA[name][1]

''' Contour processing '''


def analyze_horizontal_coin(img, mask, data):
    infos = []
    out_img = init_debug_img(img)
    # out_img = to_bgr(mask, 'gray')
    # Contour processing
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    cnts.sort(key=cv2.contourArea, reverse=True)
    if len(cnts) >= 1:
        for cnt in cnts:
            if is_correct_area(get_rect_area(cnt), 'horizontal_coin') and is_horizontal_coin(cnt):
                info = init_detected_object('horizontal_coin', data, img, cnt)
                infos.append(info)
                # Debug img
                draw_detected_object(out_img, info, cnt, len(infos))
            if len(infos) >= MAX_OBJ['coin']:
                break
    draw_text_info(out_img, format_info(infos))
    return out_img, infos


def analyze_vertical_coin(img, mask, data, name='vertical_coin', limit=MAX_OBJ['coin']):
    infos = []
    out_img = init_debug_img(img)
    out_img = to_bgr(mask, 'gray')
    # Contour processing
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    cnts.sort(key=cv2.contourArea, reverse=True)
    if len(cnts) >= 1:
        for cnt in cnts:
            if is_correct_area(get_circle_area(cnt), name) and is_circle(cnt):
                info = init_detected_object(name, data, img, cnt)
                infos.append(info)
                # Debug img
                draw_detected_object(out_img, info, cnt, len(infos))
            if len(infos) >= limit:
                break
    draw_text_info(out_img, format_info(infos))
    return out_img, infos


def analyze_tower(img, mask, data):
    """ Detects tower using horizontal coins """
    out_img, infos = analyze_horizontal_coin(img, mask, data)
    if infos:
        tower_info = init_detected_object('tower', data)
        tower_info.centroid = [np.mean([i.centroid[0] for i in infos]),
                               np.mean([i.centroid[1] for i in infos])]
        tower_info.offset = get_offset(tower_info.centroid)
        tower_info.detected = True
        tower_info.angle = infos[0].angle
        draw_centroid(out_img, tower_info.centroid)
        infos.insert(0, tower_info)
    return out_img, infos


def analyze_xmark(img, mask, data):
    infos = []
    contours = []
    # Contour processing
    out_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out_img = init_debug_img(img)
    cnts, hierrs = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1:]
    cnts.sort(key=cv2.contourArea)
    if len(cnts) >= 1:
        for cnt, hierr in zip(cnts, hierrs[0]):
            if is_correct_area(get_circle_area(cnt), 'xmark') and is_circle(cnt):
                info = init_detected_object('xmark', data, img, cnt)
                if not point_in_contours(info.centroid, contours):
                    infos.append(info)
                    contours.append(cnt)
                    # Debug img
                    draw_detected_object(out_img, info, cnt, len(infos))
            if len(infos) >= MAX_OBJ['xmark']:
                break
    draw_text_info(out_img, format_info(infos))
    return out_img, infos


def rank_xmark(cnt):
    pass


def analyze_table(img, mask, data):
    """ Detects table using xmarks """
    out_img, infos = analyze_xmark(img, mask, data)
    if infos:
        table_info = init_detected_object('table', data)
        table_info.centroid = [np.mean([i.centroid[0] for i in infos]),
                               np.mean([i.centroid[1] for i in infos])]
        table_info.offset = get_offset(table_info.centroid)
        table_info.move = get_real_distance(Config.bot_center, table_info.centroid, data['depth'],
                                            'table')
        table_info.detected = True
        draw_centroid(out_img, table_info.centroid)
        infos.insert(0, table_info)
    return out_img, infos


def analyze_overall_bin(img, mask, data):
    infos = []
    # Filter out  blue hue
    mask = mask & thresh_hsv(img, RED_GREEN[0], RED_GREEN[1])
    out_img = init_debug_img(img)
    # Contour processing
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts.sort(key=cv2.contourArea, reverse=True)
    if len(cnts) >= 1:
        for cnt in cnts:
            if is_correct_area(get_rect_area(cnt), 'bin') and is_rect(cnt):
                info = init_detected_object('overall_bin', data, img, cnt)
                infos.append(info)
                draw_detected_object(out_img, info, cnt, len(infos))
            if len(infos) >= MAX_OBJ['bin']:
                break

    if infos:
        infos[0].centroid = [np.mean([i.centroid[0] for i in infos]),
                             np.mean([i.centroid[1] for i in infos])]
        infos[0].offset = get_offset(infos[0].centroid)
        draw_centroid(out_img, infos[0].centroid)
    draw_text_info(out_img, format_info(infos))
    return out_img, infos


def analyze_cover(img, mask, data):
    infos = []
    # Filter out  blue hue
    mask = mask & thresh_hsv(img, RED_GREEN[0], RED_GREEN[1])
    out_img = init_debug_img(img)
    # out_img = to_bgr(mask, 'gray')
    # Contour processing
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts.sort(key=cv2.contourArea, reverse=True)
    if len(cnts) >= 1:
        for cnt in cnts:
            if is_correct_area(get_rect_area(cnt), 'cover') and is_rect(cnt):
                info = init_detected_object('cover', data, img, cnt)
                # if info.predicted_color == 'red':
                infos.append(info)
                # Debug img
                draw_detected_object(out_img, info, cnt, len(infos))
                break
    draw_text_info(out_img, format_info(infos))
    return out_img, infos


def analyze_bin(img, mask, data):
    infos = []
    # Filter out  blue hue
    # mask = mask & thresh_hsv(img, RED_GREEN[0], RED_GREEN[1])
    out_img = init_debug_img(img)
    # out_img = to_bgr(mask, 'gray')
    # Contour processing
    cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts.sort(key=cv2.contourArea)
    if len(cnts) >= 1:
        for cnt in cnts:
            if is_correct_area(get_rect_area(cnt), 'bin') and is_rect(cnt, 0.7):
                info = init_detected_object('bin', data, img, cnt)
                if info.color[0] >= 30:
                    infos.append(info)
                    # Debug img
                    draw_detected_object(out_img, info, cnt, len(infos))
                    break
    draw_text_info(out_img, format_info(infos))
    return out_img, infos


def analyze_rect(img, mask, data, name):
    infos = []
    # Filter out  blue hue
    mask = mask & thresh_hsv(img, RED_GREEN[0], RED_GREEN[1])
    out_img = init_debug_img(img)
    out_img = to_bgr(mask, 'gray')
    # Contour processing
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts.sort(key=cv2.contourArea, reverse=True)
    if len(cnts) >= 1:
        for cnt in cnts:
            if is_correct_area(get_rect_area(cnt), name) and is_rect(cnt):
                info = init_detected_object(name, data, img, cnt)
                infos.append(info)
                # Debug img
                draw_detected_object(out_img, info, cnt, len(infos))
    draw_text_info(out_img, format_info(infos))
    return out_img, infos


def get_norm_area(area):
    """ Returns area normalized between [0 ... 1] """
    return area / ((Config.processed_img_size[0] * Config.processed_img_size[1]) + 0.001)


def get_longer_edge(cnt):
    rect = cv2.minAreaRect(cnt)
    edge1 = rect[1][0]
    edge2 = rect[1][1]
    return edge1 if edge1 > edge2 else edge2


def get_rect_area(cnt):
    rect = cv2.minAreaRect(cnt)
    return int(rect[1][0] * rect[1][1])


def get_centroid(cnt):
    mom = cv2.moments(cnt)
    centroid_x = int((mom['m10'] + 0.0001) / (mom['m00'] + 0.0001))
    centroid_y = int((mom['m01'] + 0.0001) / (mom['m00'] + 0.0001))
    return (centroid_x, centroid_y)


def get_angle_long(cnt):
    """ Returns angle perpendicular to long side of a rectangle """
    rect = cv2.minAreaRect(cnt)
    points = np.int32(cv2.boxPoints(rect))
    edge1 = points[1] - points[0]
    edge2 = points[2] - points[1]
    if cv2.norm(edge1) > cv2.norm(edge2):
        return math.degrees(math.atan2(edge1[1], edge1[0]))
    else:
        return math.degrees(math.atan2(edge2[1], edge2[0]))


def get_vehicle_angle(rect_angle):
    """ Returns angle needed to be turned to align perpendicularly to long side of object """
    angle = 90 - abs(rect_angle) if rect_angle >= -90 else 90 - abs(rect_angle)
    return (angle + 90) % 360


def get_offset(centroid):
    x = Config.processed_img_size[0]
    y = Config.processed_img_size[1]
    dx = (centroid[0] - (x / 2)) / float(x)
    dy = ((y / 2) - centroid[1]) / float(y)
    return dx, dy


def get_mean_centroids(centroids):
    return [np.mean(centroid) for centroid in zip(*centroids)]


def get_group_contours(contours):
    hull = cv2.convexHull(np.vstack(contours))
    return hull


def point_in_contour(point, cnt):
    """ Checks if a point is located inside given contour """
    return cv2.pointPolygonTest(cnt, (int(point[0]), int(point[1])), False) > 0


def point_in_contours(point, contours):
    for cnt in contours:
        if point_in_contour(point, cnt):
            return True
    return False


def get_solidity(cnt):
    """ Calculates solidity according by area(cnt) / area(convexHull(cnt) """
    return cv2.contourArea(cnt) / cv2.contourArea(cv2.convexHull(cnt))

''' Drawing '''


def draw_square(canvas, center, color=c.WHITE, offset=8, thickness=2):
    """Draws a square
    Args:
        offset: distance from center which determines size of square
    """
    top_left = (center[0] - offset, center[1] + offset)
    bot_right = (center[0] + offset, center[1] - offset)
    cv2.rectangle(canvas, top_left, bot_right, color, thickness)


def draw_centroid(out_img, centroid, rad=4, color=c.PURPLE):
    cv2.circle(out_img, (int(centroid[0]), int(centroid[1])), rad, color, -1)


def draw_angle(cnt, centroid, out_img):
    rect = cv2.minAreaRect(cnt)
    startpt = (int(centroid[0]), int(centroid[1]))
    gradient = np.deg2rad(get_angle_long(cnt))
    endpt = (int(startpt[0] + 200 * math.cos(gradient)),
             int(startpt[1] + 200 * math.sin(gradient)))
    startpt = (int(startpt[0]), int(startpt[1]))
    cv2.line(out_img, startpt, endpt, (0, 255, 0), 2)
    cv2.drawContours(out_img, [np.int0(cv2.boxPoints(rect))], -1, (255, 0, 0), 2)


def format_info(infos):
    # Assuming that object with highest score is inserted first
    area = infos[0].area if infos else 0.0
    predicted_color = infos[0].predicted_color if infos else ""
    area = ("area: ", "{:.4f}".format(area))
    hue = ("color: ", predicted_color)
    detected = ("detected: ", str(len(infos)))
    return [area, hue, detected]


def draw_text_info(canvas, content):
    """ Draw on canvas list of (text, color) tuple in-order """
    start = (10, Config.processed_img_size[1] + 10)
    for label, text in content:
        draw_text(canvas, label, start, color=c.WHITE)
        draw_text(canvas, text, (start[0] + 70, start[1]), color=c.YELLOW)
        start = (start[0], start[1] + 15)


def draw_text(canvas, text, pos, font=cv2.FONT_HERSHEY_DUPLEX, size=0.4, color=c.WHITE,
              thickness=1):
    cv2.putText(canvas, text, pos, font, size, color, thickness, cv2.LINE_AA)


def init_debug_img(img):
    """ Create empty debug image for extra information """
    y, x = img.shape[:2]
    # Even out the size to store text information
    y = y + (x - y)
    canvas = np.zeros((y, x, 3), dtype=np.uint8)
    draw_square(canvas, Config.bot_center, offset=4, thickness=1)
    return canvas

''' Image statistics '''


def get_color_descriptor(img, cnt):
    blue, green, red = cv2.split(img)
    h, s, v = cv2.split(to_hsv(img))
    l, a, b = cv2.split(to_lab(img))
    h_mean = get_avg_intensity(h, cnt)
    l_mean = get_avg_intensity(l, cnt)
    a_mean = get_avg_intensity(a, cnt)
    b_mean = get_avg_intensity(b, cnt)
    return [h_mean, l_mean, a_mean, b_mean]


def get_predicted_color(color_desc):
    if abs(COLOR_OBJ['red_coin'][2] - color_desc[2]) > abs(COLOR_OBJ['green_coin'][2] - color_desc[2]):
        return 'green'
    else:
        return 'red'


def get_avg_intensity(channel, cnt):
    """ Get average intensity for a grayscale channel or colored image from ROI given by a contour """
    mask = np.zeros(channel.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    return cv2.mean(channel, mask=mask)[0]


def mean_value(img):
    """ Return average intensity of value channel """
    h, s, v = cv2.split(to_hsv(img))
    return np.mean(v)


def get_circle_area(cnt):
    rad = cv2.minEnclosingCircle(cnt)[1]
    return math.pi * rad * rad


def get_real_distance(origin, target, depth, name):
    """ Given pixel location on screen, returns distance (in meter) to target pixel location """
    distance = Config.depth[name] - depth
    dx = distance * math.sin((target[0] - origin[0]) * Config.bot_angular_resolution[0])
    dy = distance * math.sin((origin[1] - target[1]) * Config.bot_angular_resolution[1])
    return [dx, dy]


def get_expected_rect_area(depth, name):
    """ Only works for bottom camera with depth sensor giving range """
    distance = Config.depth[name] - depth
    original_length = Config.length[name]
    perceived = (original_length * Config.bot_focal_length[0] /
                 (distance * Config.bot_resize_factor))
    return perceived**2 / 2.0


def get_expected_circle_area(depth, name):
    """ Only works for bottom camera with depth sensor giving range """
    distance = Config.depth[name] - depth
    original_radius = Config.length[name]
    perceived = (original_radius * Config.bot_focal_length[0] /
                 (distance * Config.bot_resize_factor))
    return perceived**2 * math.pi


def is_blue(img, cnt):
    return get_avg_intensity(img, cnt) > 60


''' Color thresholding '''


def thresh_black(img):
    l, a, b = cv2.split(to_lab(img))
    return adaptive_thresh(l)


def thresh_orange(img):
    processed = finlaynorm(img)
    b, g, r = cv2.split(processed)
    # hsv thresholding
    loThresh1 = (0, 0, 0)
    hiThresh1 = (40, 255, 235)
    loThresh2 = (170, 0, 0)
    hiThresh2 = (180, 255, 235)
    mask = cv2.inRange(to_hsv(processed), loThresh1, hiThresh1)
    mask2 = cv2.inRange(to_hsv(processed), loThresh2, hiThresh2)
    orange = mask | mask2
    # grayscale thresholding using green channel
    min = np.min(g)
    max = np.max(g)
    diff = max - min
    thresh = min + diff / 1.8
    common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    # final = common & orange
    # final = cv2.erode(final, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # final = cv2.dilate(final, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return common


def thresh_yellow(img):
    processed = finlaynorm(iace(img))
    # hsv thresholding
    b, g, r = cv2.split(processed)
    h, s, v = cv2.split(to_hsv(processed))
    loThresh = (15, 0, 150)
    hiThresh = (50, 255, 255)
    yellow = cv2.inRange(to_hsv(processed), loThresh, hiThresh)
    # grayscale thresholding using blue channel
    min = np.amin(b)
    max = np.amax(b)
    diff = max - min
    thresh = min + diff / 1.2
    common = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    return common


def thresh_outline(img, w_diff=1.5):
    h, s, v = cv2.split(to_hsv(img))
    min = np.amin(v)
    max = np.amax(v)
    thresh = min + (max - min) / w_diff
    mask = np.uint8(cv2.Canny(v, thresh / 2, thresh))
    return mask


def thresh_green(img):
    mask = cv2.inRange(to_lab(img), (20, 0, 100), (255, 125, 255))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kern, iterations=2)
    mask = cv2.dilate(mask, kern, iterations=1)
    return mask


def thresh_red(img):
    mask = cv2.inRange(to_lab(img), (0, 140, 125), (255, 255, 255))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kern, iterations=2)
    mask = cv2.dilate(mask, kern, iterations=1)
    return mask


def thresh_hsv(img, lo_thresh, hi_thresh, mode='hsv'):
    if mode is 'hsv':
        return cv2.inRange(to_hsv(img), lo_thresh, hi_thresh)
    if mode is 'lab':
        return cv2.inRange(to_lab(img), lo_thresh, hi_thresh)


def thresh_bin(img, thresh_limit=60):
    """ Threshold using blue channel """
    b, g, r = cv2.split(img)
    # mask = get_salient(r)
    mask = cv2.threshold(b, 50, 255, cv2.THRESH_BINARY_INV)[1]
    return mask


def thresh_salient(img, thresh_limit=60):
    """ Threshold using blue channel """
    b, g, r = cv2.split(img)
    mask = get_salient(b)
    mask = cv2.threshold(mask, thresh_limit, 255, cv2.THRESH_BINARY)[1]
    return mask


def thresh_saturation_binary(img, thresh_limit=60):
    h, s, v = cv2.split(to_hsv(img))
    mask = cv2.threshold(s, thresh_limit, 255, cv2.THRESH_BINARY)[1]
    return mask


def thresh_saturation(img):
    h, s, v = cv2.split(to_hsv(img))
    return adaptive_thresh(s)


def thresh_table2(img):
    l, a, b = cv2.split(to_lab(img))
    return cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -10)


def thresh_table(img, offset=2):
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(to_hsv(img))
    return adaptive_thresh(s)


def adaptive_thresh(gray, block=31, offset=5):
    res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                block, offset)
    return res

""" Segmentation """


def superpixel_slic(img, n_segments=100, sigma=10, compactness=10, max_iter=3):
    """ Generates superpixels based on SLIC
        n_segments    number of superpixels wished to be generated
        sigma         smoothing used prior to segmentation
        compactness   balance between color space proximity and image space proximity.
                      higher more weight to space
        max_iter      iterations for kmeans
        return        segmented image, superpixels """
    image = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    segments = slic(image, n_segments=n_segments, sigma=sigma, compactness=compactness,
                    max_iter=max_iter)
    return segments


def get_superpixels_contours(img, segments):
    contours = []
    for (i, segVal) in enumerate(np.unique(segments)):
        # construct a mask for the segment
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        if len(cnts) >= 1:
            contours.append(cnts[0])
    return contours


def show_superpixels(img, segments):
    """ Given result of SLIC, mark boundaries between regions """
    img = mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), segments)
    return cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR)


def kmeans(img):
    Z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


""" Dealing with ROS messages and services """


def draw_detected_object(out_img, info, cnt, id):
    draw_centroid(out_img, info.centroid, 2, c.YELLOW)
    cv2.drawContours(out_img, [cnt], -1, c.PURPLE, 1)
    cnt_color = Color.hue_to_bgr(info.color[0])
    '''
    draw_text(out_img, "H:{:.1f}".format(info.color[0]), (info.centroid[0] + 20, info.centroid[1] + 10))
    draw_text(out_img, "L:{:.1f}".format(info.color[1]), (info.centroid[0] + 75, info.centroid[1] + 10))
    draw_text(out_img, "A:{:.1f}".format(info.color[2]), (info.centroid[0] + 20, info.centroid[1] + 25))
    draw_text(out_img, "B:{:.1f}".format(info.color[3]), (info.centroid[0] + 75, info.centroid[1] + 25))
    draw_text(out_img, "Area:{:.1f}".format(info.area), (info.centroid[0] + 20, info.centroid[1] + 40))
    draw_text(out_img, "Color:{}".format(info.predicted_color), (info.centroid[0] + 20, info.centroid[1] + 55))
    draw_text(out_img, "Dist:{:.1f}".format(info.distance), (info.centroid[0] + 20, info.centroid[1] + 70))
    '''
    if info.name not in ['xmark', 'vertical_coin']:
        cv2.drawContours(out_img, [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))], -1, cnt_color, 2)
    else:
        centroid, radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(out_img, (int(centroid[0]), int(centroid[1])), int(radius), cnt_color, 2)


def init_detected_object(name, data, img=[], cnt=None):
    info = DetectedObject()
    info.name = name
    if len(img) > 0:
        info.centroid = get_centroid(cnt)
        info.offset = get_offset(info.centroid)
        info.move = get_real_distance(Config.bot_center, info.centroid, data['depth'], name)
        info.color = get_color_descriptor(img, cnt)
        info.predicted_color = get_predicted_color(info.color)
        info.angle = get_vehicle_angle(get_angle_long(cnt))
        info.rectangularity = get_rectangularity(cnt)
        info.circularity = get_circularity(cnt)
        if info.name not in ['xmark', 'vertical_coin']:
            info.area = get_rect_area(cnt)
            info.distance = rect_to_distance(get_longer_edge(cnt), Config.length[name])
        else:
            radius = cv2.minEnclosingCircle(cnt)[1]
            info.area = math.pi * (radius**2)
            info.distance = circle_to_distance(radius, Config.length[name])
        info.detected = True
    return info


def get_config(data, obj_id='generic'):
    return data['config'][obj_id]
