#!/usr/bin/env python
"""Functions to annotate image and add visual information
to image
"""
import cv2
import numpy as np
import math

'''Color codes'''
BLUE = (255, 128, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (127, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

CENTERCAM_COLOR = WHITE  # color of square at center of cam


def draw_square(canvas, center, color=CENTERCAM_COLOR, offset=8, thickness=2):
    """Draws a square
    Args:
        offset: distance from center which determines size of square
    """
    top_left = (center[0] - offset, center[1] + offset)
    bot_right = (center[0] + offset, center[1] - offset)
    cv2.rectangle(canvas, top_left, bot_right, color, thickness)


def draw_circle(canvas, center, color, rad=4, thickness=2):
    cv2.circle(canvas, center, rad, color, thickness)


def draw_text(canvas, text, bot_left, color, size=0.6,
              fontface=cv2.FONT_HERSHEY_SIMPLEX):
    """Draws a text
    Args:
        bot_left: position of bottom-leftmost text on canvas
    """
    cv2.putText(canvas, text, bot_left, fontface, size, color)


def ypos(starty, inc):
    while True:
        yield starty
        starty += inc


def draw_debugtext(canvas, output):
    startx, starty = 20, ypos(30, 25)  # initializes starting coordinate of text column
    draw_text(canvas, "dx: {}, dy: {}".format(output.dx, output.dy),
              (startx, next(starty)), BLUE)
    draw_text(canvas, "area_ratio: {}".format(output.area_ratio),
              (startx, next(starty)), BLUE)


def draw_cross(canvas, center, color=RED, size=(5, 5), thickness=2):
    top_left = (center[0] - size[0], center[1] + size[1])
    top_right = (center[0] + size[0], center[1] + size[1])
    bot_left = (center[0] - size[0], center[1] - size[1])
    bot_right = (center[0] + size[0], center[1] - size[1])
    cv2.line(canvas, bot_left, top_right, color, thickness)
    cv2.line(canvas, bot_right, top_left, color, thickness)


def draw_arrow(canvas, centroid, angle, color=PINK, scalar=200):
    """Draws arrow from centroid extends with a scalar"""
    gradient = np.deg2rad(angle)
    startpt = (int(centroid[0]), int(centroid[1]))
    endpt = (startpt + scalar * math.cos(gradient), startpt[1] + scalar * math.sin(gradient))
    cv2.line(canvas, startpt, endpt, GREEN, 2)


def create_debugimg(output):
    """Generates image for debugging
    Args:
        output: Output object that stores details of detected object
    Returns:
        outimg: output image with added visual information
    """
    canvas = np.zeros_like(output.outimg)
    draw_square(canvas, output.center, offset=10)  # draws center of screen
    if output.detected:
        draw_cross(canvas, output.centroid)  # draws center of detected object
        draw_debugtext(canvas, output)
    return canvas
