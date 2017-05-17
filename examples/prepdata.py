#!/usr/bin/env python
""" Generate training and validation data from ground truth """
import sys

from selam import prepdata
from selam.utils import io

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python extract_region.py <dataset directory> <annotation file> <prefix> \n")
        exit()
    positives, negatives = prepdata.extract_training(sys.argv[1], sys.argv[2])
    io.write_to_dir(positives, "/home/parapa/github/selam/test/training/bin/positives")
    io.write_to_dir(negatives, "/home/parapa/github/selam/test/training/bin/negatives")
