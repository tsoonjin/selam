#!/usr/bin/env python
import os
import csv


def get_basename(name):
    path = os.path.abspath(name)
    return os.path.basename(os.path.normpath(path))


def generate_csv(out_path, ls):
    """ Parse detected objects to csv file """
    with open(out_path, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(ls)
