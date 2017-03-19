#!/usr/bin/env python
""" Stores default configuration and mappings """

OBJ_ID = {'red_buoy': 1, 'green_buoy': 2, 'yellow_buoy': 3,
          'lane': 4, 'pole': 5, 'bin': 6, 'bin_cover': 7,
          'torpedo_board': 8, 'torpedo_cover': 9, 'torpedo_hole': 10,
          'red_coin': 11, 'green_coin': 12, 'red_x': 13, 'green_x': 14,
          'table': 15}

PREFIX = './examples/dataset/robosub16/{}'
TEST_DATA = {'blur': [PREFIX.format('buoy/9')],
             'dark': [PREFIX.format('tower/9'), PREFIX.format('torpedo/1')],
             'bright': [PREFIX.format('torpedo/4'), PREFIX.format('torpedo/1')],
             'flicker': [PREFIX.format('table/7'), PREFIX.format('bin/8')]}


