#!/usr/bin/python2.7
""" Useful ROS utility functions mostly derived from ROS core stack """
import rospkg
import yaml
import os
from optparse import OptionParser


def create_ros_options(name, argv):
    """ Create options for ros node e.g rosrun pkg node options...
    @return:    parser, args
    """
    args = argv[1:]
    parser = OptionParser(usage="usage: %prog [options...]", prog=name)
    return (parser, args)


def load_yaml_options(args, options):
    try:
        import yaml
    except ImportError:
        raise Exception("Cannot import yaml. Please make sure the pyyaml system dependency is installed")
    node_args = yaml.load(args[0])
    for k, v in node_args:
        if k in options:
            options[k] = v
    return options


def get_config(name, category='task'):
    """ Returns configuration file located in vision package """
    rospack = rospkg.RosPack()
    filepath = os.path.join(rospack.get_path('vision'), 'config', category, name)
    with open(filepath, 'r') as stream:
        return yaml.load(stream)
