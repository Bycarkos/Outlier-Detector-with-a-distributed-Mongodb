# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import argparse

__author__ = 'Oriol Ramos Terrades'
__email__ = 'oriolrt@cvc.uab.cat'

from argparse import HelpFormatter
from operator import attrgetter


class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


class Options(argparse.ArgumentParser):

    def __init__(self):
        # MODEL SETTINGS
        super().__init__(
            description="This script inserts data into a scheme previously created for outlier detection experiments",
            formatter_class=SortingHelpFormatter)
        # Positional arguments
        super().add_argument('-c', '--config', type=str, help='JSON file with the information required to insert data')
        super().add_argument('-N', '--datasetName', type=str, help='Name of the imported dataset')
        super().add_argument('-f', '--fileName', type=str, help='File where data is stored.')
        super().add_argument('-M', '--metadata', type=str, help='JSON file with meta data information.')
        super().add_argument('-t', '--dataType', type=self.check_type, default='vector',
                             help='Data set type, vector or image.')

    def parse(self):
        return super().parse_args()

    def check_type(self, value):
        if value.lower() not in ('vector', 'image'):
            raise argparse.ArgumentTypeError("%s is an invalid value. It must be 'vector' or 'image'" % value)
        return value


class OptionsTest(argparse.ArgumentParser):

    def __init__(self):
        # MODEL SETTINGS
        super().__init__(description="This script test outlier detection methods on different datasets",
                         formatter_class=SortingHelpFormatter)
        # Positional arguments
        super().add_argument('-c', '--config', type=str, help='JSON file with the information required to insert data')
        super().add_argument('-N', '--datasetName', type=str, help='Name of the imported dataset')
        super().add_argument('-f', '--featuresImage', type=str, default="{'cnn':'AlexNet', 'layer':'Visual'}",
                             help="extracted   features  from image dataset.")
        super().add_argument('-M', '--metadata', type=str, help='JSON file with meta data information.')
        super().add_argument('-l', '--numLayers', type=str, default='1', help='Data set type, vector or image.')
        super().add_argument("-m", "--method", type=str, default="DMOD",
                             help='coma-separated list with the outlier detection methods to test ')
        super().add_argument("-p", "--params", type=str,
                             help="string on JSON format with the method parameters and their values.")

    def parse(self):
        return super().parse_args()
