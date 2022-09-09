# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: oriolrt
'''

import os
from collections import namedtuple

import numpy as np


class noConnection:
    """
    This class emulates a ficticious connection to a DBMS it provides the basic interface.

    """

    __slots__ = ['__cfg', '__conn', '__bd', '__isStarted', '__server']

    def __init__(self, cfg=""):
        '''
            Constructor
            '''

        self.__cfg = cfg
        self.__conn = 0
        self.__bd = None
        self.__isStarted = False
        self.__server = None

    @property
    def cfg(self):
        return self.__cfg

    @cfg.setter
    def cfg(self, valor):
        self.__cfg = valor

    @property
    def bd(self):
        return self.__bd

    @bd.setter
    def bd(self, nameBD):
        self.__bd = nameBD

    @property
    def conn(self):
        return self.__conn

    @conn.setter
    def conn(self, valor):
        self.__conn = valor

    @property
    def isStarted(self):
        return self.__isStarted

    @isStarted.setter
    def isStarted(self, valor):
        self.__isStarted = valor

    @property
    def server(self):
        return self.__server

    @server.setter
    def server(self, valor):
        self.__server = valor

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    @property
    def connectDB(self):
        """
          Connect to a DBMS server given the connexion information saved on the cfg member variable.

          :return: None
        """

        print("""Ara ens estariem conectant al servidor...""")

        return 0

    def close(self):
        self.__isStarted = False

    def exists(self, dataset):
        """
        :param dataset: name of the dataset
        :return: True if all the feature vectors of the dataset are already inserted in the DB
        """

        return False
