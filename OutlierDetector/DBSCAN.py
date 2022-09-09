# -*- coding: utf-8 -*-


from sklearn.mixture import GaussianMixture

try:
  from .OutlierDetector import OutlierDetector as od
except ImportError:
  from OutlierDetector import OutlierDetector as od

import numpy as np
from sklearn.cluster import DBSCAN as ODDB
from numpy.linalg import norm


class DBSCAN(od, ODDB):
    def __init__(self, **kwargs):
        od.__init__(self, **kwargs)
        ODDB.__init__(self, **kwargs)

    def detector(self, FeaturesList, y_train=None , **params ):


        x_train = [ x for  x in FeaturesList[0]]

        #x_train = x_train / np.max(norm(x_train, 2, axis=1))

        self.eps = 0.5*np.min(norm(x_train, 2, axis=1))

        #x_train = x_train/


        return np.array((self.fit_predict(x_train) == -1)).astype(int)
