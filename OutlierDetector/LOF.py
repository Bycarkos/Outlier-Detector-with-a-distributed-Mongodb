# -*- coding: utf-8 -*-


from pyod.models.lof import LOF as ODLOF

try:
  from .OutlierDetector import OutlierDetector as od
except ImportError:
  from OutlierDetector import OutlierDetector as od

from scipy.sparse import csgraph
from scipy.sparse.linalg import svds
#from sklearn.cluster import KMeans
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd


class LOF(od, ODLOF):
    def __init__(self, **kwargs):
        od.__init__(self, **kwargs)
        ODLOF.__init__(self, **kwargs)

    def detector(self, FeaturesList, y_train=None , **params ):


        x_train = [ x.features for  x in FeaturesList[0]]
        self.fit(x_train)

        return self.predict(x_train)
