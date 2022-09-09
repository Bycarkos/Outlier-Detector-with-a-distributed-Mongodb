# -*- coding: utf-8 -*-


from pyod.models.knn import KNN as ODSOGAAL

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


class SOGAAL(od, ODSOGAAL):
    def __init__(self, **kwargs):
        od.__init__(self, **kwargs)
        ODSOGAAL.__init__(self)



    def detector(self, FeaturesList, y_train=None , **params ):


        x_train = [ x.features for  x in FeaturesList[0]]
        self.fit(x_train)

        return self.predict(x_train)
