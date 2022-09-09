# -*- coding: utf-8 -*-

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


class APS(od):
    u'''

    This class implements the APS method that was introduced in [1]


    [1]	C. Wang, Z. Liu, H. Gao, Y. Fu, "Applying Anomaly Pattern Score for Outlier Detection" in IEEE Access, Vol 7, pp. 16008-16019, 2019.

    Created on 10/10/2019

    @author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
    @Institution: Computer Vision Center - Universitat Autonoma de Barcelona
    '''
    __slots__ = {'__m', '__similarity', '__k', '__numSamples', '__alpha'}


    def __init__(self,m=10, *args, **kwargs):
        '''
        Constructor
        '''

        self.__m = m

        #Default parameters
        defaultParams = {'m': 50, 'beta': 50, 'alpha': 0}

        for k in defaultParams.keys():
            if k in kwargs:
                self.__setattr__('_APS__'+k, kwargs[k])
            else:
                self.__setattr__('_APS__'+k, defaultParams[k])

        self.k = self.__k_alpha( self.alpha )

        super(APS, self).__init__(numViews=1)

    @property
    def m(self):
        return self._APS__m

    @m.setter
    def m(self, valor ):
        self._APS__m = valor


    @property
    def numSamples(self):
        return self._APS__numSamples

    @numSamples.setter
    def numSamples(self, valor ):
        self._APS__numSamples = valor



    @property
    def beta(self):
        return self._APS__beta

    @beta.setter
    def beta(self, valor):
        self._APS__beta = valor

    @property
    def alpha(self):
        return self._APS__alpha

    @alpha.setter
    def beta(self, valor):
        self._APS__alpha = valor


    @property
    def k(self):
        return self._APS__k

    @k.setter
    def k(self, valor):
        self._APS__k = valor

    def __k_alpha(self, alpha):

        if alpha==0:
            self.k = 2

        if alpha > 0:
            self.k = self.k + int(self.numSamples/self.m)


    def __computeD(self, featureMatrix ):

        numSamples = featureMatrix.shape[0]
        k = self.k

        D = dist.squareform(dist.pdist(featureMatrix))

        idx = np.argsort(D, axis=1)[:, 0:k + 1]

        A = np.array([(i, idx[i,j],1/(1+D[i,j])) for i in range(numSamples) for j in range(k)])

        return A



    def __computeA(self, FeatureMatrix, alpha ):

        self.__k_alpha(alpha)
        k = self.k
        numSamples = FeatureMatrix.shape[0]

        A = self.__computeD( FeatureMatrix )

        sA = sp.coo_matrix((A[:, 2], (A[:, 0], A[:, 1])), shape=(numSamples, numSamples))

        return sA


    def __computeT(selfs, adj):
        # D = adj.toarray()[np.array(range(adj.shape[0])), np.array(range(adj.shape[1]))]

        a = np.array(adj.sum(axis=0)).flatten()

        return adj*sp.diags(list(1 / (a + [1 if x else 0 for x in a == 0])))


    def __score(self, D):
        return D.sum(axis=0)/D.sum()



    def asp(self, FeatureMatrix, beta=.15, m=50):


        N = FeatureMatrix.shape[0]
        D = np.zeros((m,N))
        for alpha in range(m):
            A = self.__computeA(FeatureMatrix, alpha)
            T = self.__computeT(A)
            mu = eta = np.array([1]*N)/N
            converged = False
            while not converged:
                p = beta * eta + (1-beta)*T*mu
                mu_new = p/np.linalg.norm(p)
                if np.linalg.norm( mu_new - mu) < 0.1:
                    converged = True
                mu = mu_new

            D[alpha,:] = p-np.min(p)


        return self._APS__score(D)




    def detector(self, FeaturesList, y_train=None , **params ):

        FeatureMatrix = np.matrix([x.features for x in FeaturesList[0]])

        self.numSamples = FeatureMatrix.shape[0]


        return self.asp(FeatureMatrix, beta=self.beta, m=self.m)





