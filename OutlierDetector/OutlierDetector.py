'''
Created on Jun 12, 2018

@author: oriolrt
'''

#from scipy.sparse import csgraph
#from scipy.sparse.linalg import svds
#from sklearn.cluster import KMeans

#import matplotlib.pyplot as plt
#import networkx as nx
import numpy as np
#import scipy.sparse as sp

import random
import struct
from collections import namedtuple


class OutlierDetector(object):
    '''
    It implements a base class for outlier detectors
    '''


    def __init__(self, numViews=2,*args,**kargs):
        '''
        Constructor
        '''
        
        self.numViews = 2
        self.eps = 10**-6
        self.rho = 1.2
        self.mu_max = 10**6
        self.mu=10**-1
        self.maxIter = 200
        self.setViews(numViews)


    def setViews(self, numViews):
        self.numViews = numViews
        self.layers = ['Visual']*numViews

    def __generateAttrOutlier(self, data, outliersGTIdx, ratio=0.05):


        numViews = len(data)

        numSamples = len(data[0])
        numFeatures = len(data[0][0].features)
        # newFeatures = [[]]*numViews

        oid = list(set(range(numSamples)).difference(outliersGTIdx))
        # random.shuffle(id)

        if isinstance(ratio, list):
            mostres = ratio
        else:
            N = int(round(numSamples * ratio))
            mostres = random.sample(oid, N)

        fila = namedtuple("fila", "id features")
        attrOut = mostres
        rfeatures = np.random.rand(len(mostres), numFeatures, numViews)
        for idx, x in enumerate(mostres):
            # attrOut.append(x)
            for i in range(numViews):
                id = data[i][x].id
                data[i][x] = fila(id=id, features=rfeatures[idx, :, i].tolist())

        return data,outliersGTIdx + attrOut


    def __generateClassOutlier(self,data, IdsList, ratio=0.05):

        numViews = len(data)

        len_class = [len(IdsList[x]) for x in IdsList]

        total = sum(len_class)

        if isinstance(ratio, list):
            numOutliers = len(ratio)
            rid = np.array(ratio)


        else:
          numOutliers = int(round(total * ratio / 2.0))

          idClasses = [x[0] for x in sorted(IdsList.items()) if len(x[1]) > numOutliers]

          rid = np.zeros((numOutliers, len(idClasses)))

          for i, l in enumerate(idClasses):
            rid[:, i] = random.sample(IdsList[l], numOutliers)

          rid = rid.astype(int)

        fila = namedtuple("fila", "id features")

        outliersGTIdx = []
        for i in range(numOutliers):
          currentView = 0  ##random.randint(0,numViews-1)
          idSamples = random.sample(range(len(idClasses)), 2)
          features = data[currentView][rid[i, idSamples[0]]].features
          data[currentView][rid[i, idSamples[0]]] = fila(id=data[currentView][rid[i, idSamples[0]]].id,
                                                         features=data[currentView][rid[i, idSamples[1]]].features)
          data[currentView][rid[i, idSamples[1]]] = fila(id=data[currentView][rid[i, idSamples[1]]].id,
                                                         features=features)
          outliersGTIdx = outliersGTIdx + rid[i, idSamples].tolist()

        outliersGTIdx.sort()

        # y = np.zeros(total)
        # y[outliersGTIdx] = 1

        return data, outliersGTIdx


    def prepareExperimentData(self,dbms, conf, nameDataset, dataInfo, repeticio, settings = {'numViews': 2} ):
        """
        Generates the outlier data given the settings for outlier Detection methods.

        :param features: list of feature vectors used to geneta
        :param classIds:
        :param settings:
        :return:
        """


        class_outlier =  float(conf[0] / 100.0)
        attr_outlier =  float(conf[1] / 100.0)
        if "numViews" in settings:
            numViews = settings["numViews"]
        else:
            numViews = 2


        if len(dataInfo.features) == 1: dataInfo.features= dataInfo.features[0]
        numDims = len(dataInfo.features[0].features)
        numSamples = len(dataInfo.features)
        numFeatures = int(numDims / numViews)
        newFeatures = [[]]*numViews



        fila = namedtuple("fila", "id features")
        maxVal=[-1000000]*numViews
        for x in dataInfo.features:
            id=x.id

            for y in range(numViews):
                # feats = x.features[y*numFeatures:(y+1)*numFeatures]
                feats = x.features
                newFeatures[y] = newFeatures[y] + [fila(id=id, features=feats)]


        outliers,generateOutliersFlag =  dbms.loadOutliers(nameDataset, repeticio, numSamples, conf, numViews, dataInfo )



        if generateOutliersFlag:
            newFeatures, outliersGTIdx = self.__generateClassOutlier(newFeatures,dataInfo["classIds"],ratio=class_outlier)

            newFeatures, outliersGTIdx = self.__generateAttrOutlier(newFeatures, outliersGTIdx, ratio=attr_outlier)

            #Salvem els outliers en la BD
            dbms.insertOutlierData(newFeatures,nameDataset, repeticio, outliersGTIdx, conf, dataInfo)

        else:
            outliersGTIdx = list(outliers.keys())

            # TODO: Aquest codi ja no seria necessari... (a esborrar un cop es validi)
            #separem els vectors en les vistes
            for oid in outliers:
                #sprint(oid)
                for y in range(numViews):
                    # newFeatures[y][oid] = fila(id=oid, features=outliers[oid][y*numFeatures:(y+1)*numFeatures])
                    newFeatures[y][oid] = fila(id=oid, features=outliers[oid][y])

        y = np.zeros(numSamples)
        y[outliersGTIdx] = 1

        return newFeatures, y, outliersGTIdx
