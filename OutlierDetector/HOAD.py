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


class HOAD(od):
    u'''

    This class implements the COD method that was introduced in [1]


    [1]	J. Gao, N. Du, W. Fan, D. Turaga, S. Parthasarathy, J. Han, "A multi-graph spectral framework for mining multi-source anomalies" in in Graph Embedding for Pattern Analysis, New York, NY, USA:Springer, pp. 205-228, 2013.

    Created on 1/9/2018

    @author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
    @Institution: Computer Vision Center - Universitat Autonoma de Barcelona
    '''
    __slots__ = '__mutual_k', '__similarity', '__k', '__maxSamples'


    def __init__(self,mutual_k=10,numViews=2,**kwargs):
        '''
        Constructor
        '''

        self.__mutual_k= mutual_k

        #Default parameters
        defaultParams = {'sigma': 1, 'similarity': 'invDist', 'k': 2, 'maxSamples': 20000}

        for k in defaultParams.keys():
            if k in kwargs:
                self.__setattr__('_HOAD__'+k, kwargs[k])
            else:
                self.__setattr__('_HOAD__'+k, defaultParams[k])




        super(HOAD, self).__init__(numViews)


    @property
    def mutual_k(self):
        return self.__mutual_k

    @mutual_k.setter
    def mutual_k(self, mutual_k):
        self.__mutual_k = mutual_k

    @property
    def similarity(self):
        return self.__similarity

    @similarity.setter
    def similarity(self, valor):
        if valor.lower() in ['invDist', 'euclidean', 'gaussian', 'jaccard']:
            self.__similarity = valor
        else:
            self.__similarity = 'invDist'

    def computeFullyConnected(self,taula, similarity=None, sigma=1, T=1.e-6):

        if similarity is None:
            similarity = self.similarity

        # Distance threshold
        L = np.sqrt(-2 * np.log(T))
        # taula.show(10)
        numSamples = len(taula)
        numFeatures = len(taula[0].features)
        trainIds = [0] * numSamples
        featureMatrix = np.zeros((numSamples, numFeatures))
        for i, x in enumerate(taula):
            trainIds[i] = int(x.id)
            if len(x.features) != numFeatures:
                print("eps!")
            featureMatrix[i, :] = x.features

        featureMatrix = np.array(featureMatrix)
        # trainIds = range(len(trainIds))

        if similarity.lower() == 'euclidean':
            D = dist.squareform(dist.pdist(featureMatrix))
            meanDist = np.mean(D)
            varDist = np.var(D)
        elif similarity.lower() == 'gaussian':
            D = dist.squareform(dist.pdist(featureMatrix))
            meanDist = np.mean(D)
            # sigma = 1 #snp.var(D)
            # D = np.exp(-D**2/2/np.std(D))
            idx = np.argsort(np.exp(-D ** 2 / 2 / sigma))
        elif similarity.lower() == 'jaccard':
            D = dist.squareform(dist.pdist(featureMatrix, metric=similarity.lower()))

        D[D > L] = L
        # A = []
        # Compute d'ajacency matrix for the training set
        A = [(i, j, {'edist': float(D[i, j]), 'gdist': np.exp(-(float(D[i, j])) ** 2 / 2 / sigma)})
             for i in range(len(trainIds))
             for j in range(i + 1,len(trainIds)) if float(D[i, j]) <= L]
        # for i in trainIds:
        #     for j in trainIds[i + 1:]:
        #         A = A + [
        #             (int(i), int(j), {'edist': float(D[i, j]), 'gdist': np.exp(-(float(D[i, j])) ** 2 / 2 / sigma)})]
        #         # A = A +  [ (int(i),int(idx[i,pos][x]),{'edist': float(D[i,idx[i,pos]][x]), 'gdist': np.exp(-(float(D[i,idx[i,pos]][x])-meanDist)**2/2/varDist  )}) for x in range(0,len(pos))  ]


        aD = np.argsort(D, axis=1)
        prova = np.take_along_axis(D, aD[:, 1:K + 1], axis=1)
        A = np.array([(i, aD[j + 1, i]) for i in range(numSamples) for j in range(K)])
        sA = sp.coo_matrix(([1] * K * numSamples, (A[:, 0], A[:, 1])), shape=(numSamples, numSamples))
        adj = sA.toarray() * sA.toarray().transpose()

        # Build the mutual-KNN graph on the training dataset
        G = nx.Graph()
        G.add_nodes_from(range(len(trainIds)))
        G.add_edges_from(A)

        nconn = nx.number_connected_components(G)

        return G

    def computeKnn(self,taula,  **kwargs):
        '''

        :param taula:
        :param similarity:
        :param sigma:
        :return:
        '''

        # taula.show(10)
        k = self.mutual_k
        numSamples = len(taula)
        if k > numSamples / 2:
            print(
                "Number of samples (%d) lower than number of mutual friends (%d). Updating k \n" % (numSamples / 2, k))
            k = min( int(numSamples / 2 - 1), k)
            self.__mutual_k = k

        trainIds = np.array([int(x.id) for x in taula])
        featureMatrix = np.array([x.features for x in taula])

        # trainIds = range(len(trainIds))


        # __similarity = 'gaussian', sigma = 1
        if 'similarity' not in kwargs:
            similarity = self.similarity
        else:
            similarity = kwargs['similarity']

        if similarity.lower() in ('euclidean','invdist'):
            D = dist.squareform(dist.pdist(featureMatrix))

            idx = np.argsort(D, axis=1)[:,1:k + 1]
            meanDist = 0 #np.mean(D)
            varDist = 1 #np.var(D)
        elif similarity.lower() == 'gaussian':
            D = dist.squareform(dist.pdist(featureMatrix))
            meanDist = np.mean(D)
            varDist = 1  # snp.var(D)
            # D = np.exp(-D**2/2/np.std(D))
            idx = np.argsort(np.exp(-D ** 2 / 2 / varDist), axis=1)[:,-k - 1:-1]
        elif similarity.lower() == 'jaccard':
            D = dist.squareform(dist.pdist(featureMatrix, metric=similarity.lower()))
            idx = np.argsort(D, axis=1)[:,-k - 1:-1]

        # A = []
        #
        # # Compute d'ajacency matrix for the training set
        # for i in range(0, len(trainIds)):
        #     pos = [x for x in range(0, k) if any((idx[idx[i, :]] == i)[x, :])]
        #     A = A + [(int(trainIds[i]), int(trainIds[idx[i, pos][x]]), {'edist': float(D[i, idx[i, pos]][x]),
        #                                                                 'gdist': np.exp(-(float(D[i, idx[i, pos]][
        #                                                                                             x]) - meanDist) ** 2 / 2 / varDist)})
        #              for x in range(0, len(pos))]
        #     # A = A +  [ (int(i),int(idx[i,pos][x]),{'edist': float(D[i,idx[i,pos]][x]), 'gdist': np.exp(-(float(D[i,idx[i,pos]][x])-meanDist)**2/2/varDist  )}) for x in range(0,len(pos))  ]

        # aD = np.argsort(D, axis=1)
        A = np.array([(i, idx[i,j]) for i in range(numSamples) for j in range(k)])
        sA = sp.coo_matrix(([1] * k * numSamples, (A[:, 0], A[:, 1])), shape=(numSamples, numSamples))
        adj = sA.toarray() * sA.toarray().transpose()

        # Build the mutual-KNN graph on the training dataset
        G = nx.Graph()
        G.add_nodes_from(trainIds)
        pos = np.nonzero(adj)
        meanDist = np.mean(D[pos[0], pos[1]])
        varDist = np.var(D[pos[0],pos[1]])
        G.add_edges_from([ (n,m,{'invdist': 1/(1+D[n,m]) , 'edist':D[n,m],'gdist': np.exp(-(float(D[n,m])-meanDist)**2/2/varDist  )}) for (n,m) in zip(pos[0],pos[1])]  )

        # ego = lambda x: nx.ego_graph(G, x)
        #
        # stats = [ (lambda y : (len(y.nodes), len(y.edges)))(ego(node)) for node in G.nodes]
        #
        # plt.scatter([n for (n, _) in stats], [e for (_, e) in stats])


        # nconn = nx.number_connected_components(G)

        return G

    def hoad(self, graphList, m=1, k=3, show=True, similarity=None):

        if similarity is None:
            similarity = self.similarity

        if len(graphList) == 1:
            A = sp(nx.to_numpy_matrix(graphList[0], range(max(graphList[0].nodes())), weight=similarity))
            B = A
            K = int(k)  # + nx.number_connected_components(graphList[0])
            print ("Computing anomaly detection with one graph layer. No anomalies should be detected")
        elif len(graphList) == 2:
            numSamples = max(max(graphList[0].nodes()), max(graphList[1].nodes())) + 1
            if numSamples > self.__maxSamples:
                A = sp.coo_matrix(nx.to_numpy_matrix(graphList[0], range(numSamples), weight=similarity))
                B = sp.coo_matrix(nx.to_numpy_matrix(graphList[1], range(numSamples), weight=similarity))
                C = sp.eye(numSamples) * (np.mean(A[np.nonzero(A)]) + np.mean(B[np.nonzero(B)])) / 2
            else:
                A = nx.to_numpy_matrix(graphList[0], range(numSamples), weight=similarity)
                B = nx.to_numpy_matrix(graphList[1], range(numSamples), weight=similarity)
                C = np.eye(numSamples) * m
                # print "Computing anomaly detection with two graph layers. "

            K = int(k)  # + (nx.number_connected_components(graphList[0])+nx.number_connected_components(graphList[1]))/2
        else:
            print ("Anomaly detection with more than 2 layers  is still not implemented")
            return 0

        if A.shape != B.shape:
            print ("Adjacency matrices should have the same number of nodes")
            return 0

        # numSamples,_ = A.shape
        if numSamples > self.__maxSamples:
            Z = sp.vstack((sp.hstack((A, C)), sp.hstack((C, B))))
        else:
            Z = np.vstack((np.hstack((A, C)), np.hstack((C, B))))

        # D=np.diag(np.sum(Z,1))
        if show:
            I, J, V = sp.find(Z)
            W = []
            for i, j, v in zip(I, J, V):
                W = W + [(i, j, {'dist': v})]

            G = nx.Graph()
            G.add_edges_from(W)

            pos = nx.spring_layout(G)  # positions for all nodes
            labels = {}
            for i, x in enumerate(list(range(0, numSamples)) + list(range(0, numSamples))): labels[i] = r'$%d$' % (x)
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=range(0, numSamples),
                                   node_color='r',
                                   node_size=500,
                                   alpha=0.8)
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=range(numSamples, 2 * numSamples),
                                   node_color='b',
                                   node_size=500,
                                   alpha=0.8)
            nx.draw_networkx_labels(G, pos, labels, font_size=16)
            nx.draw_networkx_edges(G, pos, edgelist=W, alpha=0.5)
            plt.axis('off')
            # plt.savefig("labels_and_colors.png") # save as png
            plt.show()

        if numSamples > self.__maxSamples:
            # L = D - Z
            # U,S,V = svds(sp.eye(numSamples*2) - Z)
            # k = numSamples
            U, S, _ = svds(csgraph.laplacian(Z), K, which='SM')

        else:
            U, S, V = np.linalg.svd(np.eye(numSamples * 2) - Z)

        u = U[:numSamples, -K:]
        v = U[numSamples:, -K:]

        # ===========================================================================
        # u = U[:numSamples,np.isnan(S) == False]
        # v = U[numSamples:,np.isnan(S) == False]
        # ===========================================================================

        # ===========================================================================
        # uu=np.matmul(A,u)
        # vv=np.matmul(B,v)
        #
        # ss = 1-np.sum(uu*vv,1)/np.sqrt(np.sum(uu*uu,1))/np.sqrt(np.sum(vv*vv,1))
        # ===========================================================================

        uv = np.sum(np.multiply(u, v), 1)
        nu = np.sqrt(np.sum(np.multiply(u, u), 1))
        nv = np.sqrt(np.sum(np.multiply(v, v), 1))
        s = 1 - np.abs(uv / nu / nv)
        s[np.isnan(s)] = 0
        # ===========================================================================
        # s = 1-u*v/np.abs(u)/np.abs(v)
        # ===========================================================================

        return s  # np.where(s<.5)[0].tolist()

    def detector(self, FeaturesList, y_train=None , **params ):

        # similarities = [ 'jaccard' if layer == 'Label' else 'gaussian' if layer == 'Sigmoid' for layer in dbms.layers ]

        G = [ self.computeKnn(FeaturesList[i],  **params) for i in range(len(self.layers)) ]
        # for i, layer in enumerate(dbms.layers):
        #     if layer == 'Label': dbms.__similarity = 'jaccard'
        #     if layer == 'Sigmoid': dbms.__similarity = 'gaussian'
        #     G.append(dbms.computeKnn(FeaturesList[i],  **params))
            #G.append(dbms.computeFullyConnected( FeaturesList[i], __similarity=dbms.__similarity, sigma=params['sigma']))

        # if loadFeatures: loadDescriptors(keyspace,table,featuresFile,hosts)
        return self.hoad(G, m=params['m'], k=params['k'], show=False)





