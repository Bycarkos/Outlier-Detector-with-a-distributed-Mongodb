# -*- coding: utf-8 -*-

from os.path import basename,splitext

try:
  from .OutlierDetector import OutlierDetector as od
except ImportError:
  from OutlierDetector import OutlierDetector as od

from scipy.sparse import csgraph
from scipy.sparse.linalg import svds
#from sklearn.cluster import KMeans
from scipy.spatial import distance as dist
from scipy.integrate import cumtrapz

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import collections
import pandas as pd
from scipy.io import loadmat

from networkx.algorithms import community
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.community import girvan_newman


from sklearn.svm import LinearSVC
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC



class COD(od):
    u'''

    This class implements the COD method that was introduced in [1]


    [1]
    Created on 1/9/2018

    @author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
    @Institution: Computer Vision Center - Universitat Autonoma de Barcelona
    '''

    __slots__ = '_G', '__mutual_k', '__similarity', '__k', '__maxSamples', '_D', '__numSamples', '_C'


    def __init__(self,G=None,mutual_k=40,numViews=2,**kwargs):
        '''
        Constructor
        '''

        self._COD__mutual_k = mutual_k

        self._COD__G = G

        self._COD__SVM =  LinearSVC(random_state=0, tol=1e-5)

        if 'name' not in kwargs:
            kwargs['name'] = 'dummy'

        #Default parameters
        defaultParams = {'sigma': 1, 'similarity': 'invDist', 'k': 3, 'maxSamples': 20000, 'node_ids': None,
                         'clf_name': 'clf_'+kwargs['name']+'.joblib', 'percentile':.75}

        for k in defaultParams.keys():
            if k in kwargs:
                self.__setattr__('_COD__'+k, kwargs[k])
            else:
                self.__setattr__('_COD__'+k, defaultParams[k])




        super(COD, self).__init__(numViews)


    @property
    def mutual_k(self):
        return self._COD__mutual_k

    @mutual_k.setter
    def mutual_k(self, mutual_k):
        self._COD__mutual_k = mutual_k

    @property
    def k(self):
        return self._COD__k

    @k.setter
    def k(self, value):
        self._COD__k = value

    @property
    def similarity(self):
        return self._COD__similarity

    @similarity.setter
    def similarity(self, valor):
        if valor.lower() in ['invDist']: #, 'euclidean', 'gaussian', 'jaccard']:
            self._COD__similarity = valor
        else:
            self._COD__similarity = 'invDist'

    @property
    def clf(self):
        return self._COD__SVM

    @clf.setter
    def clf(self, value):
        self._COD__SVM = value

    @property
    def clf_name(self):
        return self._COD__clf_name

    @clf_name.setter
    def clf_name(self, value):
        self._COD__clf_name = value

    @property
    def percentile(self):
        return self._COD__percentile

    @percentile.setter
    def percentile(self, value):
        if value >= 0.0 and value <=1.0:
            self._COD__percentile = value
        else:
            print("percentile value must be in the range [0,1]")



    def __load_matlab_file(self,fileMat):

        mat = loadmat(fileMat)

        x = mat['SVMSpace2']
        OutIdxTrain = mat['OutIdx'][0]-1
        # # Ent1 contains features        for the 1st view, Ent2 for the 2nd one
        # NTrials = size(OutLierSpaceTrain.Feat1, 2);
        # Ent = [OutLierSpaceTrain(:).Feat1];
        # Ent1 = Ent(:, [1: 2:NTrials]);
        # Ent2 = Ent(:, [2: 2:NTrials]);
        # Ent1 = Ent1(:);
        # Ent2 = Ent2(:);
        # RhoMean = [OutLierSpaceTrain(:).Feat2];
        # RhoMean1 = RhoMean(:, [1: 2:NTrials]);
        # RhoMean2 = RhoMean(:, [2: 2:NTrials]);
        # RhoMean1 = RhoMean1(:);
        # RhoMean2 = RhoMean2(:);
        #

        Ent1 = x['Feat1'][0][0][:,::2].transpose().flatten()
        Rho1 = x['Feat2'][0][0][:,::2].transpose().flatten()

        Ent2 = x['Feat1'][0][0][:, 1::2].transpose().flatten()
        Rho2 = x['Feat2'][0][0][:,1::2].transpose().flatten()

        #
        # XSVM = [Ent2 RhoMean2];

        X = np.vstack((Ent2, Rho2)).transpose()
        NSamples = X.shape[0]

        # IsolatedIdx = find((XSVM(:, 1) == 1). * (XSVM(:, 2) == 0));[

        IsolatedIdx = np.argwhere((X[:,0]==1)*(X[:,1]==0)).flatten()

        # MixIdx = find((XSVM(:, 1) == 0). * (XSVM(:, 2) == 0));
        MixIdx = np.argwhere((X[:,0]==0)*(X[:,1]==0)).flatten()

        # XSVM(OutIdxTrain,:)=[Ent1(OutIdxTrain) RhoMean1(OutIdxTrain)];
        X[OutIdxTrain,0],X[OutIdxTrain,1] = Ent1[OutIdxTrain], Rho1[OutIdxTrain]
        #
        # Balance Classes
        # Class2Idx = [IsolatedIdx; MixIdx; OutIdxTrain(:)];
        Class2Idx = IsolatedIdx.tolist() + MixIdx.tolist() +   OutIdxTrain.tolist()
        # Class1Idx = setdiff(1:size(XSVM, 1), Class2Idx);
        Class1Idx = set(range(NSamples)) - set(Class2Idx)

        # Class1IdxSub = Class1Idx(1:2: end);
        Class1IdxSub = list(Class1Idx)[::2]

        # Class2IdxSub = length(Class1IdxSub) + 1:length(Class1IdxSub) + length(Class2Idx);
        Class2IdxSub = range(len(Class1IdxSub) ,len(Class1IdxSub) + len(Class2Idx))

        #
        # YSVM = ones(1, size(XSVM, 1));
        # YSVM(Class2Idx) = 2;

        Y = np.zeros(NSamples)
        Y[Class2Idx] = 1


        #
        # YSVM = YSVM([Class1IdxSub(:); Class2Idx(:)]);
        # XSVM = XSVM([Class1IdxSub(:); Class2Idx(:)],:);

        Y = Y[Class1IdxSub + Class2Idx]
        X = X[Class1IdxSub + Class2Idx,:]


        #Outliers = mat['OutIdx'][0]



        return X,Y

    def __rho_c(self,G,C):



        S = set()
        for c in C:
            S = S.union(c)

        GS = nx.subgraph(G, S)
        ds = GS.degree()
        #rho = []
        #for i, c in enumerate(C):
        #    GC = nx.subgraph(G, c)
        #    dg = GC.degree()
        #    rho.append({id: g / ds[id] for id, g in dg})


        return [ {id: g / ds[id] for id, g in nx.subgraph(G, c).degree()}  for c in C ], len(C)

    def __compute_delta(self,G):

        pD = dist.pdist(list(nx.get_node_attributes(G, 'features').values()))
        h = 1.0 /(1.0 + pD)

        delta = np.quantile(h, self.percentile)
        h = dist.squareform(h)

        #delta = np.max(np.array(nx.to_numpy_matrix(G, weight='invDist')))  # we should replace by the histogram
        return delta,h


    def __extend_communities(self,G):

        numSamples = G.number_of_nodes()

        C = [list(x) for x in list(k_clique_communities(G, self.k))]

        #C = [list(x) for x in list(girvan_newman(G))]



        rho, numC = self.__rho_c(G, C)
        delta,h = self.__compute_delta(G)

        #IC
        IC = [np.sum(list(x.values())) for x in rho]

        #CS
        CS = dict()

        #h = np.array(nx.to_numpy_matrix(G, weight='invDist'))
        #h = 1/(1+dbms._D)

        V = []
        for c in range(numC):
            V = V + list(rho[c].keys())
        W = set(range(numSamples)) - set(V)


        for c in range(numC):
            z = np.zeros(numSamples)
            z[np.array(list(rho[c].keys()))] = list(rho[c].values())
            for w in W:
                CS[c,w] = np.sum(h[w, :] * z)
                #CS.append(np.sum( h[np.array(list(W)), :] * z, axis=1).tolist())


        #idx = range(numSamples)
        for c in range(numC):
            for w in W:
                if CS[c,w] >= delta * IC[c]:
                    C[c].append(w)


            # for j, value in enumerate((CS[i] >= delta * IC[i]).tolist()):
            #     if value:
            #         print(i,j)
            #         print('list(W): ',list(W)[j])
            #         C[i].append(idx[list(W)[j]])

        #dbms._C = C
        return [list(set(c)) for c in C]
        #return [[0, 1, 2, 3, 4, 6], [4, 5, 6, 7, 8]]

    def __compute_topologic_features(self, G, C):
        """

        """

        T = 1.e-6
        numC = len(C)
        numPunts = G.number_of_nodes()

        node_classes = nx.get_node_attributes(G, 'class')

        labels = np.unique(list(node_classes.values()))

        pc = [[np.sum(np.array([node_classes[x] for x in c]) == l) for l in labels] for c in C]
        pc = np.array([[v / np.sum(pc[x]) for v in pc[x]] for x in range(numC)])


        # Calculem entropia
        Ent = -np.sum(np.log(pc + (pc == 0).astype(int)) * pc, axis=1)
        pi = (Ent < T).astype(int)

        #print(Ent)

        phi1 = np.zeros(numPunts)
        Sv = np.zeros(numPunts)
        for i, c in enumerate(C):
            for x in c:
                phi1[x] = phi1[x] + pi[i]
                Sv[x] = Sv[x] + 1

        phi1 = [a / b for a, b in zip(phi1 + (Sv == 0), Sv + (Sv == 0))]
        # [ pi[c] for x in range(N) for c in range(nC) if C[x,c]> 0]
        #print(phi1)


        # 2a caracteristica
        qi = np.zeros(numPunts)
        for i, c in enumerate(C):
            nc = np.array([node_classes[x] for x in c])
            for x in c:
                qi[x] = qi[x] + (np.sum(nc == node_classes[x]) - 1) / (len(c) - 1)

        # qi = np.array([[(np.sum(d['class'][c] == d['class'][x])-1)/(len(c)-1) for x in c] for c in C ])
        phi2 = [a / b for a, b in zip(qi, (Sv + (Sv == 0)))]
        # np.sum(qi,axis=0)/(nC + [1 if x==0 else 0 for x in nC ] )

        #print(phi2)
        #print(np.transpose(np.array([list(phi1), list(phi2)])))
        phi = np.transpose(np.array([list(phi1), list(phi2)]))

        return phi[:,0], phi[:,1]

    def __transform(self,phi1,phi2):

        phi1 = [1 / (1 - np.log(x)) if x != 0 else 0 for x in phi1]
        phi2 = [1 / (1 - np.log(x)) if x != 0 else 0 for x in phi2]

        return np.transpose(np.array([list(phi1), list(phi2)]))

    def computeMutualKnn(self,taula, **kwargs):
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

        trainIds = []
        featureMatrix = []
        for i, v in taula.items():
            trainIds.append(i)
            featureMatrix.append(v)
        trainIds = np.array(trainIds)
        featureMatrix = np.array(featureMatrix)


        self._D = D = dist.squareform(dist.pdist(featureMatrix))

        idx = np.argsort(D, axis=1)[:, 1:k + 1]

        A = np.array([(i, idx[i,j]) for i in range(numSamples) for j in range(k)])
        sA = sp.coo_matrix(([1] * k * numSamples, (A[:, 0], A[:, 1])), shape=(numSamples, numSamples))
        adj = sA.toarray() * sA.toarray().transpose()

        # Build the mutual-KNN graph on the training dataset
        G = nx.Graph()
        G.add_nodes_from(trainIds)
        pos = np.nonzero(adj)
        G.add_edges_from([(n, m, {'invdist': 1/(1+D[n, m])}) for (n, m) in zip(pos[0], pos[1])])

        if 'classes' in kwargs:
            nx.set_node_attributes(G, kwargs['classes'], 'class')

        if 'pos' in kwargs:
            nx.set_node_attributes(G, kwargs['pos'], 'pos')




        return G

    def cod(self, graphList, m=1, k=3, show=True, similarity=None):

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


        uv = np.sum(np.multiply(u, v), 1)
        nu = np.sqrt(np.sum(np.multiply(u, u), 1))
        nv = np.sqrt(np.sum(np.multiply(v, v), 1))
        s = 1 - np.abs(uv / nu / nv)
        s[np.isnan(s)] = 0

        return s  # np.where(s<.5)[0].tolist()

    def predict_graph(self, G,labels=True):

        try:
            self.clf = load(self.clf_name)
        except IOError:
            self.__fit_graph(G)

        phi, y = self.computeTopologicalFeatures(G, labels=labels)

        return self.clf.predict(phi), y



    def detector(self, FeaturesList, y_train=None , **params ):
        """

        """

        for i in range(len(self.layers)):
            G = self.computeMutualKnn(FeaturesList[i],  **params)
            if 'pos' in params:
                nx.draw(G,params['pos'])
            else:
                nx.draw(G)
            plt.show()

            C = self.__extend_communities(G)
            phi1, phi2 = self.__compute_topologic_features(C)
            phi = self.__transform(phi1,phi2)


        # G = [ dbms.computeMutualKnn(FeaturesList[i],  **params) for i in range(len(dbms.layers)) ]

        # if loadFeatures: loadDescriptors(keyspace,table,featuresFile,hosts)
        return self.cod(G, m=params['m'], k=params['k'], show=False)

    def computeTopologicalFeatures(self, G, labels=False):

        self.__COD_nodes_ids = {x: i for i, x in enumerate(G.nodes)}
        C = self.__extend_communities(G)
        phi1, phi2 = self.__compute_topologic_features(G, C)
        phi = self.__transform(phi1, phi2)

        if labels:
            a = nx.get_node_attributes(G, 'outlier')
            y = np.zeros(G.number_of_nodes())
            for n,v in self.__COD_nodes_ids.items():
                y[v] = a[n]

            return phi, y

        return phi


    def __fit_graph(self,G):

        x,y = self.computeTopologicalFeatures(G,labels=True)

        self.clf.fit(x,y)

        dump(self.clf, self.clf_name)


    def __fit_topologicalFeatures(self,X,Y):


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

        tuned_parameters = [{'kernel': ['linear'], 'C': [0.5, 1, 5, 10, 100, 1000]}]

        scores = ['precision', 'recall']
        # scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                SVC(), tuned_parameters, scoring='%s_macro' % score, cv=10
            )
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()



            self.clf = clf


        dump(self.clf, self.clf_name)


    def fit(self,data,y=None):

        if type(data) == nx.Graph:
            self.__fit_graph(data)
            return

        if type(data) == np.ndarray:
            if y is not None:
                self.__fit_topologicalFeatures(data,y)
                return

        return

    def predict_topological_features(self,x):
        return self.clf.predict(x)

    def predict(self,x):

        if type(x) == nx.Graph:
            return self.predict_graph(x)


        if type(x) == np.ndarray:
            return self.predict_topological_features(x)


        return None


    def load_training_data(self, fileName):
        _, ext = splitext(basename(fileName))

        if ext == '.mat':
            return self.__load_matlab_file(fileName)

        return None







