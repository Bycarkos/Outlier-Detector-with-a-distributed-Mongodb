# -*- coding: utf-8 -*-
from __future__ import division

u'''
 
This script evaluates the performance of the following outlier detection method:
    - Consensus Regularized Multi-View Outlier Detection (CMOD)
    - DMOD
    - A multi-graph spectral framework for mining multi-source anomalies (COD) [3]

Arguments:
    -c, --config: JSON file with the information required to insert data
    -N, --datasetName: name of the imported dataset
    -D, --dbms: Database management system used to import data (Oracle or MongoDB).
    -f, --featuresImage: extracted features from image dataset. e.g -f "{'cnn':'AlexNet', 'layer':'Visual'}"
    -m, --method: coma-separated list with the outlier detection methods to test (either CMOD, DMOD or COD)
    -p, --params: string on JSON format with the method parameters and their values. e.g. -p "{'k':2, 'sigma':.1, 'm':1}"


[3]	J. Gao, N. Du, W. Fan, D. Turaga, S. Parthasarathy, J. Han, "A multi-graph spectral framework for mining 
multi-source anomalies" in in Graph Embedding for Pattern Analysis, New York, NY, USA:Springer, pp. 205-228, 2013.

Created on 26/2/2018
  
@author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
@Institution: Computer Vision Center - Universitat Autonoma de Barcelona
'''

__author__ = 'Oriol Ramos Terrades'
__email__ = 'oriolrt@cvc.uab.cat'

import ast
import getopt
import json
import os.path
import pymongo
import sys
import _pickle as cPickle
from bson.binary import Binary

import logging

from collections import namedtuple

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import random

try:
    from .OutlierDetector.CMOD import CMOD
    from .OutlierDetector.DMOD import DMOD
    from .OutlierDetector.HOAD import HOAD
    from .OutlierDetector.LOF import LOF
    from .OutlierDetector.LOCI import LOCI
    from .OutlierDetector.iForest import iForest
    from .OutlierDetector.KNN import KNN
    from .OutlierDetector.SO_GAAL import SOGAAL
    from .OutlierDetector.GMM import GMM
    from .OutlierDetector.APS import APS
    from .OutlierDetector.DBSCAN import DBSCAN
    from .OutlierDetector.COD import COD
    from .processData import config as cfg, mongoConnection as mg
    from .processData import noConnection as nc
    from .options import OptionsTest
except ImportError:
    from OutlierDetector.CMOD import CMOD
    from OutlierDetector.DMOD import DMOD
    from OutlierDetector.HOAD import HOAD
    from OutlierDetector.LOF import LOF
    from OutlierDetector.LOCI import LOCI
    from OutlierDetector.iForest import iForest
    from OutlierDetector.KNN import KNN
    from OutlierDetector.SO_GAAL import SOGAAL
    from OutlierDetector.GMM import GMM
    from OutlierDetector.APS import APS
    from OutlierDetector.DBSCAN import DBSCAN
    from OutlierDetector.COD import COD
    from processData import config as cfg, mongoConnection as mg
    from processData import noConnection as nc
    from options import OptionsTest

"""Funcions a implementar"""


def insertExperiment(dbms, conf, numViews, repeticio, method, paramsMethod):
    """
    Inserim en la taula Experiment la informació bàsica de l'experiment, si no existeix. Si existeix retorna el OID del document

    :param dbms: object with the data connection
    :param conf: ratio of class outliers and attribute outliers
    :param numViews: Number of views
    :param method: name of the evaluated outlier detection algorithm
    :param paramsMethod: list of parameter names
    :return: idExperiment
    """

    #print("FINAAAL")
    if "Experimental" not in dbms.bd.list_collection_names():
        dbms.bd.create_collection("Experimental")
        id = 0  # No  experiment in database

    else:
        #agafem el ultim id
        idEM = list(dbms.bd.Experimental.find({}, {"_id": 1}).sort("_id", -1).limit(1))

        for x in idEM:
            id = x['_id'] + 1 #incrementem el id per no repetir
    dbms.bd.Experimental.insert_one({"_id": id, "experiment": paramsMethod,
                                         "conf": str(conf[0])+"-"+str(conf[1]), "num_views": numViews, "rep": repeticio, "met": method})

    return id


def insertOutlierData(dbms, newFeatures, nameDataset, repeticio, outliersGTIdx, conf):
    """
    Inserim els outliers

    :param dbms: object with the data connection
    :param newFeatures: new outlier features
    :param nameDataset: name of the dataset
    :param repeticio: experiment iteration number
    :param outliersGTIdx: Id of the modified samples
    :param conf: string containing the experiment configuration
    :return:

    """
    print(nameDataset)
    #print(newFeatures)
    if "New_Outliers" not in dbms.bd.list_collection_names(): #creem col·lecció per els outliers generats
        dbms.bd.create_collection("New_Outliers")
    for x in newFeatures:
        for y in x:
            if y[0] in outliersGTIdx: #recorrem la llista per agafar outliers
                res = 'None'
                for value in dbms.bd[nameDataset].find({'vector':y[1]}):
                    res = value['label']
                dbms.bd.New_Outliers.insert_one(
                    {'dataset': nameDataset, 'rep': repeticio, 'conf': str(conf[0])+"-"+str(conf[1]), 'id': y[0], 'features': y[1], 'label':res})

                #inserim les caracteristiques a la colecció, amb:
                    #id-->id del outlier original
                    #features-->caracteristiques dels nous outliers



def insertResults(dbms, nameDataset, idExperiment, fpr, tpr, auc, **kwargs):
    """
    inserir els resultats

    :param dbms: object with the data connection
    :param nameDataset: name of the dataset
    :param idExperiment: Id identifying the experiment
    :param fpr: vector containing false positive values
    :param tpr: vector containg true positive values
    :param auc: area under the curve score (scalar)
    :return:
    """
    #Actualitzem els resultats en la base de dades segons el id del experiment
    dbms.bd.Experimental.update_many({'_id': idExperiment},
                                     {
                                         "$set": {'fpr': [i for i in fpr], 'tpr': [i for i in tpr], 'auc': auc, 'dataset': nameDataset}})

    return True


def loadOutliers(dbms, nameDataset, repeticio, numSamples, conf, numViews):
    """
    Cal llegir els outliers

    :param dbms: object with the data connection
    :param nameDataset: name of the dataset
    :param repeticio: experiment iteration number
    :param numSamples: number of samples
    :param conf: string containing the experiment configuration
    :return:
    """
    # carregar outliers que hem inserit, bee

    numTotalOutliers = (int(2 * round(conf[0] / 100.0 / 2.0 * numSamples)) + int(
        round(conf[1] / 100.0 * numSamples))) * numViews

    # TODO: Implementeu la funció. Teniu en compte que generateOutliersFlag ha de ser Cert si el nombre de outliers que recupereu de la BD no coincideix amb el valor de numTotalOutliers.
    # Outliers ha de tenir una estructura adequada

    outliers = dict()
    generateOutliersFlag = True

    data = dbms.bd.New_Outliers.find({'dataset': nameDataset, "rep":repeticio, "conf":conf})
    for x in data:
        outliers[x['id']] = x

    if numTotalOutliers == len(list(dict(outliers).keys())):
        generateOutliersFlag = False

    return outliers, generateOutliersFlag


def loadVectorData(dbms, nameDataset, **kwargs):
    '''
    Carrega els vectors de caracteristiques dels datasets de la UCI

    :param dbms: object with the data connection
    :param nameDataset: name of the dataset
    :return:
    '''

    # TODO: Implementa aquesta funció tenint en compte el format dels paràmetres de sortida
    data = dbms.bd[nameDataset]
    """
    for x in data.find():
        print(x)
    """
    """Estructura que esperea de retorn"""
    # res = [{'id': 0, 'features': [0, 0]}]
    #print(data)
    res = data.find({'utility': 'Outliers'}) #{'utility': 'Outilers'},{"_id": 1, "id":1,"label": 1, "vector": 1}

    fila = namedtuple("fila", "id features")
    #print("Here we are",list(res))
    taula = []
    ids = {}
    # res=dict(res)
    #print(res[0])
    count = 0
    for row in res:
        #print(row)
        taula.append(fila(id=row["id"], features=row["vector"]))  # ??? features, serà vector??
        if row["label"] in ids.keys():
            ids[row["label"]] = ids[row["label"]] + [row["id"]]
        else:
            ids[row["label"]] = [row["id"]]
        count+=1
    #print(taula,"\n", ids)
    return taula, ids


def loadImageData(dbms, nameDataset, **kwargs):
    '''
    Carrega els vectors de caracteristiques dels datasets de la UCI

    :param dbms: object with the data connection
    :param nameDataset: name of the dataset
    :return:
    '''

    # TODO: Implementa aquesta funció tenint en compte el format dels paràmetres de sortida
    """Estructura que esperea de retorn"""
    # res = [{'id': 0, 'features': [0, 0]}]
    res = dbms.bd[nameDataset].find({'utility': "Outilers"}, {"_id": 1, "label": 1, "path": 1, "descriptors": 1})
    fila = namedtuple("fila", "id features")
    taula = []
    ids = {}
    # res=dict(res)
    for row in res:
        #print("row",row)
        taula.append(fila(id=row["_id"], features=row["descriptors"]))  ## features == descriptors?
        if row["label"] in ids.keys():
            ids[row["label"]] = ids[row["label"]] + [row["_id"]]
        else:
            ids[row["label"]] = [row["_id"]]

    return taula, ids


"""Final de les funcions a implementar"""


def loadData(dbms, nameDataset, type, **kwargs):
    if nameDataset.lower() == "synthetic data":
        features, classIds, numClasses = generateSyntheticData()
    else:
        print("entra",nameDataset,type)
        if type == "vector":
            features, classIds = loadVectorData(dbms, nameDataset, **kwargs)

        if type == "image":
            features, classIds = loadImageData(dbms, nameDataset, **kwargs)

    return features, classIds


def generateSyntheticData(numSamples=1000, numDims=6):
    numClusters = 2

    fila = namedtuple("fila", "id features")

    ids = {}
    inici = 0
    pas = int(numSamples / numClusters)
    for j in range(0, numClusters):
        ids[str(j)] = range(inici, inici + pas)
        inici = inici + pas
        data = np.random.rand(pas, numDims)
        data = np.vstack((data, j * np.sqrt(numDims) + np.random.rand(pas, numDims)))

    # creem l'estructura tabular per emular el resultat d'una query en una BD
    taula = []
    for id in range(0, numSamples):
        tupla = fila(id=id, features=data[id, :].tolist())
        taula.append(tupla)

    return taula, ids, numClusters


def getConf(confDict):
    confString = confDict.split()
    conf = []
    for c in confString:
        conf.append(tuple([int(x) for x in c.replace("(", "").replace(")", "").split(",")]))

    return conf


def generateAttrOutlier(data, outliersGTIdx, ratio=0.05):
    numViews = len(data)

    numSamples = len(data[0])
    numFeatures = len(data[0][0].features)

    oid = list(set(range(numSamples)).difference(outliersGTIdx))

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

    return data, outliersGTIdx + attrOut


def generateClassOutlier(data, IdsList, ratio=0.05):
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
        currentView = 0
        idSamples = random.sample(range(len(idClasses)), 2)
        features = data[currentView][rid[i, idSamples[0]]].features
        data[currentView][rid[i, idSamples[0]]] = fila(id=data[currentView][rid[i, idSamples[0]]].id,
                                                       features=data[currentView][rid[i, idSamples[1]]].features)
        data[currentView][rid[i, idSamples[1]]] = fila(id=data[currentView][rid[i, idSamples[1]]].id, features=features)
        outliersGTIdx = outliersGTIdx + rid[i, idSamples].tolist()

    outliersGTIdx.sort()

    return data, outliersGTIdx


def prepareExperimentData(dbms, conf, nameDataset, features, repeticio, settings={'numViews': 2}):
    """
    Generates the outlier data given the settings for outlier Detection methods.

    :param features: list of feature vectors used to geneta
    :param classIds:
    :param settings:
    :return:
    """

    class_outlier = float(conf[0] / 100.0)
    attr_outlier = float(conf[1] / 100.0)
    if "numViews" in settings:
        numViews = settings["numViews"]
    else:
        numViews = 2

    if len(features) == 1: features = features[0]
    numDims = len(features[0].features)
    numSamples = len(features)
    newFeatures = [[]] * numViews

    fila = namedtuple("fila", "id features")
    maxVal = [-1000000] * numViews
    for x in features:
        id = x.id

        for y in range(numViews):
            # feats = x.features[y*numFeatures:(y+1)*numFeatures]
            feats = x.features
            newFeatures[y] = newFeatures[y] + [fila(id=id, features=feats)]

    outliers, generateOutliersFlag = loadOutliers(dbms, nameDataset, repeticio, numSamples, conf, numViews)

    if generateOutliersFlag:
        newFeatures, outliersGTIdx = generateClassOutlier(newFeatures, settings["classIds"], ratio=class_outlier)

        newFeatures, outliersGTIdx = generateAttrOutlier(newFeatures, outliersGTIdx, ratio=attr_outlier)

        # Salvem els outliers en la BD
        insertOutlierData(dbms, newFeatures, nameDataset, repeticio, outliersGTIdx, conf)

    else:
        outliersGTIdx = list(outliers.keys())

    y = np.zeros(numSamples)
    y[outliersGTIdx] = 1

    return newFeatures, y, outliersGTIdx


if __name__ == '__main__':

    # Parse options
    opts = OptionsTest()
    args = opts.parse()

    """
    Database default parameters
    """
    DBMS = ""

    params = {}
    datasetName = 'LetterRecognition'#'synthetic data'
    params["numSamples"] = 200
    numViews = None
    isConfigFile = False

    if args.config is not None:
        with open(args.config) as f:
            data = json.load(f)
            #print(data,args)
    else:
        opts.print_help()
        sys.exit(1)

    print(args.metadata) #perque??
    if args.metadata is not None:
        with open(args.metadata) as f:
            metadata = json.load(f)
        print("dataset {}  metadata {} ".format(args.datasetName,metadata.keys()))
        if args.datasetName is not None and args.datasetName.lower() in metadata.keys():
            metadata = metadata[args.datasetName.lower()]
            #print(metadata)
            if args.datasetName == 'iris':
                metadata == 'vector'
        else:
            metadata = dict()
    else:
        metadata = dict() #carga syntetic data

    if DBMS.lower() == "":
        db = nc.noConnection()

    db = mg.mongoConnection(data)

    if db is None:
        opts.print_help()
        sys.exit(1)

    """Iniciem la sessio"""
    db.startSession()

    """Carreguem les dades dels datasets guardats a la BD"""
    features, classIds = loadData(db, datasetName, 'vector')#metadata
    """---"""

    paramNames = []
    if "data" in locals():
        if "numIterations" in data:
            numRepeticions = int(data['numIterations'])
        else:
            numRepeticions = 2
        if "conf" in data:
            confList = getConf(data['conf'])
        else:
            confList = [(2, 0)]
        if numViews is None:
            if "numViews" in data:
                numViews = data["numViews"]
            else:
                numViews = 2
    else:
        numRepeticions = 2
        confList = [(2, 0)]
        if numViews is None:
            numViews = 2

    if args.params is not None:
        paramsMethod = ast.literal_eval(args.params)

    """Inicialitzem"""
    if args.method.upper() == "DMOD":
        od = DMOD(numViews, params=paramsMethod)
    if args.method.upper() == "CMOD":
        od = CMOD(numViews, params=paramsMethod)
    if args.method.upper() == "COD":
        print(data['mutual_k'])
        od = HOAD(int(data['mutual_k']), numViews, **paramsMethod) ## canvi per data en comptes metadata
    if args.method.upper() == "LOF":
        if 'n_neighbors' not in paramsMethod:
            # Fem servir el valor de k "optim" pel mutual-k
            paramsMethod['n_neighbors'] = data['mutual_k']
        od = LOF(**paramsMethod)
    if args.method.upper() == "LOCI":
        od = LOCI(**paramsMethod)
    if args.method.upper() == "IFOREST":
        od = iForest(**paramsMethod)
    if args.method.upper() == "KNN":
        od = KNN(**paramsMethod)
    if args.method.upper() == "GMM":
        od = GMM(**paramsMethod)
    if args.method.upper() == "APS":
        od = APS(**paramsMethod)
    if args.method.upper() == "SO-GAAL":
        od = SOGAAL(**paramsMethod)

    for conf in confList:
        print("""
        ==================================================================

        Iniciant experiment amb configuració: {}-{}

        """.format(conf[0], conf[1]))
        aucVector = []
        for i in range(numRepeticions):
            """Per a cada repetició hem de generar els outliers del dataset """

            isInserted = False

            while not isInserted:
                #print(classIds)
                newFeatures, y_train, outliersGTIdx = prepareExperimentData(db, conf, datasetName, features, i,
                                                                            settings={'numViews': numViews,
                                                                                      "classIds": classIds})

                idExperiment = insertExperiment(db, conf, numViews, i, args.method, paramsMethod)

                outliersIdx = od.detector(newFeatures, y_train=y_train, **paramsMethod)
                """Comprovem que el detector a retornat valors"""
                if outliersIdx is None:
                    logging.warning(
                        "El mètode de detecció d'outliers no retorna res. Comprova que s'han carregat correctament totes les dades")
                    break

                """Calculem les mètriques d'avaluació"""
                # Evaluate Outliers
                fpr, tpr, thresholds = metrics.roc_curve(y_train, outliersIdx, pos_label=1)
                auc = roc_auc_score(y_train, outliersIdx)

                """Inserim els resultats a la BD """
                isInserted = insertResults(db, datasetName, idExperiment, fpr, tpr, auc, **metadata)

                """Mostrem els resultats per consola"""
                if isInserted:
                    valorsStr = "Method: {}".format(args.method)
                    for key in paramsMethod:
                        valorsStr = valorsStr + ", {}={}".format(key, paramsMethod[key])
                    valorsStr = valorsStr + ", {}-{} (repeticio {}): %.3f".format(conf[0], conf[1], i) % (auc)

                    print(valorsStr)
                    aucVector.append(auc)

        print(
            "Conf: {}-{}, {} : {} +/- {}".format(conf[0], conf[1], args.method, np.mean(aucVector), np.std(aucVector)))

    db.close()
    print("Experiments fets")
    sys.exit(0)
