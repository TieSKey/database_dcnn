import os
import numpy as np
import pyflann
import psycopg2
from sqlobject.sqlbuilder import *

import utils
from enum import Enum


class Indexes(Enum):
    fc7 = 'fc7.index'


maxMemory = 6291456


def loadRawFeatureDataSet(layerName):
    """

    :param layerName:
    :return:
    """

    # ## Load features dataset ###
    table = layerName
    featureColumn = "feature"
    idColumn = "imagenet_id"
    limit = 10000

    return loadTable(table, idColumn, featureColumn, limit)


def loadProcesesedFeatureDataSet(layer, algorithm, dimension):
    """
    Load the features from the db table.

    :rtype : list, numpy.ndarray
    :param layer: Layer name as in DB.
    :param algorithm: Algorithm name as in DB.
    :param dimension: Dimension number as in DB.
    :return: A list of ids and a numpy array with all the features.
    """

    # ## Load features dataset ###
    table = algorithm + "_" + layer
    featureColumn = "feature" + dimension
    idColumn = "imagenet_id"
    limit = 10000

    return loadTable(table, idColumn, featureColumn, limit)


def loadTable(table, idColumn, featureColumn, limit):
    import re
    import resource

    # ## Open DB connection ###
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    # ## Get size of array ###
    # prepare size query
    sizeColumn = "array_dims(" + featureColumn + ")"
    select = Select(staticTables=[table]).newItems([sizeColumn])
    sizeStr = sqlrepr(select)
    sizeStr += " LIMIT 1"

    # get size
    cur.execute(sizeStr)
    sizes = re.findall('\d+', cur.fetchone()[0])
    arraySize = int(sizes[1])

    # close this cursor
    cur.close()

    # ## Get feature and id data. ###
    # prepate sql statement
    select = Select(staticTables=[table]).newItems([idColumn, featureColumn])
    selectStr = sqlrepr(select)

    # allocate memory
    ids = np.empty([50000], dtype='int32')
    features = np.empty([50000, arraySize], dtype='float32')

    # create a server curso for efficiency
    cur = conn.cursor("serverCursor")

    # execute sql and retrieve values
    cur.arraysize = 5000
    cur.execute(selectStr)

    # fetch data in batches
    for x in range(0, 10):
        # unpack the lists
        idRows, arrayRows = zip(*cur.fetchmany())
        low = x * 5000
        high = (x + 1) * 5000
        ids[low:high] = np.asanyarray(idRows)
        features[low:high] = np.asanyarray(arrayRows)
        del arrayRows
        del idRows

        # control memory usage
        currentMemory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print "Using: " + str(currentMemory)
        if currentMemory > maxMemory:
            raise ValueError('Not enough memory to perform operation.')

    # close connection
    cur.close()
    conn.close()

    return ids, features


def loadIndex(fileName, dataset):
    """

    :param fileName:
    :return:
    """
    import os

    flann = pyflann.FLANN(cores=4)

    path = os.path.join(utils.flann_indexes_dir, fileName)
    flann.load_index(path, dataset)

    return flann


if __name__ == '__main__':
    import time


    dimensions = ['128', '256', '512']
    algorithms = ['pca']
    layers = utils.feature_layers

    # ids, dataSet = loadProcesesedFeatureDataSet(layers[0], algorithms[0], dimensions[0])
    ids, dataSet = loadRawFeatureDataSet(layers[0])

    # create random list of images to search later
    maxIdx = np.shape(dataSet)[0]
    numberOfSamples = 500
    numberOfNeighbors = min(5, maxIdx)
    idx = np.random.randint(maxIdx, size=numberOfSamples)
    testSet = dataSet[idx, :]

    # ## Train index with FLANN ###
    flann = pyflann.FLANN(cores=4)
    startTime = time.clock()

    # Use hierarchical kMeans
    params = flann.build_index(dataSet, checks=128, algorithm='kmeans', branching=32, iterations=10, log_level="info",
                               cores=4)

    # delete reference to dataSet to free memory for the search step
    del dataSet

    print "Training Took: "
    str(time.clock() - startTime)
    print params

    # Save index to file
    indexName = os.path.join(utils.flann_indexes_dir, str(algorithms[0]) + ".index")
    flann.save_index(indexName)

    # Search for the images in testset using the idex
    startTime = time.clock()
    result, distances = flann.nn_index(testSet, numberOfNeighbors, checks=params["checks"])
    took = time.clock() - startTime
    print "Searching Took: " + str(took)


