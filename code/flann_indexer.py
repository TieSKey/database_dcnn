import numpy as np

import psycopg2
from sqlobject.sqlbuilder import *

import utils


def loadDataSet(layer, algorithm, dimension):
    """
    Load the features from the db table.

    :rtype : list, numpy.ndarray
    :param layer: Layer name as in DB.
    :param algorithm: Algorithm name as in DB.
    :param dimension: Dimension number as in DB.
    :return: A list of ids and a numpy array with all the features.
    """

    # ## Open DB connection ###
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    # ## Load features dataset ###
    table = algorithm + "_" + layer
    featureColumn = "feature" + dimension
    idColumn = "imagenet_id"
    limit = 10000

    # prepate sql statement
    select = Select(staticTables=[table]).newItems([idColumn, featureColumn]).orderBy(idColumn)
    selectStr = sqlrepr(select)

    # execute sql and retrieve values
    cur.execute(selectStr)
    rawData = cur.fetchall()

    # unpack the lists
    ids, features = zip(*rawData)

    # delete reference to rawData so GC can retrieve the memory if needed
    del rawData

    # close connection
    conn.close()

    # make numpy
    dataSet = np.asarray(features)

    # delete reference to features so GC can retrieve the memory if needed
    del features

    return ids, dataSet


if __name__ == '__main__':
    import time
    import pyflann

    dimensions = ['128', '256', '512']
    algorithms = ['pca']
    layers = utils.feature_layers

    ids, dataSet = loadDataSet(layers[0], algorithms[0], dimensions[0])

    # create random list of images to search later
    maxIdx = np.shape(dataSet)[0]
    numberOfSamples = maxIdx / 4
    numberOfNeighbors = min(5, maxIdx)
    idx = np.random.randint(maxIdx, size=numberOfSamples)
    testSet = dataSet[idx, :]

    # ## Train index with FLANN ###
    flann = pyflann.FLANN(cores=4)
    startTime = time.clock()

    # Use hierarchical kMeans
    params = flann.build_index(dataSet, checks=128, algorithm='kmeans', branching=32, iterations=50, log_level="info",
                               cores=4)

    # delete reference to dataSet to free memory for the search step
    del dataSet

    print "Training Took: "
    str(time.clock() - startTime)
    print params

    # Save index to file
    flann.save_index("index.data")

    # Search for the images in testset using the idex
    startTime = time.clock()
    result, distances = flann.nn_index(testSet, numberOfNeighbors, checks=params["checks"])
    took = time.clock() - startTime
    print "Searching Took: " + str(took)


