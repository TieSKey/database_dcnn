__author__ = 'tigershark'
import pyflann
import numpy as np
import utils
from enum import Enum


def search(flann, features, numberOfNeighbors):
    return flann.nn_index(features, numberOfNeighbors, checks=128)


def searchForNeighbors(flann, imageFileName, numberOfNeighbors, layer, algorithm, dimensions):
    features = utils.get_features(imageFileName, layer, algorithm, dimensions)

    return flann.nn_index(features, numberOfNeighbors, checks=128)


if __name__ == '__main__':
    import flann_indexer as fi

    trueIds, testSet = utils.load_test_set('fc7', 'raw', 0)

    ids, dataSet = fi.loadRawFeatureDataSet('fc7')

    flann = fi.loadIndex(fi.Indexes.fc7, dataSet)

    ids, distances = search(flann, testSet, 5)

    correct = 0
    incorrect = 0
    for idx, val in enumerate(trueIds):
        if val in ids[idx]:
            correct += 1
        else:
            incorrect += 1

    print "Correct: " + str(correct)
    print "Errors: " + str(incorrect)