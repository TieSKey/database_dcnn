__author__ = 'tigershark'

# postgresql-server-dev-9.4
# install with sudo pip install lshash==0.0.4dev
import os

from lshash import LSHash
import numpy as np
import psycopg2


class LSHCompressor(object):
    def __init__(self, dims=128, planesFileName='randomPlanes128'):
        self.nDims = dims
        self.planesFileName = planesFileName


    def transform(self, X):
        return generateSingleHash(X, self.planesFileName, self.nDims)


def generatePlanesWithBias(fileName, nPlanes, nDims, sigma):
    planes = np.random.randn(nPlanes, nDims)

    for idx in range(0, nPlanes):
        if sigma == 0:
            b = 0
        else:
            b = (np.random.uniform(0, sigma * 2) - sigma) / nDims
        planes[idx, :] += b

    np.savez(fileName, planes)


def generateHashes(X, scalar, planesFileName, n_bits=64):
    """
    Generate a n_bits long hash for each input in X
    :param X:
    :param n_bits:
    :return:
    """
    import utils

    # overwrite old matrixes an build some random new ones
    fileName = os.path.join(utils.lsh_planes_dir, planesFileName + '.npz')
    lsh = LSHash(n_bits, np.shape(X[0])[0], matrices_filename=fileName, overwrite=False)
    hashValues = []
    for input_point in X:
        input_point = scalar.transform(input_point)
        hashValues.append(lsh._hash(lsh.uniform_planes[0], input_point))

    return hashValues


def generateSingleHash(X, planesFileName, n_bits=64):
    """
    Generate a n_bits long hash for each input in X
    :param X:
    :param n_bits:
    :return:
    """
    import utils

    # overwrite old matrixes an build some random new ones
    fileName = os.path.join(utils.lsh_planes_dir, planesFileName + '.npz')
    lsh = LSHash(n_bits, np.shape(X)[0], matrices_filename=fileName, overwrite=False)

    return lsh._hash(lsh.uniform_planes[0], X.tolist())


def storeHashesInDb(ids, labels, hashes, tableName, nBits):
    """
    Store the hash and imagenet ids into the db. Creates the table is not exists, adds the column if number of bits is
    not already present.
    :param ids: Imagenet ids.
    :param hashes: bit hashes
    :param tableName: name of the table to insert the hashes
    :param nBits: number of bits of the hashes
    :return:
    """
    import utils

    # ## Open DB connection ###
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    columnName = "feature" + str(nBits)
    columnExists = False

    # check if table exists
    tableCheckSql = "SELECT EXISTS ( SELECT 1 FROM information_schema.tables   WHERE  table_schema = 'public'   AND    table_name = '" + tableName + "')"
    cur.execute(tableCheckSql)
    tableExists = cur.fetchone()[0]

    if tableExists:
        # check if column exists
        whereStr1 = "table_name='" + tableName + "'"
        whereStr2 = "column_name='" + columnName + "'"
        columnCheckSql = "SELECT TRUE FROM information_schema.columns where " + whereStr1 + " AND " + whereStr2

        cur.execute(columnCheckSql)
        columnExists = cur.fetchone()

        if not columnExists:
            # create column
            createColumnStr = "ALTER TABLE " + tableName + " ADD " + columnName + " bit(" + str(nBits) + ")"
            cur.execute(createColumnStr)

    else:
        # create table
        createTableStr = "CREATE TABLE " + tableName + " (id serial PRIMARY KEY, file text, class integer, " + columnName + " bit(" + str(
            nBits) + "))"
        cur.execute(createTableStr)

    updateStr = "UPDATE " + tableName + " SET " + columnName + "=%s WHERE file=%s"
    insertStr = "INSERT INTO " + tableName + " (file, class, " + columnName + ") VALUES (%s, %s, %s)"

    for idx, hash in enumerate(hashes):
        if tableExists:
            sqlCommand = cur.mogrify(updateStr, (hash, ids[idx]))
        else:
            sqlCommand = cur.mogrify(insertStr, (ids[idx], str(labels[ids[idx]]), hash))
        cur.execute(sqlCommand)

    conn.commit()
    cur.close()
    conn.close()


def test():
    import utils

    trueIds, testSet = utils.load_test_set('fc7', 'raw', 0)

    lsh = LSHash(128, np.shape(testSet[0])[0], matrices_filename='lsh_planes.data.npz', overwrite=True)

    for idx, input_point in enumerate(testSet):
        hastValue = lsh._hash(lsh.uniform_planes[0], input_point.tolist())
        print hastValue

        lsh.index(input_point, idx)

    print lsh.query(testSet[3], 3)

    return None


if __name__ == '__main__':
    import utils

    nBits = 4096

    scalar = utils.load_scalar(layer='fc7')

    fileName = os.path.join(utils.lsh_planes_dir, 'randomPlanesBias' + str(nBits))
    var = np.mean(scalar.std_)
    generatePlanesWithBias(fileName, nBits, 4096, var)

    # trueIds, testSet = utils.load_test_set('fc7', 'raw', 0)
    dataset, ids = utils.load_feature_layer('fc7')
    labels = utils.load_train_class_labels()

    hashes = generateHashes(dataset, scalar, 'randomPlanesBias' + str(nBits), nBits)
    storeHashesInDb(ids, labels, hashes, 'lsh_fc7', nBits)
