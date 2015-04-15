__author__ = 'tigershark'

# postgresql-server-dev-9.4
# install with sudo pip install lshash==0.0.4dev
from lshash import LSHash
import numpy as np
import psycopg2
from sqlobject.sqlbuilder import *
import os
import utils


def generateHashes(X, planesFileName, n_bits=64):
    """
    Generate a n_bits long hash for each input in X
    :param X:
    :param n_bits:
    :return:
    """
    # overwrite old matrixes an build some random new ones
    fileName = os.path.join(utils.lsh_planes_dir, planesFileName + '.npz')
    lsh = LSHash(n_bits, np.shape(X[0])[0], matrices_filename=fileName, overwrite=True)
    hashValues = []
    for input_point in X:
        hashValues.append(lsh._hash(lsh.uniform_planes[0], input_point.tolist()))

    return hashValues


def storeHashesInDb(ids, hashes, tableName, nBits):
    """
    Store the hash and imagenet ids into the db. Creates the table is not exists, adds the column if number of bits is
    not already present.
    :param ids: Imagenet ids.
    :param hashes: bit hashes
    :param tableName: name of the table to insert the hashes
    :param nBits: number of bits of the hashes
    :return:
    """
    # ## Open DB connection ###
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    columnName = "lsh" + str(nBits)

    # check if table exists
    tableCheckSql = "SELECT EXISTS ( SELECT 1 FROM information_schema.tables   WHERE  table_schema = 'public'   AND    table_name = '" + tableName + "')"
    cur.execute(tableCheckSql)
    tableExists = cur.fetchone()[0]

    if tableExists:
        # check if column exists
        whereStr1 = "table_name='" + tableName + "'"
        whereStr2 = "column_name='" + columnName + "'"
        select = Select(items=['TRUE'], staticTables=['information_schema.columns'], where=[whereStr1, whereStr2])
        columnCheckSql = sqlrepr(select)
        cur.execute(columnCheckSql)
        columnExists = cur.fetchone()

        if not columnExists:
            # create column
            createColumnStr = "ALTER TABLE " + tableName + " ADD " + columnName + " bit(" + str(nBits) + ")"
            cur.execute(createColumnStr)
    else:
        # create table
        createTableStr = "CREATE TABLE " + tableName + " (id serial PRIMARY KEY, imagenet_id integer, " + columnName + " bit(" + str(
            nBits) + "))"
        cur.execute(createTableStr)

    insertStr = "INSERT INTO " + tableName + " (imagenet_id, " + columnName + ") VALUES (%s, %s)"

    for idx, hash in enumerate(hashes):
        cur.execute(insertStr, [ids[idx], hash])

    conn.commit()
    cur.close()
    conn.close()


def test():
    trueIds, testSet = utils.load_test_set('fc7', 'raw', 0)

    lsh = LSHash(128, np.shape(testSet[0])[0], matrices_filename='lsh_planes.data.npz', overwrite=True)

    for idx, input_point in enumerate(testSet):
        hastValue = lsh._hash(lsh.uniform_planes[0], input_point.tolist())
        print hastValue

        lsh.index(input_point, idx)

    print lsh.query(testSet[3], 3)

    return None


if __name__ == '__main__':
    trueIds, testSet = utils.load_test_set('fc7', 'raw', 0)

    hashes = generateHashes(testSet, 'fc7_128', 128)
    storeHashesInDb(trueIds, hashes, 'lsh_fc7', 128)