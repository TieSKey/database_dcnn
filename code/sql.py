import psycopg2
import time
import utils
import numpy as np


class KQuery(object):
    def __init__(self):
        self.conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
        self.cur = self.conn.cursor()


    def query_top_k(self, k, features, compression, layer, dimension, distance):

        if distance == 'distance2':
            sql_command = "SELECT file, class, distance2( %s ," + create_feature_name(
                dimension) + ") as D FROM " + create_table_name(compression, layer) + " ORDER BY D ASC LIMIT " + str(k)
        elif distance == 'hamming':
            sql_command = "SELECT file, class, hamming( %s ," + create_feature_name(
                dimension) + ") as D FROM " + create_table_name(compression, layer) + " ORDER BY D ASC LIMIT " + str(k)

        if isinstance(features, np.ndarray):
            self.cur.execute(sql_command, [features.tolist()])
        else:
            self.cur.execute(sql_command, (features,))

        return self.cur.fetchall()


def store_feature(layers, compression):
    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    start_time = time.clock()

    train_labels = utils.load_train_class_labels()
    for layer in layers:
        # CREATE THE TABLE
        table_name = create_table_name(compression, layer)
        cur.execute("DROP TABLE IF EXISTS " + table_name + ";")
        table_command = "CREATE TABLE " + table_name + " (id serial PRIMARY KEY, file text, class integer, "
        insert_command = "INSERT INTO " + table_name + " (file, class,"
        values_sql = "VALUES(%s,%s,"

        dimensions = utils.get_dimension_options(layer, compression)
        if len(dimensions) == 0:
            print 'POSSIBLE ERROR: No dimensions loaded for ', layer, ' with ', compression
            continue

        for dim in dimensions:
            table_command += create_feature_column(dim)
            insert_command += create_feature_name(dim) + ","
            values_sql += "%s,"

        table_command = table_command[:-1] + ");"
        values_sql = values_sql[:-1] + ");"
        insert_command = insert_command[:-1] + ") " + values_sql

        print table_command
        print insert_command

        cur.execute(table_command)

        # INSERT DATA INTO TABLE

        # load the data
        X, imagenet_ids = utils.load_feature_layer(layer)
        scalar = utils.load_scalar(layer)

        X = scalar.transform(X)

        transforms = []
        # apply the compression algorithm
        for dim in dimensions:
            compressor = utils.load_compressor(layer, dim, compression)
            transforms.append(compressor.transform(X))

        value = []
        for i in range(X.shape[0]):
            file_name = imagenet_ids[i]
            value = [file_name, train_labels[file_name]]
            for X_prime in transforms:
                value.append(X_prime[i, :].tolist())

            cur.execute(insert_command, value)

        conn.commit()

    cur.close()
    conn.close()

    print 'Done Creating Tables'
    print 'Total Time : ', time.clock() - start_time


def query_top_k(k, features, compression, layer, dimension):
    """
    Returns top k results according to the distance2 function

    Results are of the form [(file, class, distance)] sorted with closest item at 0

    :type k: int
    :param k: top k items returns
    :param features:

    :type compression: str
    :param compression: compression type identifier

    :type layer: str
    :param layer: feature layer

    :type dimension: int
    :param dimension: feature dimensionality

    :return:
    """
    # if dimension != features.size:
    # raise ValueError('Feature size did not match dimension of query requested.')

    conn = psycopg2.connect(dbname=utils.dbname, user=utils.user, password=utils.password, host=utils.host)
    cur = conn.cursor()

    sql_command = "SELECT file, class, hamming( %s ," + create_feature_name(
        # sql_command = "SELECT file, class, distance2( %s ," + create_feature_name(
        dimension) + ") as D FROM " + create_table_name(compression, layer) + " ORDER BY D ASC LIMIT " + str(k)

    # print sql_command
    # cur.execute(sql_command, [features.tolist()])
    cur.execute(sql_command, (features,))

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results


def create_table_name(compression, layer):
    return compression + '_' + layer


def create_feature_column(dim):
    return " feature" + str(dim) + " float8[],"


def create_feature_name(dim):
    return " feature" + str(dim)


if __name__ == '__main__':
    layers = ['pool5']
    compression = 'pca'
    store_feature(layers, compression)


