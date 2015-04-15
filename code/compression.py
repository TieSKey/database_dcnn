import os
import utils
from sklearn.decomposition import KernelPCA, IncrementalPCA
import numpy as np
import hickle as hkl


def generate_pca_compression(X, n_components=16, batch_size=100):
    """
    Compresses the data using sklearn PCA implementation.

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for PCA to keep
    :param batch_size: Batch size for incrimental PCA

    :return: X_prime (the compressed representation), pca
    """

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pca.fit(X)

    return pca.transform(X), pca


def generate_kpca_compression(X, n_components=16):
    """
    Compresses the data using sklearn KernelPCA implementation.

    :param X: Data (n_samples, n_features)
    :param n_components: Number of dimensions for PCA to keep

    :return: X_prime (the compressed representation), pca
    """

    kpca = KernelPCA(n_components=n_components, kernel='rbf', eigen_solver='arpack', fit_inverse_transform=False)
    kpca.fit(X)

    return kpca.transform(X), kpca


def generate_lsh_compression(X, layer, n_bits=64):
    """
    Placeholder to respect current "entry point" for compression
    :param X:
    :param n_bits:
    :return:
    """
    import lsh_indexer as li

    return li.generateHashes(X, str(layer) + "_" + str(n_bits), n_bits), None


if __name__ == '__main__':
    import time

    for layer in utils.feature_layers:
        # layer = 'fc7'
        n_components = 128

        X, ids = utils.load_feature_layer(layer)
        scalar = utils.load_scalar(layer)
        X = X[0:1000, :]
        ids = ids[0:1000]

        print 'PCA Num Components : ', n_components
        start_time = time.clock()
        X_prime, pca = generate_pca_compression(X, n_components)

        print 'PCA Compression time : ', (time.clock() - start_time)
        print 'PCA Compression time per sample : ', (time.clock() - start_time) / X.shape[0]

        compression = 'pca'
        compression_path = os.path.join(utils.compression_dir, compression, layer)
        file_name = compression + '_' + str(n_components) + '_gzip.hkl'

        file_path = os.path.join(compression_path, file_name)

        hkl.dump(pca, file_path, mode='w', compression='gzip')

        print X_prime.shape



