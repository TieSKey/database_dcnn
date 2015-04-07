import utils
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np



def generate_pca_compression(X, n_components = 16, batch_size=100):
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

if __name__ == '__main__':
    import time

    layer = 'fc7'
    n_components = 128

    X, ids = utils.load_feature_layer(layer)
    X = X[0:1000,:]
    ids = ids[0:1000]

    print 'PCA Num Components : ', n_components
    start_time = time.clock()
    X_prime, pca = generate_pca_compression(X, n_components)

    print 'PCA Compression time : ', (time.clock() - start_time)
    print 'PCA Compression time per sample : ', (time.clock() - start_time) / X.shape[0]


    print X_prime.shape



