import sql
import utils
import os, time
import caffe
import numpy as np

# ------------------------------------------------
# Script Params
# ------------------------------------------------

compression_types = ['lsh']

distance_matrix_layer = 'fc7'

# feature_layers = utils.feature_layers

feature_layers = ['fc7']
# dimensions = [32,64,128,256,512]
dimensions = [512]


# top k items to be retrieved and measured
k = 5

# number of test files to evaluate. Must keep small otherwise it will take too long
N = 1000

# ------------------------------------------------
# End Params
# ------------------------------------------------

batch_size = 10  # shouldn't change this to anything more than 10 because caffe handles it in an unknown way.

# load the test set
test_files = utils.load_test_set()
test_files = test_files[:N]
net, params, blobs = utils.load_network()

# validation labels are the test set labels
test_labels = utils.load_test_class_labels()
labels = utils.load_english_labels()

dist_mat = utils.load_distance_matrix(distance_matrix_layer)

dbObj = sql.KQuery()



# initialize results data object
results = {}
for c_type in compression_types:
    results[c_type] = {}
    for layer in feature_layers:
        results[c_type][layer] = {}
        for n_components in dimensions:
            results[c_type][layer][n_components] = {'similarity_dist': [], 'avg_time': []}

for c_type in compression_types:
    for layer in feature_layers:

        scalar = utils.load_scalar(layer=layer)

        for n_components in dimensions:
            succs = 0  # how many times we retrieved at least 1 image of the true class
            hits = 0  # how many images of the true class we retrieved in total

            compressor = utils.load_compressor(layer=layer,
                                               dimension=n_components,
                                               compression=c_type)

            count = 0
            for t_files in utils.batch_gen(test_files, batch_size=batch_size):


                images = []
                for t_file in t_files:
                    image_path = os.path.join(utils.test_dir, t_file)
                    images.append(caffe.io.load_image(image_path))

                # predict takes any number of images, and formats them for the Caffe net automatically
                prediction = net.predict(images, oversample=False)

                for i in range(batch_size):
                    t_file = t_files[i]
                    feat = net.blobs[layer].data[i].ravel()
                    feat = scalar.transform(feat)

                    comp_feat = compressor.transform(feat)
                    if isinstance(comp_feat, np.ndarray):
                        comp_feat = comp_feat.ravel()

                    # run the top k query and time it
                    st = time.time()

                    query_results = dbObj.query_top_k(k=k,
                                                      features=comp_feat,
                                                      compression=c_type,
                                                      layer=layer,
                                                      dimension=n_components,
                                                      distance='hamming')

                    et = time.time()

                    t_class = test_labels[t_file]

                    worst_case = np.mean(dist_mat[t_class, :])
                    best_case = 0

                    class_distance = 0
                    last_hits = hits
                    for x in query_results:
                        class_distance += dist_mat[t_class, x[1]]
                        if t_class == x[1]:
                            hits += 1

                    if last_hits < hits:
                        succs += 1

                    if len(query_results) == 0:
                        avg_dist = 0
                    else:
                        avg_dist = class_distance / len(query_results)

                    results[c_type][layer][n_components]['similarity_dist'].append(
                        (worst_case - avg_dist) / (worst_case - best_case))
                    results[c_type][layer][n_components]['avg_time'].append(et - st)

                count += batch_size

                if count % 500 == 0:
                    mean_dist = np.mean(results[c_type][layer][n_components]['similarity_dist'])
                    mean_time = np.mean(results[c_type][layer][n_components]['avg_time'])
                    print 'Evaluate Script :: C Type : ', c_type, ' // Layer : ', layer, ' // Dim : ', n_components, ' // Count : ', count
                    print 'Evaluate Script :: Similarity Distance : ', mean_dist, ' // Avg Time : ', mean_time
                    print "'Evaluate Script :: Success: " + str(succs) + " Hits: " + str(hits)

            results[c_type][layer][n_components]['similarity_dist'].append(
                (worst_case - avg_dist) / (worst_case - best_case))
            results[c_type][layer][n_components]['avg_time'].append(et - st)

    utils.dump_results(results, c_type, distance_matrix_layer)

