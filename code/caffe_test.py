import numpy as np
import matplotlib.pyplot as plt
import os


def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data, interpolation='none')

# Make sure that caffe is on the python path:
caffe_root = '/home/eric/caffe/caffe-master/'  # this file is expected to be in {caffe_root}/examples

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../caffe/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../caffe/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'


# alexnet
# MODEL_FILE = './caffe/bvlc_alexnet/deploy.prototxt'
# PRETRAINED = './caffe/bvlc_alexnet/bvlc_alexnet.caffemodel'

IMAGE_FILE = '../images/examples/cat.jpg'
IMAGE_FILE = '../images/imagenet/ILSVRC2012_val_00010306.JPEG'

mean = np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
print mean

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(256, 256))

blobs = [(k, v.data.shape) for k, v in net.blobs.items()]
params = [(k, v[0].data.shape) for k, v in net.params.items()]

print 'Blobs : ', blobs
print 'Params : ', params

net.set_phase_test()
net.set_mode_gpu()

input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show() # you should see the cat

# predict takes any number of images, and formats them for the Caffe net automatically
prediction = net.predict([input_image])
print 'prediction shape:', prediction[0].shape
print 'predicted class:', prediction[0].argmax()
# plt.plot(prediction[0])
# plt.show()

# filters = net.params['conv1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1))
# plt.show()

# get the first image's conv5 layer
feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)
plt.axis('off')
plt.show()


# load labels
imagenet_labels_filename = os.path.join('../caffe/synset_words.txt')
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    import subprocess
    # subprocess.call(['../data/ilsvrc12/get_ilsvrc_aux.sh'])
    # labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
print labels[top_k]


# Simple generator for looping over an array in batches
def batch_gen(data, batch_size):
    for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]


# import os
# dir = "../images/imagenet"
# for files in batch_gen(os.listdir(dir),10):
#     print files