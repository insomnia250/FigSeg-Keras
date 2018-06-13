import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
import datetime
import cv2
from glob import glob
import os
from Sdata.Saug import *
from keras.backend.tensorflow_backend import set_session
from utils.preprocessing import get_train_val

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

prefix = ''


class valAug(object):
    def __init__(self,size=(256,256)):
        self.augment = Compose([
            ResizeImg(size=size),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


import random


def vis_segmentation(image,mask, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(12, 8))
    grid_spec = gridspec.GridSpec(2, 3)

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    # plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    plt.imshow(seg_map)
    # plt.axis('off')
    plt.title('pred map')

    plt.subplot(grid_spec[2])
    # seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_map[seg_map > 0.3] = 1
    seg_map[seg_map < 0.3] = 0
    seg_image = seg_map
    plt.imshow(seg_image)
    # plt.axis('off')
    plt.title('threshed pred map')


    plt.subplot(grid_spec[3])
    # seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(mask)
    # plt.axis('off')
    plt.title('annotation')

    plt.subplot(grid_spec[4])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.4)
    # plt.axis('off')
    plt.title('segmentation overlay')
    # plt.savefig('data/' + str(int(random.random() * 1000)) + '.jpg')
    plt.show()


LABEL_NAMES = np.asarray([
    'background', 'person'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def main(pb_file, img_file):
    """
    Predict and visualize by TensorFlow.
    :param pb_file:
    :param img_file:
    :return:
    """
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)

        for op in graph.get_operations():
            print(op.name)
        y = graph.get_tensor_by_name('proba/Sigmoid:0')
        x = graph.get_tensor_by_name('input_1:0')

    # with tf.gfile.FastGFile("seg.pb", mode='wb') as f:
    #     f.write(graph.SerializeToString())

    img_root = '/media/hszc/data1/seg_data/diy_seg'
    _, val_pd = get_train_val(img_root, test_size=1.0, random_state=42)

    img_paths = val_pd['image_paths'].tolist()
    mask_paths = val_pd['mask_paths'].tolist()

    transform = valAug(size=(256,256))
    deNormalizer = deNormalize(mean=None, std=None)

    import time
    with tf.Session(graph=graph) as sess:
        for img_path, mask_path in zip(img_paths, mask_paths):

            # img = img[:, :, 0:3]
            img =  cv2.cvtColor(cv2.imread(img_path),  cv2.COLOR_BGR2RGB)
            img_h = img.shape[0]
            img_w = img.shape[1]

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((img_h, img_w))
            # batch and val aug
            print '=='*20
            print img_path
            print img.shape
            print mask.shape

            img, _ = transform(img, mask)
            img_batched = img[np.newaxis, :, :, :]
            start_time = time.time()
            pred = sess.run(y, feed_dict={
                x: img_batched
            })
            end_time = time.time()
            print('time:', end_time - start_time)   # jian's : 224: 0.018  448:0.05

            img = cv2.resize(deNormalizer(img) ,dsize=(img_w,img_h))
            pred = cv2.resize(pred[0,:,:,0],dsize=(img_w,img_h))
            print img.shape, pred.shape
            if True:
                vis_segmentation(img, mask, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pb_file',
        type=str,
        default='artifacts/Ms2.pb',
    )
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/images-256.npy',
        help='image file as numpy format'
    )
    args, _ = parser.parse_known_args()
    main(**vars(args))
