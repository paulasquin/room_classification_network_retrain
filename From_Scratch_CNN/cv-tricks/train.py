import dataset
import tensorflow as tf
import math
import random
import numpy as np
import os
import sys

sys.path.append('../../')
from tools import *

# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

# Free not allocated memory
import gc

gc.collect()

# Hide useless tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Model parameters ===
NUM_ITERATION = 3000
BATCH_SIZE = 32
VALIDATION_PERCENTAGE = 0.2  # 20% of the data will automatically be used for validation
LEARNING_RATE = 1e-4
SHORTER_DATASET_VALUE = 1900
IMG_SIZE = 500
DATASET_TYPE = np.float16
NUM_CHANNELS = 1
# SHORTER_DATASET_VALUE = 2000
DATASET_PATH = '../../JPG_Scannet_Aug'
DATASET_SAVE_DIR_PATH = os.getcwd() + "/" + DATASET_PATH.split("/")[-1].lower()

# Network graph params
LES_NUM_FILTERS_CONV = [64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
LES_CONV_FILTER_SIZE = [3] * len(LES_NUM_FILTERS_CONV)
FC_LAYER_SIZE = 64

EXPORTS_DIR_PATH = '/media/nas/ScanNet/From_Scratch_CNN/cv-tricks/exports/'
createFolder(EXPORTS_DIR_PATH)
EXPORTNUM_DIR_PATH = EXPORTS_DIR_PATH + "export_" + str(getExportNumber(EXPORTS_DIR_PATH))
createFolder(EXPORTNUM_DIR_PATH)
INFO_TXT_PATH = EXPORTNUM_DIR_PATH + "/info.txt"
CSV_TRAIN = EXPORTNUM_DIR_PATH + "/train.csv"

if len(LES_CONV_FILTER_SIZE) != len(LES_NUM_FILTERS_CONV):
    print("Convolutional layers params aren't the same length")
    raise

# === Global variables ===
g_total_iterations = 0

""" Default Model :
    filter_size_conv1 = 3
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64

    fc_layer_size = 128
"""


class ConvolutionLayer:
    inputt = 0
    num_input_channels = 0
    conv_filter_size = 0
    num_filters = 0
    layer = 0

    def __init__(self, inputt, num_input_channels, conv_filter_size, num_filters):
        self.inputt = inputt
        self.num_input_channels = num_input_channels
        self.conv_filter_size = conv_filter_size
        self.num_filters = num_filters
        self.layer = create_convolutional_layer(
            input=inputt,
            num_input_channels=num_input_channels,
            conv_filter_size=conv_filter_size,
            num_filters=num_filters
        )


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # We shall define the weights that will be trained using create_weights function.

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels.
    # But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, session, accuracy, i, milestone=False):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    prefix = ""
    if milestone:
        prefix = "\t"

    with open(CSV_TRAIN, 'a') as f:
        txt = str(i) + "\t" + str(epoch + 1) + "\t" + str(acc) + "\t" + str(val_acc) + "\t" + str(val_loss) + "\n"
        f.write(txt.replace('.', ','))

    print(prefix + msg.format(epoch + 1, acc, val_acc, val_loss))


def train(num_iteration, session, data, cost, saver, accuracy, optimizer, x, y_true):
    global g_total_iterations

    for i in range(g_total_iterations,
                   g_total_iterations + num_iteration):

        print("\t" + str(i) + " : [" + str(g_total_iterations) + ", " + str(
            g_total_iterations + num_iteration) + "]. Save every " + str(int(data.train.num_examples / BATCH_SIZE)))

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(BATCH_SIZE)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(BATCH_SIZE)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / BATCH_SIZE) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / BATCH_SIZE))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, session=session, accuracy=accuracy, i=i)
            saver.save(session, EXPORTNUM_DIR_PATH + "/model")

        elif i % 5 == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / BATCH_SIZE))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, session=session, accuracy=accuracy, i=i,
                          milestone=True)

    g_total_iterations += num_iteration


def init():
    with open(INFO_TXT_PATH, 'w') as f:
        txt = "\nBATCH_SIZE :" + str(BATCH_SIZE) + \
              "\nLEARNING_RATE : " + str(LEARNING_RATE) + \
              "\nSHORTER_DATASET_VALUE : " + str(SHORTER_DATASET_VALUE) + \
              "\nDATASET_PATH : " + str(DATASET_PATH) + \
              "\nLES_CONV_FILTER_SIZE : " + str(LES_CONV_FILTER_SIZE) + \
              "\nLES_NUM_FILTERS_CONV : " + str(LES_NUM_FILTERS_CONV) + \
              "\nFC_LAYER_SIZE : " + str(FC_LAYER_SIZE)
        f.write(txt)
    with open(CSV_TRAIN, 'w') as f:
        f.write("Iteration\tEpoch\tTraining Accuracy\tValidation Accuracy\tValidation Loss\n")


def main():
    init()
    session = tf.Session()
    # Prepare input data
    classes = os.listdir(DATASET_PATH)
    i = 0
    while i < len(classes):
        if "." in classes[i]:
            classes.pop(i)
        else:
            i += 1
    print(classes)
    num_classes = len(classes)

    data = dataset.read_train_sets(
        DATASET_PATH,
        IMG_SIZE,
        classes,
        validation_size=VALIDATION_PERCENTAGE,
        shorter=SHORTER_DATASET_VALUE,
        num_channels=NUM_CHANNELS,
        dataset_save_dir_path=DATASET_SAVE_DIR_PATH,
        dataset_type=DATASET_TYPE
    )


    print("Complete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name='x')

    # labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    lesLayers = []
    # Adding Convolutional layers
    for i in range(len(LES_NUM_FILTERS_CONV)):
        if i == 0:
            inputt = x
            num_input_channels = NUM_CHANNELS
        else:
            inputt = lesLayers[-1].layer
            num_input_channels = lesLayers[-1].num_filters

        lesLayers.append(ConvolutionLayer(inputt, num_input_channels, LES_CONV_FILTER_SIZE[i], LES_NUM_FILTERS_CONV[i]))

    # Adding flatten layer
    lesLayers.append(create_flatten_layer(lesLayers[-1].layer))

    # Adding fully connected layers
    lesLayers.append(
        create_fc_layer(
            input=lesLayers[-1],
            num_inputs=lesLayers[-1].get_shape()[1:4].num_elements(),
            num_outputs=FC_LAYER_SIZE,
            use_relu=True
        )
    )

    lesLayers.append(
        create_fc_layer(
            input=lesLayers[-1],
            num_inputs=FC_LAYER_SIZE,
            num_outputs=num_classes,
            use_relu=False
        )
    )

    y_pred = tf.nn.softmax(lesLayers[-1], name='y_pred')
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    session.run(tf.global_variables_initializer())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=lesLayers[-1],
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    train(
        num_iteration=NUM_ITERATION,
        session=session,
        data=data,
        cost=cost,
        saver=saver,
        accuracy=accuracy,
        optimizer=optimizer,
        x=x,
        y_true=y_true
    )
    session.close()
    gc.collect()  # Free not allocated memory


if __name__ == '__main__':
    main()
