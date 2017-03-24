import subprocess
import tensorflow as tf
import os
import random
import math
import numpy as np
from PIL import Image

def retrain_inception():
    numberOfSteps = ' --how_many_training_steps=150000'
    imagesdir = ' --image_dir=/home/osboxes/Desktop/ML/Final_Project/Datasets/Fullset_prepared'
    output_graph = ' --output_graph=/home/osboxes/PycharmProjects/ML_final_project/Models/inception_retrained.pb'
    output_labels = ' --output_labels=/home/osboxes/PycharmProjects/ML_final_project/Models/inception_retrained.txt'
    summaries_dir = ' --summaries_dir=/home/osboxes/PycharmProjects/ML_final_project/Summaries/inception_retrained'
    bottleneck_dir = ' --bottleneck_dir=/home/osboxes/Desktop/ML/Final_Project/bottlenecks/inception_retrained'
    retrain_script_path = '/home/osboxes/PycharmProjects/ML_final_project/retrain.py'
    '''
    numberOfSteps = ' --how_many_training_steps=16000'
    imagesdir = ' --image_dir=/home/osboxes/PycharmProjects/ML_final_project/DataBase/4'
    output_graph = ' --output_graph=/home/osboxes/PycharmProjects/ML_final_project/Models/inception_retrained.pb'
    output_labels = ' --output_labels=/home/osboxes/PycharmProjects/ML_final_project/Models/inception_retrained.txt'
    summaries_dir = ' --summaries_dir=/home/osboxes/PycharmProjects/ML_final_project/Summaries/inception_retrained'
    bottleneck_dir = ' --bottleneck_dir=/home/osboxes/PycharmProjects/ML_final_project/tmp/4.0'
    retrain_script_path = '/home/osboxes/PycharmProjects/ML_final_project/retrain.py'
    '''

    command = 'python ' + \
              retrain_script_path + \
              numberOfSteps + \
              imagesdir + \
              output_graph + \
              output_labels + \
              summaries_dir + \
              bottleneck_dir
    print command

    subprocess.call(command.split())
    print 'finished'


def build_mnist_graph(number_of_classes):
    image_shape = (28,28)
    graph_input_shape = tuple([-1] + [i for i in image_shape] + [1])
    input_size = 1
    for i in image_shape:
        input_size *= i

    x = tf.placeholder(tf.float32, [None, input_size], name='input_placeholder')
    y = tf.placeholder(tf.float32, [None, number_of_classes], name='ground_truth_placeholder')
    x_normalized = tf.mul(x, 1.0 / 255, name='values_scaling')

    x_tensor = tf.reshape(x_normalized, graph_input_shape, name='image_reshape')

    def conv2d(x, W, name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

    def max_pool_2x2(x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    # functions for parameter initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Weight matrix is [height x width x input_channels x output_channels]
    # Bias is [output_channels]
    filter_size = 5
    n_filters_1 = 32
    n_filters_2 = 64

    # parameters
    W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])
    b_conv1 = bias_variable([n_filters_1])
    W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
    b_conv2 = bias_variable([n_filters_2])

    # layers
    h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1, name='conv1') + b_conv1, name='conv1relu')
    h_pool1 = max_pool_2x2(h_conv1, name='max_pooling1')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name='conv2') + b_conv2, name='conv1relu')
    h_pool2 = max_pool_2x2(h_conv2, name='max_pooling2')

    # 7x7 is the size of the image after the convolutional and pooling layers (28x28 -> 14x14 -> 7x7)
    h_conv2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * n_filters_2], name='flatten_input1')

    # %% Create the one fully-connected layer:
    n_fc = 1024
    W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
    b_fc1 = bias_variable([n_fc])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name='fc1')

    keep_prob = tf.placeholder(tf.float32, name='dropout_prob1')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout1')

    W_fc2 = weight_variable([n_fc, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])
    y_pred = tf.matmul(h_fc1_drop, W_fc2, name='final_result') + b_fc2

    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    tf.summary.scalar('cross_entropy', cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy, name='train')

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', accuracy)


def add_prefix_to_elements(arr, prefix):
    return [prefix + i for i in arr]



def shuffle_2_arrays(array1, array2):
    perm = np.arange(len(array1))
    np.random.shuffle(perm)
    array1 = array1[perm]
    array2 = array2[perm]

    return array1, array2


def load_pathes(pathes, image_size, f_log, max_files=-1):
    images = list()

    for i, path in enumerate(pathes):
        if max_files == i:
            break

        img = np.array(Image.open(path), dtype='float')
        if img.shape != image_size:
            f_log.write('image: {} shape is: {} instead of expected: {}{}'.format(path, img.shape, image_size, '\n'))
            f_log.flush()
        else:
            images.append(img)
        #img = img /255


    image_num_of_bytes = 1
    for i in image_size:
        image_num_of_bytes *= i

    new_shape = tuple([-1, image_num_of_bytes])

    return np.array(images).reshape(new_shape)


def labels_to_onehotvector(labels, classes):
    labels_vector = np.zeros((len(labels), len(classes)), dtype='int')
    for i, label in enumerate(labels):
        labels_vector[i, int(label)] = 1

    return labels_vector


def train(images_path,
          image_size,
          train_percent=85,
          validate_percent=10,
          test_percent=5,
          src_model_path='',
          src_labels_path='',
          summaries_path='',
          output_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1'):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create set's pathes list
    train_images_path = []
    trainset_labels = []
    validate_images_path = []
    validateset_labels = []
    test_images_path = []
    testset_labels = []
    classes = []

    dirs = os.listdir(images_path)
    dirs = map(int, dirs)
    dirs.sort()
    dirs = map(str, dirs)

    for dir in dirs:
        classes.append(dir)
        images = os.listdir(images_path + '/' + dir)
        images = add_prefix_to_elements(images, images_path + '/' + dir + '/')
        total_images_count = len(images)
        labels = [dir for i in xrange(total_images_count)]
        trainset_size = int(math.floor(total_images_count * train_percent / 100))
        validateset_size = int(math.floor(total_images_count * validate_percent / 100))
        testset_size = int(math.floor(total_images_count * test_percent / 100))

        random.shuffle(images)
        train_images_path.extend(images[:trainset_size])
        trainset_labels.extend(labels[:trainset_size])
        validate_images_path.extend(images[trainset_size:trainset_size + validateset_size])
        validateset_labels.extend(labels[trainset_size:trainset_size + validateset_size])
        test_images_path.extend(images[trainset_size + validateset_size:])
        testset_labels.extend(labels[trainset_size + validateset_size:])


    if len(src_labels_path) > 0:
        f = open(src_labels_path, 'r')
        lines = f.readlines()
        classes = [str(w).replace("\n", "") for w in lines]
    else:
        f = open(output_path + '/' + 'classes.txt', 'w')
        for i in classes:
            f.write(i + '\n')
        f.close()

    number_of_classes = len(classes)


    train_images_path = np.array(train_images_path, dtype='str')
    validate_images_path = np.array(validate_images_path, dtype='str')
    test_images_path = np.array(test_images_path, dtype='str')

    trainset_labels = labels_to_onehotvector(trainset_labels, classes)
    validateset_labels = labels_to_onehotvector(validateset_labels, classes)
    testset_labels = labels_to_onehotvector(testset_labels, classes)

    # sets image shape
    #image_shape = (1)

    image_shape = (28,28)
    graph_input_shape = tuple([-1] + [i for i in image_shape] + [1])
    input_size = 1
    for i in image_shape:
        input_size *= i

    build_mnist_graph(number_of_classes)

    saver = tf.train.Saver()
    sess = tf.Session()

    # Builds graph - either load or build new one
    # Create new graph
    if len(src_model_path) == 0:
        sess.run(tf.global_variables_initializer())
    # Load model
    else:
        saver.restore(sess, src_model_path)
        '''
        if os.path.exists(src_model_path):
            saver.restore(sess, src_model_path)
        else:
            raise NameError('source model does not exist')
        '''


    # Get input, labels and result variables
    x = sess.graph.get_tensor_by_name('input_placeholder:0')
    y = sess.graph.get_tensor_by_name('ground_truth_placeholder:0')
    results = sess.graph.get_tensor_by_name('final_result:0')
    optimizer = sess.graph.get_operation_by_name('train')
    accuracy = sess.graph.get_tensor_by_name('accuracy:0')
    keep_prob = sess.graph.get_tensor_by_name('dropout_prob1:0')


    # Set training relevant variables
    train_batch_size = min(100, len(train_images_path))
    validate_batch_size = min(100, len(validate_images_path))
    test_batch_size = min(20, len(test_images_path))
    n_epochs = 100000
    l_loss = list()
    best_loss = 0

    #f = open(output_path + '/loss.txt', 'a')
    f_log = open(output_path + '/' + 'error_log.txt', 'w')
    #f_full_validation = open(output_path + '/full_validation_loss.txt', 'a')
    cyclic_names = ['1', '2']
    current_name_index = 0

    merged = tf.summary.merge_all()

    if len(summaries_path) > 0:
        if not os.path.exists(summaries_path + '/train'):
            os.makedirs(summaries_path + '/train')
        if not os.path.exists(summaries_path + '/validation'):
            os.makedirs(summaries_path + '/validation')

        train_writer = tf.summary.FileWriter(summaries_path + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(summaries_path + '/validation')

    for epoch_i in xrange(n_epochs):
        train_images_path, trainset_labels = shuffle_2_arrays(train_images_path, trainset_labels)
        train_batch_xs_pathes, train_batch_ys = train_images_path[:train_batch_size], trainset_labels[:train_batch_size]
        train_batch_xs = load_pathes(train_batch_xs_pathes, image_size=image_size, f_log=f_log)

        summary, _ = sess.run([merged, optimizer], feed_dict={
            x: train_batch_xs,
            y: train_batch_ys,
            keep_prob: 0.5})

        train_writer.add_summary(summary, epoch_i)


        if best_loss > 0.85:
            validate_batch_size = min(1000, len(validate_images_path))

        validate_images_path, validateset_labels = shuffle_2_arrays(validate_images_path, validateset_labels)
        validate_batch_xs_pathes, validate_batch_ys = validate_images_path[:validate_batch_size], validateset_labels[:validate_batch_size]
        validate_batch_xs = load_pathes(validate_batch_xs_pathes, image_size=image_size, f_log=f_log)

        summary, loss = sess.run([merged, accuracy], feed_dict={
            x: validate_batch_xs,
            y: validate_batch_ys,
            keep_prob: 1.0})
        validation_writer.add_summary(summary, epoch_i)

        print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))

        if loss > best_loss:
            best_loss = loss
            saver.save(sess, output_path + '/model' + cyclic_names[current_name_index] + '/my_model')
            current_name_index = (current_name_index + 1) % len(cyclic_names)

    '''
    print("Accuracy for test set: {}".format(sess.run(accuracy,
                                                      feed_dict={
                                                          x: testData,
                                                          y: testLabels,
                                                          keep_prob: 1.0
                                                      })))
    '''
    #f.close()
    #f_full_validation.close()
    f_log.close()
