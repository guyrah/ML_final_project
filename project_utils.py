# coding: utf-8
import math
import os
from scipy import misc
import numpy
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import  Image, ImageEnhance
import subprocess
import shutil
from tensorflow.examples.tutorials.mnist import input_data


#signs_dict = ['א','ב','ג','ד','ה','ו','ז','ח','ט','י','כ','ל','מ','נ','ס','ע','פ','צ','ק','ר','ש','ת',' ','ך','ם','ן','ף','ץ','','','','','','','','','','','','','','',]
SIGNS_DICT = ['א','ב','ג','ד','ה','ו','ז','ח','ט','י','כ','ל','מ','נ','ס','ע','פ','צ','ק','ר','ש','ת',' ','ך','ם','ן','ף','ץ','(',')','+','-','∫','≤','≥','∞','∑','∩','≠','÷','×','0','1','2','3','4','5','6','7','8','9']
#SIGNS_DICT = ['א','ב','ג','ד','ה','ו','ז','ח','ט','י','כ','ל','מ','נ','ס','ע','פ','צ','ק','ר','ש','ת','ך','ם','ן','ף','ץ','(',')','+','-','∫','≤','≥','∞','∑','∩','≠','÷','×','0','1','2','3','4','5','6','7','8','9']

def get_dataset(dataset_path, max_classes=-1, max_files = 70):
    images = []
    labels = []
    classes = []
    examples_count = 0

    os.chdir(dataset_path)
    #min(len(os.listdir('.')), max_classes)

    dirs = os.listdir('.')
    if max_classes == -1:
        max_classes = len(dirs)

    numberOfClasses = min(len(dirs), max_classes)

    if dirs > 0:
        dirs = map(int, dirs)
        dirs.sort()
        dirs = map(str, dirs)
    else:
        raise NameError('No classes to get')

    for dir in dirs:
        if os.path.isdir(dir):
            os.chdir(dir)

            for files_count, file in enumerate(os.listdir('.')):
                if os.path.isfile(file):
                    #print (examples_count, int(dir))
                    labels.append([])
                    labels[examples_count].extend([0 for i in xrange(numberOfClasses)])
                    labels[examples_count][int(dir)] = 1
                    im_arr = misc.imread(file)
                    #im_arr = misc.imread(file, mode='YCbCr')
                    #images.append([])
                    #images[i].extend(im_arr)
                    images.append(im_arr)
                    examples_count += 1
                if files_count > max_files:
                    break

            os.chdir('..')
            classes.append(dir)
            if len(classes) == max_classes:
                break

    return numpy.array(images), numpy.array(labels), classes


def write_dataset_to_disk(data, labels, path):
    if not os.path.exists(path):
        os.makedirs(path)

    os.chdir(path)
    i = 0

    for label, image in zip(labels, data):
        label = str(numpy.argmax(label))
        if not os.path.exists(label):
            os.mkdir(label)

        os.chdir(label)
        try:
            misc.toimage(image, cmin=0, cmax=255).save(str(i) + '.jpg')
        except:
            print 1

        i += 1
        os.chdir('..')


def seperate_sets(images, labels, trainset_percent=70, testset_percent=20, validateset_percent=10):
    if trainset_percent + testset_percent + validateset_percent > 100:
        raise NameError('Total percents are over 100')
    else:
        dataset_size = len(images)

        trainset_size = int(math.floor(dataset_size * trainset_percent / 100))
        testset_size = int(math.floor(dataset_size * testset_percent / 100))
        validateset_size = int(math.floor(dataset_size * validateset_percent / 100))

        # Shuffles 2 Images to make sure the dataset in suffled
        images, labels = shuffle_2_arrays(images, labels)

        trainset_data = images[:trainset_size]
        trainset_labels = labels[:trainset_size]
        testset_data = images[trainset_size:trainset_size + testset_size]
        testset_labels = labels[trainset_size:trainset_size + testset_size]
        validateset_data = images[trainset_size + testset_size:trainset_size + testset_size + validateset_size]
        validateset_labels = labels[trainset_size + testset_size: trainset_size + testset_size + validateset_size]

        print 'Trainset size: {}'.format(len(trainset_data))
        print 'Testset size: {}'.format(len(testset_data))
        print 'Validateset size: {}'.format(len(validateset_data))

        return trainset_data, trainset_labels, testset_data, testset_labels, validateset_data, validateset_labels


def shuffle_2_arrays(array1, array2):
    perm = numpy.arange(len(array1))
    numpy.random.shuffle(perm)
    array1 = array1[perm]
    array2 = array2[perm]

    return array1, array2


def create_dataset_for_inception(srcPath, dstPath, image_size, do_image_to_gray, smooth_factor):
    images, labels, classes = get_dataset(srcPath)
    images = prepare_data(images, image_size=image_size, do_image_to_gray=do_image_to_gray, smooth_factor=smooth_factor)
    write_dataset_to_disk(images, labels, dstPath)


def create_datasets(srcPath, dstPath, image_size=(64,64)):
    images, labels, classes = get_dataset(srcPath)

    count = [0] * len(classes)
    trainset_data, trainset_labels, testset_data, testset_labels, validateset_data, validateset_labels = seperate_sets(images,labels)
    #trainset_data = prepare_data(trainset_data, image_size, do_image_to_gray=True, do_scale_image_values=True)
    #testset_data = prepare_data(testset_data, image_size, do_image_to_gray=True, do_scale_image_values=True)
    #validateset_data = prepare_data(validateset_data, image_size, do_image_to_gray=True, do_scale_image_values=True)
    write_dataset_to_disk(trainset_data, trainset_labels, dstPath + '/Trainset');
    write_dataset_to_disk(testset_data, testset_labels, dstPath + '/Testset');
    write_dataset_to_disk(validateset_data, validateset_labels, dstPath + '/Validateset');
    print "train set Diversification: {}".format(numpy.sum(trainset_labels, axis=1))
    print "test set Diversification: {}".format(numpy.sum(testset_labels, axis=1))
    print "validation set Diversification: {}".format(numpy.sum(validateset_labels, axis=1))


def prepare_data(images, image_size=(-1,-1), do_scale_image_values=False, do_image_to_gray=False, smooth_factor=1):
    resultSet = [[] for i in xrange(len(images))]

    for i in xrange(len(images)):
        if not image_size == (-1,-1):
            resultSet[i] = numpy.array(scale_image(images[i], image_size))
        else:
            resultSet[i] = numpy.array(images[i])

        if do_image_to_gray:
            resultSet[i] = image_to_grey(resultSet[i])


        if not smooth_factor == 1:
            resultSet[i] = smooth_image(resultSet[i], smooth_factor)

        if do_scale_image_values:
            resultSet[i] = scale_image_values(numpy.array(resultSet[i], dtype='float'))
        #misc.imshow(images[i])
        #tmp_image = scale_image(images[i], image_size)
        #images[i] = scale_image(images[i], image_size)
        #images[i] = []
        #images[i].extend(tmp_image)
        #tmp_image = scale_image(images[i], image_size)
        #images[i] = image_to_grey(images[i])

        #images = flatten_matrix_with_first_dim_kept(images)

    #return numpy.array(resultSet).reshape(len(images), -1)
    return resultSet


def scale_image_values(img):
    return img/255


def flatten_matrix_with_first_dim_kept(mat):
    mat = numpy.array(mat)
    data_size = 1
    #data_size = reduce(lambda data_size, y: data_size*y, mat.shape[1:])
    #new_shape = tuple([len(mat), data_size])

    #mat = mat.reshape(new_shape)
    mat = mat.reshape(-1, 28,28,1)

    return mat


def image_to_grey(image):
    # If grey scale already
    if len(image.shape) == 2:
        #misc.imshow(image)

        return image
    # Assume RGB
    elif len(image.shape) == 3:
    #    misc.imshow(image)
    #    misc.imshow(numpy.dot(image[...,:3], [0.299, 0.587, 0.114]))
        return numpy.array(Image.fromarray(image).convert('L'))
        #return numpy.dot(image[...,:3], [0.299, 0.587, 0.114])
    # Do best and return 1 layer
    else:
        return image[:, :, 0]





def show_image(image):

    misc.imshow(image.reshape(28,28,-1))


def train(trainData, trainLabels, validateData, validateLabels, testData, testLabels, inputSize, numOfClasses, model_path, save_model, load_model):
    if save_model:
        cyclic_names = ['1', '2', '3']
        current_name_index = 0
    trainData = numpy.array(trainData).reshape(-1, 4096)
    testData = numpy.array(testData).reshape(-1, 4096)
    validateData = numpy.array(validateData).reshape(-1, 4096)

    # We start
    best_loss = 0
    n_input = 1
    for i in inputSize:
        n_input *= i

    n_output = numOfClasses

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])

    x_tensor = tf.reshape(x, [-1, inputSize[0], inputSize[1], 1])

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

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
    h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1) + b_conv1, name='conv1')
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='conv2')
    h_pool2 = max_pool_2x2(h_conv2)

    # 7x7 is the size of the image after the convolutional and pooling layers (28x28 -> 14x14 -> 7x7)
    h_conv2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * n_filters_2])

    # %% Create the one fully-connected layer:
    n_fc = 1024
    W_fc1 = weight_variable([16 * 16 * n_filters_2, n_fc])
    b_fc1 = bias_variable([n_fc])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([n_fc, n_output])
    b_fc2 = bias_variable([n_output])
    y_pred = tf.matmul(h_fc1_drop, W_fc2, name='final_result') + b_fc2

    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    # optimizer = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # variables:
    saver = tf.train.Saver()
    sess = tf.Session()
    #tf.identity(sess.graph.get_tensor_by_name('final_result:0'), 'final_result')
    if load_model:
        saver.restore(sess, model_path + '/MNIST/trained/my_model')
    else:
        sess.run(tf.global_variables_initializer())

    # We'll train in minibatches and report accuracy:
    batch_size = min(20, len(trainData))
    n_epochs = 100000
    l_loss = list()
    f = open(model_path + '/loss', 'a')

    for epoch_i in xrange(n_epochs):
        trainData, trainLabels = shuffle_2_arrays(trainData, trainLabels)
        batch_xs, batch_ys = trainData[:batch_size], trainLabels[:batch_size]

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})

        loss = sess.run(accuracy, feed_dict={
            x: validateData,
            y: validateLabels,
            keep_prob: 1.0})
        print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))

        if epoch_i % 10 == 0:
            f.write(str(loss))
            f.flush()

        if loss > best_loss and save_model:
            best_loss = loss
            saver.save(sess, model_path + '/MNIST/saved_model' + cyclic_names[current_name_index] + '/my_model')
            current_name_index = (current_name_index + 1) % len(cyclic_names)

    print("Accuracy for test set: {}".format(sess.run(accuracy,
                                                      feed_dict={
                                                          x: testData,
                                                          y: testLabels,
                                                          keep_prob: 1.0
                                                      })))

    f.close()


def create_graph(modelFullPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def predict_no_labels(imagesPath, labelsFullPath, modelFullPath):
    tmp_path = imagesPath
    '''
    tmp_path = '/home/osboxes/PycharmProjects/ML_final_project/workspace'

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
        os.mkdir(tmp_path)
    else:
        os.mkdir(tmp_path)

    create_dataset_for_inception(imagesPath, tmp_path)
    '''
    answer = None
    isdir = False
    if not os.path.exists(tmp_path):
        raise NameError('Path does not exist')
    else:
        if os.path.isdir(tmp_path):
            os.chdir(tmp_path)
            tmp_path = os.listdir(tmp_path)
            tmp_path.sort()
            #imagesPath = map()
        elif os.path.isfile(tmp_path):
            tmp_path = [tmp_path]

    # Creates graph from saved GraphDef.
    create_graph(modelFullPath)

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        for imagePath in tmp_path:
            image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = numpy.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
            '''
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            '''

            print imagePath, SIGNS_DICT[int(labels[top_k[0]])]
            #print imagePath, SIGNS_DICT[labels[top_k[0]]]


    shutil.rmtree(tmp_path)

def smooth_image(image, smooth_factor):
    #misc.imshow(image)
    tmp_image = ndimage.filters.gaussian_filter(image, sigma=smooth_factor)
    #misc.imshow(image)
    return tmp_image

def retrain_inception_options():
    srcDataset = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/FinalDataset/'
    dstDataSetPath = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/'
    numberOfSteps = ' --how_many_training_steps=16000'
    imagesdir = ' --image_dir=' + dstDataSetPath
    output_graph = ' --output_graph=/home/osboxes/PycharmProjects/ML_final_project/Models/'
    output_labels = ' --output_labels=/home/osboxes/PycharmProjects/ML_final_project/Models/'
    config_path = '/home/osboxes/PycharmProjects/ML_final_project/configs/'
    summaries_dir = ' --summaries_dir=/home/osboxes/PycharmProjects/ML_final_project/Summaries/'
    bottleneck_dir = ' --bottleneck_dir=/home/osboxes/PycharmProjects/ML_final_project/tmp/'
    random_brightness = ' --random_brightness='
    retrain_script_path = '/home/osboxes/PycharmProjects/ML_final_project/retrain.py'


    #image_size_options = [(-1,-1), (28,28), (128,128)]
    image_size_options = [(28, 28)]
    #do_image_to_gray_options = [False, True]
    do_image_to_gray_options = [False]
    #smooth_factor_options = [1,2]
    smooth_factor_options = [1]
    #random_brightness_options = [0,20, 30]
    random_brightness_options = [0]

    counter = 0

    for image_size in image_size_options:
        for do_image_to_gray in do_image_to_gray_options:
            for smooth_factor in smooth_factor_options:
                create_dataset_for_inception(srcDataset, dstDataSetPath + str(counter), image_size,
                                             do_image_to_gray, smooth_factor)

                for j, random_bright in enumerate(random_brightness_options):
                    f = open(config_path + str(counter), 'w')

                    f.write('image_size: ' + str(image_size) + '\n' +
                            'do_image_to_gray: ' + str(do_image_to_gray) + '\n' +
                            'smooth_factor: ' + str(smooth_factor) + '\n' +
                            'random_bright: ' + str(random_bright) + '\n')
                    f.close()

                    command = 'python ' + retrain_script_path + numberOfSteps + imagesdir +str(counter) + \
                              output_graph + str(counter)+ '.' + str(j) + '.pb' + \
                    output_labels + str(counter)+ '.' + str(j) + '.txt' + \
                    summaries_dir + str(counter)+ '.' + str(j) + \
                    bottleneck_dir + str(counter)+ '.' + str(j) + \
                    random_brightness + str(random_bright)
                    print command
                    #os.system(command=command)


                    #subprocess.call(['pwd'])
                    subprocess.call(command.split())
                    print 'finished'
                counter += 1

def load_image(self, imagepath, randrange):
    im = Image.open(imagepath)
    randnum = random.uniform(-1 * (5 * randrange), (5 * randrange))
    im = im.rotate(randnum, resample=Image.BICUBIC, expand=False)
    contr = ImageEnhance.Contrast(im)
    randnum = random.uniform(1 - (randrange / 2), 1 + (randrange / 2))
    im = contr.enhance(randnum)
    bright = ImageEnhance.Brightness(im)
    randnum = random.uniform(1 - (randrange / 2), 1 + (randrange / 2))
    im = bright.enhance(randnum)
    if random.randint(int(2 - randrange), 2) == 1:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return im


def predict_with_labels(images_path, labels, labelsFullPath, modelFullPath, is_inception):
    sess = tf.Session()
    #confusion_matrix = numpy.zeros((len(labels), len(labels)), dtype='int')

    if is_inception:
        create_graph(modelFullPath)
        y_pred = softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels_dict = [str(w).replace("\n", "") for w in lines]
        f.close()

        for current_dir in os.listdir(images_path):
            for current_image_name in os.listdir(str('{}/{}').format(images_path, current_dir)):
                current_image_path = str('{}/{}/{}').format(images_path, current_dir, current_image_name)
                image_data = tf.gfile.FastGFile(current_image_path, 'rb').read()

                predictions = sess.run(softmax_tensor,
                                       {'DecodeJpeg/contents:0': image_data})
                predictions = numpy.squeeze(predictions)

                top_k = predictions.argsort()[-2:][::-1]  # Getting top 5 predictions

                for node_id in top_k:
                    human_string = labels[node_id]
                    score = predictions[node_id]
                    print('%s (score = %.5f)' % (human_string, score))

                answer = labels[top_k[0]]
                return answer

    else:
        saver = tf.train.Saver()
        saver.restore(sess, modelFullPath)
        y_pred = sess.graph.get_tensor_by_name('Variable_7:0')
        x = sess.graph.get_operation_by_name('Placeholder_0')
        keep_prob = sess.graph.get_operation_by_name('Placeholder_2')

    results = sess.run(y_pred, feed_dict={
        x: images, keep_prob: 1.0})

    for i in xrange(results):
        confusion_matrix[labels[i], tf.argmax(results[i])] += 1





def train_mnist():
    cyclic_names = ['1', '2']
    current_name_index = 0
    model_path = '/home/osboxes/PycharmProjects/ML_final_project/Models/MNIST'
    load_model = False
    save_model = True
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    inputSize = (28,28)
    trainData = mnist.train.images
    trainLabels = mnist.train.labels
    testData = mnist.test.images
    testLabels = mnist.test.labels
    validateData = mnist.validation.images
    validateLabels = mnist.validation.labels

    for i in range(len(trainData)):
        misc.imshow(trainData[i].reshape((28,28)))

    # We start
    best_loss = 0
    n_input = 1
    for i in inputSize:
        n_input *= i

    n_output = 10

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])

    x_tensor = tf.reshape(x, [-1, inputSize[0], inputSize[1], 1])

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

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
    h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1) + b_conv1, name='conv1')
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='conv2')
    h_pool2 = max_pool_2x2(h_conv2)

    # 7x7 is the size of the image after the convolutional and pooling layers (28x28 -> 14x14 -> 7x7)
    h_conv2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * n_filters_2])

    # %% Create the one fully-connected layer:
    n_fc = 1024
    W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
    b_fc1 = bias_variable([n_fc])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([n_fc, n_output])
    b_fc2 = bias_variable([n_output])
    y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    # optimizer = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # variables:
    saver = tf.train.Saver()
    sess = tf.Session()

    if load_model:
        saver.restore(sess, model_path + '/saved_model1/my_model')
    else:
        sess.run(tf.global_variables_initializer())

    # We'll train in minibatches and report accuracy:
    batch_size = min(50, len(trainData))
    n_epochs = 10000
    l_loss = list()
    for epoch_i in xrange(n_epochs):
        trainData, trainLabels = shuffle_2_arrays(trainData, trainLabels)
        batch_xs, batch_ys = trainData[:batch_size], trainLabels[:batch_size]

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})

        loss = sess.run(accuracy, feed_dict={
            x: validateData,
            y: validateLabels,
            keep_prob: 1.0})
        print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))

        if epoch_i % 10 == 0:
            l_loss.append(loss)

        if loss > best_loss and save_model:
            best_loss = loss
            saver.save(sess, model_path + '/saved_model' + cyclic_names[current_name_index] + '/my_model')
            current_name_index = (current_name_index + 1) % len(cyclic_names)

    '''
    plt.title('CNN Acuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(l_loss, color='m')
    plt.show()
'''
    print("Accuracy for test set: {}".format(sess.run(accuracy,
                                                      feed_dict={
                                                          x: testData,
                                                          y: testLabels,
                                                          keep_prob: 1.0
                                                      })))

    f = open(model_path + '/loss', 'w+')

    for i in l_loss:
        f.write(str(i) + '\n')

    f.close()
