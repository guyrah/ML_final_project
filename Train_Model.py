import datetime
import random
from sklearn.externals import joblib
import math
import os
from scipy import misc
from scipy import ndimage
from scipy import signal
import numpy
from sklearn.svm import SVC

def get_dataset(dataset_path, max_classes=-1):
    images = []
    labels = []
    classes = []

    os.chdir(dataset_path)

    for dir in os.listdir('.'):
        if os.path.isdir(dir):
            os.chdir(dir)

            for file in os.listdir('.'):
                if os.path.isfile(file):
                    labels.append(dir)
                    im_arr = misc.imread(file)
                    #im_arr = misc.imread(file, mode='YCbCr')
                    images.append(im_arr)

            os.chdir('..')
            classes.append(dir)
            if len(classes) == max_classes:
                break

    return images, labels, classes


def feature_extraction(images, image_size=(255,255), smooth_factor=2, pixel_threshold=200):
    features = [[] for _ in range(len(images))]
    c=0
    for i in xrange(len(images)):
        #misc.imshow(images[i])

        images[i] = image_to_grey(images[i])
        rmin, rmax, cmin, cmax = bounding_box(images[i])
        trimmed_img = images[i][rmin:rmax, cmin:cmax]
        thresholded_img = thresholding_image(trimmed_img, 160)


        #thresholded_img = smooth_image(thresholded_img, smooth_factor)
        #misc.imshow(trimmed_img)


        images[i] = scale_image(thresholded_img, image_size)
        #misc.imshow(images[i])
        #images[i] = smooth_image(images[i], smooth_factor)
        #misc.imshow(images[i])
        '''
        images[i] = scale_image(images[i], image_size)
        images[i] = image_to_grey(images[i])
        images[i] = smooth_image(images[i], smooth_factor)

        #images[i] = thresholding_image(images[i], pixel_threshold)
        #misc.imshow(images[i])
        '''
        # Features
        #print hog_image(images[i]).shape
        features[i].extend(count_regions(images[i], 3, 3))
        #features[i].extend(hog_image(images[i]))
        features[i].extend(diagonal_sum(images[i]))
        #features[i].extend(images[i].flatten())
        features[i].extend([rmin, rmax, cmin, cmax])
        #rmin, rmax, cmin, cmax =



    features = numpy.array(features)

    print len(features)
    print features.shape
    return features

def smooth_image(image, smooth_factor):
    #misc.imshow(image)
    tmp_image = ndimage.filters.gaussian_filter(image, sigma=smooth_factor)
    #misc.imshow(image)
    return tmp_image


#def count_pixels:



def thresholding_image(image, threshold=-1):
    #misc.imshow(image)
    if threshold == -1:
        threshold = (image.max() + image.min()) / 2

    tmp_image = numpy.copy(image)

    idx = tmp_image[:] > threshold
    tmp_image[idx] = 255
    idx = tmp_image[:] < threshold
    tmp_image[idx] = 0
    #misc.imshow(image)

    return tmp_image


def hog_image(image):
    kernel = [[1, 0, -1]]
    kernel2 = [[1], [0], [-1]]
    kernel3 = [[0,0,1],[0,0,0],[-1,0,0]]
    kernel4 = [[1, 0, 0], [0, 0, 0], [0, 0, -1]]

    if len(image.shape) > 2:
        raise NameError("HOG got object with dimension bigger than 2")
    else:
        #misc.imshow(image)
        hog_matrix = numpy.absolute(signal.convolve2d(image, kernel, mode='same', boundary='wrap'))
        hog_matrix += numpy.absolute(signal.convolve2d(image, kernel2, mode='same', boundary='wrap'))
        hog_matrix += numpy.absolute(signal.convolve2d(image, kernel3, mode='same', boundary='wrap'))
        hog_matrix += numpy.absolute(signal.convolve2d(image, kernel4, mode='same', boundary='wrap'))
        #print hog_matrix

        #thresholding_image(hog_matrix)
        #misc.imshow(hog_matrix)

        return hog_matrix.flatten()


def scale_image(image, size):
    return misc.imresize(image, size)


def count_regions(image, rows, cols):
    if len(image.shape) > 2:
        raise NameError("count_regions got object with dimension bigger than 2")
    else:
        tmp_image = thresholding_image(image)
        results = []
        ver_segment_size = int(math.floor(tmp_image.shape[0] / rows))
        hor_segments_size = int(math.floor(tmp_image.shape[1]/cols))

        for i in xrange(rows):
            results.extend([numpy.count_nonzero(tmp_image[ver_segment_size * i: ver_segment_size * (i + 1), :])])

        for i in xrange(cols):
            results.extend([numpy.count_nonzero(tmp_image[hor_segments_size * i: hor_segments_size * (i + 1), :])])

        return results
'''
        for i in xrange(rows):
            for j in xrange(cols):
                col_start_index = hor_segments_size * j
                col_end_index = hor_segments_size * (j + 1)
                row_start_index = ver_segment_size * i
                row_end_index = ver_segment_size * (i + 1)

                if j == cols - 1:
                    col_end_index = image.shape[1] - 1
                if i == rows - 1:
                    #row_start_index = image.shape[0] - hor_segments_size
                    row_end_index = image.shape[0] - 1

                #print row_start_index, row_end_index, col_start_index, col_end_index
                #print numpy.count_nonzero(image[row_start_index:row_end_index,col_start_index:col_end_index])
                results.extend([numpy.count_nonzero(image[row_start_index:row_end_index,col_start_index:col_end_index])])
        #print results
'''






def image_to_grey(image):
    # If grey scale already
    if len(image.shape) == 2:
        misc.imshow(image)

        return image
    # Assume RGB
    elif len(image.shape) == 3:
    #    misc.imshow(image)
    #    misc.imshow(numpy.dot(image[...,:3], [0.299, 0.587, 0.114]))
        return numpy.dot(image[...,:3], [0.299, 0.587, 0.114])
    # Do best and return 1 layer
    else:
        return image[:, :, 0]


def diagonal_sum(image):
    if len(image.shape) > 2:
        raise NameError("diagonal_sum got object with dimension bigger than 2")
    else:
        return [numpy.trace(image), numpy.trace(numpy.transpose(image))]









def bounding_box(image):
    tmp_image = numpy.absolute(255 - thresholding_image(image, 160))
    #misc.imshow(tmp_image)

    rows = numpy.any(tmp_image, axis=1)
    cols = numpy.any(tmp_image, axis=0)
    if True in rows and True in cols:
        rmin, rmax = numpy.where(rows)[0][[0, -1]]
        cmin, cmax = numpy.where(cols)[0][[0, -1]]
    else:
        rmin = 0
        rmax = 1
        cmin = 0
        cmax = 1
    #misc.imshow(tmp_image[rmin:rmax, cmin:cmax])
    rmax += 1
    cmax += 1
    #return tmp_image[rmin:rmax, cmin:cmax]
    return rmin, rmax, cmin, cmax


def flatten_matrix_with_first_dim_kept(mat):
    mat = numpy.array(mat)
    data_size = 1
    data_size = reduce(lambda data_size, y: data_size*y, mat.shape[1:])
    new_shape = tuple([len(mat), data_size])

    mat = mat.reshape(new_shape)

    return mat


def print_confusion_matrix(confusion, classes_dict, output_path=''):
    swapped_dict = dict((v,k) for k,v in classes_dict.iteritems())

    output = open(output_path + 'output.csv', mode='w')
    total_correct = 0
    total_error = 0

    for i in xrange(len(confusion) + 1):
        row_correct = 0
        row_error = 0
        for j in xrange(len(confusion) + 1):
            # Prints to file confusion
            if i == 0 and j == 0:
                output.write(' ')
            elif i == 0 and j != 0:
                output.write(swapped_dict[j-1])
            elif i != 0 and j == 0:
                output.write(swapped_dict[i - 1])
            else:
                if i == j:
                    row_correct += confusion[i - 1][j - 1]
                else:
                    row_error += confusion[i - 1][j - 1]

                output.write(str(confusion[i-1][j-1]))
                c = 1

            if j < (len(confusion) + 1):
                output.write(',')

        percent = 0;
        if row_correct + row_error > 0:
            percent = float(row_correct) / float(row_correct + row_error) * 100

        total_correct += row_correct
        total_error += row_error

        output.write(str(percent) + '\n')

    output.write("total correct: " + str(total_correct) + '\n')
    output.write("total error: " + str(total_error)+ '\n')
    output.write("total percent: " + str(float(total_correct)/float(total_correct+total_error) * 100)+ '\n')

    output.close()


def write_dataset_to_disk(data, labels, path):
    if not os.path.exists(path):
        os.makedirs(path)

    os.chdir(path)
    i = 0

    for label, image in zip(labels, data):
        if not os.path.exists(label):
            os.mkdir(label)

        os.chdir(label)
        misc.toimage(image, cmin=0, cmax=255).save(str(i) + '.png')

        i += 1
        os.chdir('..')


def load_svm_model(model_path):
    return joblib.load(model_path)


def train_with_svm(train_data, train_labels, output_path, save_model=False):
    train_data = flatten_matrix_with_first_dim_kept(train_data)

    # Train
    model = SVC(kernel='linear', degree=3, C=1)
    model.fit(train_data, train_labels)

    if save_model:
        joblib.dump(model, output_path)

    return model


def test_svm_model(model, test_data, test_labels, class_num_dict, output_path):
    errors = 0
    correct = 0
    confusion = numpy.zeros((len(class_num_dict), len(class_num_dict)), dtype='int32')
    test_data = flatten_matrix_with_first_dim_kept(test_data)

    prediction = model.predict(test_data)

    for i in range(len(test_data)):

        confusion[class_num_dict[test_labels[i]], class_num_dict[prediction[i]]] += 1

        if prediction[i] == test_labels[i]:
            correct += 1
        else:
            errors += 1

    print_confusion_matrix(confusion, class_num_dict, output_path)
    print "correct: ", correct
    print "error: ", errors
    print "correct percentage: ", correct / (correct + errors * 1.)


def predict_with_svm(model, dataset_path, is_batch=True):
    os.chdir(dataset_path)

    if is_batch:
        start_time = datetime.datetime.now()

        images = []
        labels = []

        for file in os.listdir('.'):
            if os.path.isfile(file):
                labels.append(file)
                im_arr = misc.imread(file, mode='YCbCr')
                images.append(im_arr)

        images = feature_extraction(images)
        images = flatten_matrix_with_first_dim_kept(images)
        prediction = model.predict(images)

        for i in range(len(images)):
            print labels[i], prediction[i]

        end_time = datetime.datetime.now()
        print "total time: ", end_time - start_time
        '''
    # Event prediction
    else:
        for file in os.listdir('.'):
            start_time = datetime.datetime.now()
            if os.path.isfile(file):
                print file, ',',
                im_arr = misc.imread(file, mode='YCbCr')
                images[0] = im_arr
                feature_extraction(images)
                images = flatten_matrix_with_first_dim_kept(images)
                prediction = model.predict(images)
                print prediction[0], ',',
            end_time = datetime.datetime.now()

            print end_time - start_time
    '''



SAVE_DATASET = False
TRAIN = True

MODEL_NAME = "model.pkl"
DATASET_PATH = "/home/osboxes/Desktop/ML/Final_Project/Dataset/"
TRAIN_SET_PATH = "/home/osboxes/Desktop/ML/Final_Project/Datasets/Trainset/"
TEST_SET_PATH = "/home/osboxes/Desktop/ML/Final_Project/Datasets/Testset/"
WORKSPACE_PATH = "/home/osboxes/Desktop/ML/Final_Project/Model_Output/"
MAX_NUM_OF_CLASSES = -1

if SAVE_DATASET:
    print "save dataset"
    images, labels, classes = numpy.array(get_dataset(DATASET_PATH, MAX_NUM_OF_CLASSES))
    train_images, train_labels, test_images, test_labels = seperate_sets(images, labels)

    write_dataset_to_disk(train_images, train_labels, TRAIN_SET_PATH)
    write_dataset_to_disk(test_images, test_labels, TEST_SET_PATH)

    class_num_dict = dict(zip(classes, range(len(classes))))

    print "number of classes: ", len(classes)

    count = [0] * len(class_num_dict)
    for i in test_labels:
        count[class_num_dict[i]] += 1

    print "test set Diversification:", count

    count[:] = [0] * len(count)
    for i in train_labels:
        count[class_num_dict[i]] += 1

    print "train set Diversification:", count

elif TRAIN:
    print "train"
    USE_STATIC_DATA = True
    SAVE_MODEL = True

    if USE_STATIC_DATA:
        train_images, train_labels, classes = numpy.array(get_dataset(TRAIN_SET_PATH, MAX_NUM_OF_CLASSES))
        train_images = feature_extraction(train_images)
        test_images, test_labels, classes = numpy.array(get_dataset(TEST_SET_PATH, MAX_NUM_OF_CLASSES))
        test_images = feature_extraction(test_images)
    else:
        images, labels, classes = numpy.array(get_dataset(DATASET_PATH, MAX_NUM_OF_CLASSES))
        images = feature_extraction(images)
        train_images, train_labels, test_images, test_labels = seperate_sets(images, labels)

    class_num_dict = dict(zip(classes, range(len(classes))))

    model = train_with_svm(train_images, train_labels, WORKSPACE_PATH + '/' + MODEL_NAME, SAVE_MODEL)
    test_svm_model(model, test_images, test_labels, class_num_dict, WORKSPACE_PATH)
elif not TRAIN:
    print "prod"
    PREDICT_PATH = "/home/osboxes/Desktop/ML/Final_Project/Predictset/"
    IS_BATCH = True

    model = load_svm_model(WORKSPACE_PATH + '/' + MODEL_NAME)
    predict_with_svm(model, PREDICT_PATH, IS_BATCH)
