# coding: utf-8

from train_model import build_mnist_graph, load_pathes
import tensorflow as tf
import numpy as np
import os
import dataset_utils
import datetime

#SIGNS_DICT = ['א','ב','ג','ד','ה','ו','ז','ח','ט','י','כ','ל','מ','נ','ס','ע','פ','צ','ק','ר','ש','ת',' ','ך','ם','ן','ף','ץ','(',')','+','-','∫','≤','≥','∞','∑','∩','≠','÷','×','0','1','2','3','4','5','6','7','8','9']
SIGNS_DICT = {
0: 'א',
1: 'ב',
2: 'ג',
3: 'ד',
4: 'ה',
5: 'ו',
6: 'ז',
7: 'ח',
8: 'ט',
9: 'י',
10: 'כ',
11: 'ל',
12: 'מ',
13: 'נ',
14: 'ס',
15: 'ע',
16: 'פ',
17: 'צ',
18: 'ק',
19: 'ר',
20: 'ש',
21: 'ת',
22: ' ',
23: 'ך',
24: 'ם',
25: 'ן',
26: 'ף',
27: 'ץ',
28: '(',
29: ')',
30: '+',
31: '-',
32: '∫',
33: '≤',
34: '≥',
35: '∞',
36: '∑',
37: '∩',
38: '≠',
39: '÷',
40: '×',
41: '0',
42: '1',
43: '2',
44: '3',
45: '4',
46: '5',
47: '6',
48: '7',
49: '8',
50: '9'
}


def create_graph(modelFullPath):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def print_confusion_matrix(confusion, output_path='/home/osboxes/PycharmProjects/ML_final_project/workspace'):
    output = open(output_path + '/confusion_matrix.csv', mode='w')
    total_correct = 0
    total_error = 0

    # prints confusion matrix
    for i in xrange(len(confusion) + 1):
        row_correct = 0
        row_error = 0
        for j in xrange(len(confusion) + 1):
            # Prints to file confusion
            if i == 0 and j == 0:
                output.write(' ')
            elif i == 0 and j != 0:
                output.write(SIGNS_DICT[j-1])
            elif i != 0 and j == 0:
                output.write(SIGNS_DICT[i - 1])
            else:
                if i == j:
                    row_correct += confusion[i - 1][j - 1]
                else:
                    row_error += confusion[i - 1][j - 1]

                output.write(str(confusion[i-1][j-1]))
                c = 1

            if j < (len(confusion) + 1):
                output.write(',')

        # Calculates label correct percent
        if row_correct + row_error > 0:
            percent = float(row_correct) / float(row_correct + row_error) * 100
        elif row_correct + row_error == 0:
            percent = -1
        else:
            raise NameError('Something went wrong with printing confusion matrix')

        # Sums row correct and error to total correct and error
        total_correct += row_correct
        total_error += row_error

        output.write(str(percent) + '\n')

    # Print aggregated results
    output.write("total correct: " + str(total_correct) + '\n')
    output.write("total error: " + str(total_error)+ '\n')
    output.write("total percent: " + str(float(total_correct)/float(total_correct+total_error) * 100)+ '\n')

    output.close()


def test_model1(images_path, labels_path, model_path, output_path='/home/osboxes/PycharmProjects/ML_final_project/workspace'):
    all_image_path = list()
    labels = list()

    # Gets all images pathes and labels
    for current_dir in os.listdir(images_path):
        for current_image_path in os.listdir(images_path + '/' + current_dir):
            all_image_path.append(images_path + '/' + current_dir + '/' + current_image_path)
            labels.append(current_dir)

    # Get labels
    f = open(labels_path, 'rb')
    lines = f.readlines()
    labels_dict = [str(w).replace("\n", "") for w in lines]
    f.close()
    confusion_matrix = np.zeros((len(SIGNS_DICT), len(SIGNS_DICT)), dtype='int')

    # Create the graph from the given path
    create_graph(model_path)

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for i, image_path in enumerate(all_image_path):
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

            confusion_matrix[int(labels[i]), int(labels_dict[top_k[0]])] += 1
            print i
            #print 'real label: ', SIGNS_DICT[int(labels[i])]
            #print 'predicted label: ', SIGNS_DICT[int(labels_dict[top_k[0]])]
            #Image.open(image_path).show()

    print_confusion_matrix(confusion_matrix,output_path)


def test_model2(images_path, labels_path, model_path, output_path='/home/osboxes/PycharmProjects/ML_final_project/workspace2'):
    image_size = (28,28)
    all_image_path = list()
    labels = list()

    # Gets all images pathes and labels
    for current_dir in os.listdir(images_path):
        for current_image_path in os.listdir(images_path + '/' + current_dir):
            all_image_path.append(images_path + '/' + current_dir + '/' + current_image_path)
            labels.append(current_dir)

    # Get labels
    f = open(labels_path, 'rb')
    lines = f.readlines()
    labels_dict = [str(w).replace("\n", "") for w in lines]
    f.close()
    confusion_matrix = np.zeros((len(SIGNS_DICT), len(SIGNS_DICT)), dtype='int')

    # Create the graph from the given path
    build_mnist_graph(len(labels_dict))

    f_log = open(output_path + '/' + 'error_log.txt', 'w')


    #create_graph(model_path)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        x = sess.graph.get_tensor_by_name('input_placeholder:0')
        #y = sess.graph.get_tensor_by_name('ground_truth_placeholder:0')
        results = sess.graph.get_tensor_by_name('final_result:0')
        #optimizer = sess.graph.get_operation_by_name('train')
        #accuracy = sess.graph.get_tensor_by_name('accuracy:0')
        keep_prob = sess.graph.get_tensor_by_name('dropout_prob1:0')

        for i, image_path in enumerate(all_image_path):
            image_data = load_pathes([image_path], image_size, f_log)
            predictions = sess.run(results,
                                   {x: image_data, keep_prob: 1.0})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

            confusion_matrix[int(labels[i]), int(labels_dict[top_k[0]])] += 1
            print i
            #print 'real label: ', SIGNS_DICT[int(labels[i])]
            #print 'predicted label: ', SIGNS_DICT[int(labels_dict[top_k[0]])]
            #Image.open(image_path).show()


    f_log.close()
    print_confusion_matrix(confusion_matrix,output_path)



def predict(imagesPath, labelsFullPath, modelFullPath):
    tmp_path = imagesPath

    # Checks if path is dir or single file
    if not os.path.exists(tmp_path):
        raise NameError('Path does not exist')
    else:
        if os.path.isdir(tmp_path):
            os.chdir(tmp_path)
            tmp_path = os.listdir(tmp_path)
            tmp_path.sort()
        elif os.path.isfile(tmp_path):
            tmp_path = [tmp_path]

    # Creates graph from saved GraphDef.
    create_graph(modelFullPath)

    # Gets labels
    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    f.close()

    # Predicts image's labels with model
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for imagePath in tmp_path:
            image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

            print imagePath, SIGNS_DICT[int(labels[top_k[0]])]


def predict2(imagesPath, labelsFullPath, modelFullPath, image_size):
    tmp_path = imagesPath

    # Checks if path is dir or single file
    if not os.path.exists(tmp_path):
        raise NameError('Path does not exist')
    else:
        if os.path.isdir(tmp_path):
            os.chdir(tmp_path)
            tmp_path = os.listdir(tmp_path)
            tmp_path.sort()
        elif os.path.isfile(tmp_path):
            tmp_path = [tmp_path]



    # Gets labels
    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    f.close()

    build_mnist_graph(len(labels))

    f_log = open('/tmp/error_log_for_ML_delete.txt', 'w')

    # Predicts image's labels with model
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, modelFullPath)
        x = sess.graph.get_tensor_by_name('input_placeholder:0')
        keep_prob = sess.graph.get_tensor_by_name('dropout_prob1:0')


        results = sess.graph.get_tensor_by_name('final_result:0')

        for imagePath in tmp_path:

            image_data = load_pathes([imagePath], image_size, f_log)

            predictions = sess.run(results,
                                   {x: image_data, keep_prob: 1.0})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

            print imagePath, 'first guess: ' + SIGNS_DICT[int(labels[top_k[0]])], 'second guess: ' + SIGNS_DICT[int(labels[top_k[1]])]


def predict_unprepared_data2(imagesPath,
                             labelsFullPath,
                             modelFullPath,
                             scale_image_size=(-1, -1),
                             normalize_values=False,
                             change_to_gray=False,
                             smooth_factor=1):
    tmp_path = imagesPath

    # Checks if path is dir or single file
    if not os.path.exists(tmp_path):
        raise NameError('Path does not exist')
    else:
        if os.path.isdir(tmp_path):
            os.chdir(tmp_path)
            tmp_path = os.listdir(tmp_path)
            tmp_path.sort()
        elif os.path.isfile(tmp_path):
            tmp_path = [tmp_path]



    # Gets labels
    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    f.close()

    build_mnist_graph(len(labels))

    f_log = open('/tmp/error_log_for_ML_delete.txt', 'w')
    time_taken_list = list()

    # Predicts image's labels with model
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, modelFullPath)
        x = sess.graph.get_tensor_by_name('input_placeholder:0')
        keep_prob = sess.graph.get_tensor_by_name('dropout_prob1:0')


        results = sess.graph.get_tensor_by_name('final_result:0')

        for imagePath in tmp_path:
            start_time = datetime.datetime.now()
            image_data = dataset_utils.load_image(imagePath, manipulateImage=False)
            image_data = dataset_utils.prepare_image(image=image_data,
                                                     scale_image_size=scale_image_size,
                                                     normalize_values=normalize_values,
                                                     change_to_gray=change_to_gray,
                                                     smooth_factor=smooth_factor)

            if scale_image_size == (-1,-1):
                scale_image_size = image_data.size

            image_num_of_bytes = 1
            for i in scale_image_size:
                image_num_of_bytes *= i

            new_shape = tuple([1, image_num_of_bytes])
            image_data = np.array(image_data).reshape(new_shape)

            predictions = sess.run(results,
                                   {x: image_data, keep_prob: 1.0})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
            end_time = datetime.datetime.now()
            time_taken = timedelta_milliseconds(end_time - start_time)
            time_taken_list.append(time_taken)
            print imagePath, 'first guess: ' + SIGNS_DICT[int(labels[top_k[0]])], 'second guess: ' + SIGNS_DICT[int(labels[top_k[1]])]

        avg_time_per_image = sum(time_taken_list) / len(time_taken_list)

        print 'average time for prediction: {} milliseconds'.format(avg_time_per_image)


def timedelta_milliseconds(td):
    return td.days*86400000 + td.seconds*1000 + td.microseconds/1000

