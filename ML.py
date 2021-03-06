import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

# read the data and labels as ont-hot vectors
# one-hot means a sparse vector for every observation where only
# the class label is 1, and every other class is 0.
# more info here:
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print (mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

print(mnist.train.images.shape, mnist.train.labels.shape)

print(np.min(mnist.train.images), np.max(mnist.train.images))
#plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')

# We start
n_input = 784
n_output = 10

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

x_tensor = tf.reshape(x, [-1, 28, 28, 1])

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
n_filters_1 = 16
n_filters_2 = 16

# parameters
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])

# layers
h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
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
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# variables:
sess = tf.Session()
sess.run(tf.global_variables_initializer())




saver = tf.train.Saver()
#saver.restore(sess, '/home/osboxes/PycharmProjects/ML_FinalProject/my-save-dir/my-model-10000')

#sess = tf.Session()
#new_saver = tf.train.import_meta_graph('/home/osboxes/PycharmProjects/ML_FinalProject/my-save-dir/my-model-10000.meta')
#new_saver.restore(sess, tf.train.latest_checkpoint('/home/osboxes/PycharmProjects/ML_FinalProject/my-save-dir/my-model-10000'))



print("Accuracy for test set: {}".format(sess.run(accuracy,
               feed_dict={
                   x: mnist.test.images[:5000],
                   y: mnist.test.labels[:5000],
                   keep_prob: 1.0
               })))

# We'll train in minibatches and report accuracy:
batch_size = 10
n_epochs = 10
l_loss = list()
for epoch_i in range(n_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    print batch_ys.shape
    sess.run(optimizer, feed_dict={
        x: batch_xs, y: batch_ys, keep_prob: 0.5})
    '''
    for batch_i in range(0, mnist.train.num_examples, batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})
    '''
    loss = sess.run(accuracy, feed_dict={
                       x: mnist.validation.images,
                       y: mnist.validation.labels,
                       # x: mnist.test.images,
                        #y: mnist.test.labels,
                       keep_prob: 1.0 })
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    l_loss.append(loss)
    saver.save(sess, '/home/osboxes/PycharmProjects/ML_FinalProject/my-save-dir/my-model-10000', global_step=epoch_i)

plt.title('CNN Acuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(l_loss, color='m')
plt.show()


#saver.save(sess, '/home/osboxes/PycharmProjects/ML_FinalProject/my-save-dir/my-model-10000', global_step=epoch_i)
#saver.export_meta_graph('/home/osboxes/PycharmProjects/ML_FinalProject/my-save-dir/my-model-10000.meta')

print("Accuracy for test set: {}".format(sess.run(accuracy,
               feed_dict={
                   x: mnist.test.images[:5000],
                   y: mnist.test.labels[:5000],
                   keep_prob: 1.0
               })))
