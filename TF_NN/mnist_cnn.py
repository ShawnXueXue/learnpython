import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

dataPath = 'E:\wgXue\TF-MNIST'
logdir='log_test/'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, W):
    # padding='SAME'进行边缘0填充,保证输出与输入大小一致
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # poolling之后,大小会缩小一倍
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )


def m1():
    sess = tf.InteractiveSession()
    # get data
    mnist = input_data.read_data_sets(dataPath, one_hot=True)

    # layer1
    # 卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x = tf.placeholder("float", [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # layer2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # full connect layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # drop out
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter(logdir, sess.graph)
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def poolingTest():
    t = [
        [7, 2, 3, 4, 2],
        [5, 6, 7, 8, 3],
        [4, 3, 2, 1, 1],
        [8, 7, 6, 1, 5],
        [3, 5, 6, 3, 1]
    ]
    a = tf.reshape(t, [1, 5, 5, 1])
    r = tf.nn.max_pool(a,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 1, 1, 1],
                       padding='SAME')
    with tf.Session() as sess:
        print("org:")
        print(sess.run(a))
        print("aft:")
        print(sess.run(r))

def convTest():
    t = [
        [7.0, 2.0, 3.0, 4.0, 2.0],
        [5.0, 6.0, 7.0, 8.0, 3.0],
        [4.0, 3.0, 2.0, 1.0, 1.0],
        [8.0, 7.0, 6.0, 1.0, 5.0],
        [3.0, 5.0, 6.0, 3.0, 1.0]
    ]
    a = tf.reshape(t, [1, 5, 5, 1])
    W = tf.Variable(tf.truncated_normal([3, 3, 1, 1], stddev=0.1))
    r = tf.nn.conv2d(a, W, strides=[1, 1, 1, 1], padding='SAME')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("org:")
        print(sess.run(a))
        print("W:")
        print(sess.run(W))
        print("aft:")
        print(sess.run(r))
    # 使用SAME时,会在输入矩阵外边补一圈0,以保证大小相同

if __name__ == '__main__':
    # convTest()
    m1()