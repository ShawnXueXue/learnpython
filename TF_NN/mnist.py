# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def m1():
    # 生成一些三维数据, 然后用一个平面拟合它.
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], 0.1, 0.2))
    y = tf.matmul(W, x_data) + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(0, 201):
        sess.run(train) # equal to sess.run([train, loss])
        print(step, sess.run(W), sess.run(b))

dataPath = 'E:\wgXue\TF-MNIST'

def m2():
    #get data
    mnist = input_data.read_data_sets(dataPath, one_hot=True)

    #y = softmax(Wx + b)
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784,10])) # tf.zeros()
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #define loss
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    #train
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    #init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    m2()