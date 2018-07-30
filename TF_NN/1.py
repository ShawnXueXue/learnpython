import tensorflow as tf

# with tf.device('/cpu:0'):
logdir='log_test/'
def m1():
    hello =  tf.constant('Hello TF!')
    sess = tf.Session()
    tf.summary.FileWriter(logdir, sess.graph)
    print(sess.run(hello))

if __name__ == '__main__':
    m1()