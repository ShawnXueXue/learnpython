import tensorflow as tf

# with tf.device('/cpu:0'):

def m1():
    hello = tf.constant('Hello TF!')
    sess = tf.Session()
    print(sess.run(hello))

if __name__ == '__main__':
    m1()