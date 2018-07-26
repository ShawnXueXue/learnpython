import tensorflow as tf
import numpy as np

import PIL.Image
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd

def DisplayFractal(a, fmt='jpeg'):
  """显示迭代计算出的彩色分形图像。"""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  # 文件方式
  path = "1.jpeg"
  PIL.Image.fromarray(a).save(path, fmt)
  img = plt.imread(path)
  plt.imshow(img)
  plt.show()

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    Y,X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    Z = X + 1j * Y
    xs = tf.constant(Z.astype("complex64"))
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, "float32"))
    init = tf.global_variables_initializer()
    sess.run(init)
    zs_ = zs * zs + xs
    # complex_abs is replaced by abs
    not_diverged = tf.abs(zs_) < 4
    step = tf.group(
        zs.assign(zs_),
        ns.assign_add(tf.cast(not_diverged, "float32"))
    )
    for i in range(200):
        step.run()
    DisplayFractal(ns.eval())
