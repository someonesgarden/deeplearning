import tensorflow as tf
import numpy as np
sess = tf.Session()

x_datas = np.array([3., 4., 5., 6., 7.])
x_val = tf.placeholder(tf.float32)
cons = tf.constant(3.)
x_prod = tf.multiply(x_val,cons)


for x_d in x_datas:
    print(sess.run(x_prod, feed_dict={x_val:x_d}))
