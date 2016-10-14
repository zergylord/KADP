import tensorflow as tf
import numpy as np
W = tf.Variable(np.random.rand(100))
foo = tf.reduce_sum(W*np.random.rand(100))
bar = tf.gradients(foo,W)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(bar))
