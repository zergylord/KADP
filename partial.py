import numpy as np
import tensorflow as tf

sess = tf.Session()
a = tf.placeholder(tf.float32, shape=[])
b = tf.placeholder(tf.float32, shape=[])
c = tf.placeholder(tf.float32, shape=[])
r1 = tf.add(a, b)
r2 = tf.mul(r1, c)
h = sess.partial_run_setup([r1, r2], [a, b, c])
res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
res1 = sess.partial_run(h, r2, feed_dict={c: res})
res2 = sess.partial_run(h, r2, feed_dict={c: res})
