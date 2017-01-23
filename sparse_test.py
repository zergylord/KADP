import numpy as np
import tensorflow as tf
sess = tf.Session()
theta = tf.get_variable('theta',[2,],tf.float32)
A = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=theta, shape=[3, 4])
dAdTheta = tf.gradients(A,[theta])
sess.run(tf.global_variables_initializer())
out = sess.run(dAdTheta)
print(out)
