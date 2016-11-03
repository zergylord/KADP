import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x = tf.constant(np.random.rand(100,5))
y = tf.nn.softmax(x)
grads = tf.gradients(y,x)
sess = tf.Session()
foo,bar = sess.run([x,grads])
print(bar[0].shape)
plt.plot(foo[:,0],bar[0][:,0])
plt.show()
