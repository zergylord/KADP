import time
import numpy as np
import tensorflow as tf

W = tf.random_uniform((100,5))
normed_W = tf.nn.softmax(W)
R = tf.random_uniform((100,))
inds = np.random.choice(100,5*100).reshape(100,5)

R_ = tf.gather(R,inds)
V = [tf.zeros((100,))]
#V_ = tf.gather(V[0],inds)
for t in range(1000):
    V_ = tf.gather(V[t],inds)
    V.append(tf.reduce_sum(normed_W*(R_+.9*V_),1))
_target = tf.placeholder(tf.float32,shape=(100,))
loss = tf.reduce_sum(tf.square(_target-V[-1]))
get_grads = tf.gradients(loss,W)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
cur_time = time.clock()
for i in range(100):
    foo = sess.run(get_grads,feed_dict={_target:np.random.randn(100)})
print(foo,time.clock()-cur_time)
