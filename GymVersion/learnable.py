import tensorflow as tf
import numpy as np
from ops import *
hid_dim = 10

x1 = tf.placeholder(tf.float32,shape=(None,1))
x2 = tf.placeholder(tf.float32,shape=(None,1))

embed1 = linear(linear(x1,hid_dim,'foo',tf.nn.relu),1,'bar')
pred = embed1**2+1
embed2 = linear(linear(x1,hid_dim,'foo',tf.nn.relu,tied=True),1,'bar',tied=True)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred-embed2),1))
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

with tf.variable_scope('bar',reuse=True):
    weights = tf.get_variable('W')
    bias = tf.get_variable('b')
for i in range(4000):
    data = np.expand_dims(np.linspace(0,100),1)
    _,cur_loss,W,b = sess.run([train_step,loss,weights,bias],feed_dict={x1:data,x2:data+1})
    print(cur_loss)

rep,output,target = sess.run([embed1,pred,embed2],feed_dict={x1:data,x2:data+1})
print(np.concatenate([rep,output,target],1))
