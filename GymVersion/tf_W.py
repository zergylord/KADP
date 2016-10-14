import tensorflow as tf
import numpy as np

n_samples = 100
n_neighbors = 5
weights = np.zeros((n_samples,n_samples))
for i in range(n_samples):
    weights[i,np.random.choice(n_samples,n_neighbors)] = np.random.rand(n_neighbors)
W = tf.Variable(weights)
W = tf.nn.softmax(W,dim=-1)
R = tf.constant(np.random.randn(n_samples))
gamma = .9
V = [np.zeros((n_samples,))]
V_target = []
loss = 0
for t in range(1000):
    V.append(tf.squeeze(tf.matmul(W,tf.expand_dims(R+V[t],1))))
    V_target.append(np.random.randn(n_samples))
    loss = loss + tf.square(V[-1]-V_target[-1])
train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(W).sum(1))
for t in range(10):
    _,foo = sess.run([train_step,V[n_samples-1]])
    print(foo)
