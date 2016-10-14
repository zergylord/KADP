
import tensorflow as tf
import numpy as np

n_samples = 100
n_neighbors = 5
indices = []
values = []
for i in range(n_samples):
    knn = np.random.choice(n_samples,n_neighbors)
    for j in knn:
        indices.append([i,j])
        values.append(np.random.randn())
values = tf.constant(values,dtype='float64')
W = tf.SparseTensor(indices,values,[n_samples,n_samples])
normed_W = tf.sparse_softmax(W)
R = tf.constant(np.random.randn(n_samples).astype(np.float64))
gamma = .9
V = [np.zeros((n_samples,)).astype(np.float64)]
V_target = []
loss = 0
for t in range(1000):
    V.append(tf.squeeze(tf.sparse_tensor_dense_matmul(normed_W,tf.expand_dims(R+V[t],1))))
    V_target.append(np.random.randn(n_samples))
    loss = loss + tf.square(V[-1]-V_target[-1])
get_grads = tf.gradients(loss,values)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(V[-1]).sum())
for t in range(10):
    foo = sess.run(get_grads)
    print(foo.sum())
