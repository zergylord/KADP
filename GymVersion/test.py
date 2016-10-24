import tensorflow as tf
import numpy as np

def tf_kernel(x,X):
    dist = tf.sqrt(tf.reduce_sum(tf.square(x-X),-1))
    sim = tf.exp(-tf.clip_by_value(dist,0,10))
    k_sim,k_inds = tf.nn.top_k(sim,k=5,sorted=False)
    return k_sim,k_inds
gamma = .9
row_offsets = np.expand_dims(np.arange(4)*10,-1)
R_view = np.random.rand(40)
cur_V = np.random.rand(40)
NT_view = np.float64(np.random.rand(40) < .5)
def tf_get_value(s,S):
    weights,inds = tf_kernel(s,S)
    normed_weights = tf.nn.softmax(weights)
    row_inds = row_offsets+inds
    R_ = tf.gather(R_view,row_inds)
    NT_ = tf.gather(NT_view,row_inds)
    V_ = tf.gather(cur_V,row_inds)
    val = tf.reduce_max(tf.reduce_sum(normed_weights*(R_+NT_*gamma*V_),-1),0)
    return val

a = tf.constant(np.random.randn(2))
B = tf.constant(np.random.randn(4,10,2))
sim,inds = tf_kernel(a,B)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#foo,bar = sess.run([sim,inds])
foo = sess.run(tf_get_value(a,B))
print(foo)
