import tensorflow as tf
import time

sess = tf.Session()
A = tf.random_uniform((1000,1000))
cur_time = time.clock()
for i in range(int(1e3)):
	sess.run(tf.matmul(A,A))
print(time.clock()-cur_time)
