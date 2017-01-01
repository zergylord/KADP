import tensorflow as tf
import time

sess = tf.Session()
A = tf.random_uniform((10000,10000))
for i in range(int(1e3)):
	A = (tf.matmul(A,A))
cur_time = time.clock()
sess.run(A.op)
print(time.clock()-cur_time)
