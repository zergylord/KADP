import tensorflow as tf
import time

sess = tf.Session()
A = tf.random_uniform((1000,1000))
for i in range(int(1e3)):
	A = (tf.matmul(A,A))
cur_time = time.clock()
sess.run(A)
print(time.clock()-cur_time)
