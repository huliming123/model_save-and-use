import tensorflow as tf
import numpy as np
from sklearn import linear_model
x=np.arange(100).reshape(100,1).astype('f4')
y=10*x+1
# lay_1
w1=tf.Variable(tf.truncated_normal(shape=[1,10],dtype=tf.float32),name='w1')
b1=tf.Variable(tf.constant(0.1,shape=[10],dtype=tf.float32),name='b1')
l1=tf.nn.relu(tf.matmul(x,w1)+b1)
#lay_2
w2=tf.Variable(tf.truncated_normal(shape=[10,1],dtype=tf.float32),name='w2')
b2=tf.Variable(tf.constant(0.1,shape=[1],dtype=tf.float32),name='b2')
l2=tf.nn.relu(tf.matmul(l1,w2)+b2)
#loss
loss=tf.reduce_mean(tf.square(l2-y))
train_step=tf.train.AdadeltaOptimizer(0.1).minimize(loss)
#sess
init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(10001):
        sess.run(train_step)
        if _ % 100==0:
            print('%d次'%_,sess.run(loss))
            # print('%d次' % _, sess.run(l2))
        saver.save(sess,'头疼/mmm.ckpt')
