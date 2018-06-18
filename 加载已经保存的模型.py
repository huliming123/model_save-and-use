import tensorflow as tf
import numpy as np
saver=tf.train.import_meta_graph('头疼/mmm.ckpt.meta')
x= np.arange(100,200).reshape(100, 1).astype('f4')
with tf.Session() as sess:
    saver.restore(sess,'头疼/mmm.ckpt')
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("w1:0")))
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("b1:0")))tf.get
    w1=tf.get_default_graph().get_tensor_by_name('w1:0')
    b1=tf.get_default_graph().get_tensor_by_name('b1:0')
    l1=tf.nn.relu(tf.matmul(x,w1)+b1)
    w2=tf.get_default_graph().get_tensor_by_name("w2:0")
    b2=tf.get_default_graph().get_tensor_by_name("b2:0")
    l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    # print(x)
    print(x,sess.run(l2))