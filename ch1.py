import tensorflow as tf
import numpy as np

sess = tf.Session()
#initializing variable
my_var  = tf.Variable(tf.zeros([2,3]))
init_op = tf.global_variables_initializer()#we initialize all of the global variables at this point
sess.run(init_op)

#initializing placeholder

x = tf.placeholder(tf.float32,shape=[2,2])#placehol
y = tf.identity(x)
x_vals = np.random.rand(2,2)#the actual values
sess.run(y,feed_dict={x:x_vals})

#working with matrices



#downloading the dataset
# from sklearn import datasets
# iris = datasets.load_iris()

id = tf.identity([3,3])
for


