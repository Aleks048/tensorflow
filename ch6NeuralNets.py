import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

tf.reset_default_graph()
sess = tf.Session()

def simpleNN():
    x_val = 5.
    #model
    x_data = tf.placeholder(shape=[1],dtype=tf.float32)
    a = tf.Variable(tf.constant(1.))
    b = tf.Variable(tf.constant(1.))
    out = tf.add(tf.multiply(a,x_data),b)

    loss = tf.square(tf.subtract(out,50.))
    myopt = tf.train.GradientDescentOptimizer(0.01)
    trainStep = myopt.minimize(loss)
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(trainStep,feed_dict={x_data:[x_val]})
        print("a: ",sess.run(a),"b: ",sess.run(b))
def irisDataSimpleNN():
    #load data
    iris = datasets.load_iris() 
    x_vals = np.array([x[0:3] for x in iris.data]) 
    y_vals = np.array([x[3] for x in iris.data]) 
    sess = tf.Session() 
    #random seed//seed the data to make the results reproducable 
    seed = 2 
    tf.set_random_seed(seed) #https://www.tensorflow.org/api_docs/python/tf/random/set_random_seed
    np.random.seed(seed) 
    #split train/test
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False) 
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices))) 
    x_vals_train = x_vals[train_indices] 
    x_vals_test = x_vals[test_indices] 
    y_vals_train = y_vals[train_indices] 
    y_vals_test = y_vals[test_indices]
    #min/max cols normalization
    def normalize_cols(m):    
            col_max = m.max(axis=0)    
            col_min = m.min(axis=0)    
            return (m-col_min) / (col_max - col_min)
    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train)) 
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test)) 
    
    #model
    batch_size = 50
    x_data = tf.placeholder(shape=[None,3],dtype = tf.float32)
    y_target = tf.placeholder(shape = [None,1],dtype=tf.float32)

    hidden_layer_nodes = 5
    A1 = tf.Variable(tf.random_normal(shape = [3,hidden_layer_nodes]))
    b1 = tf.Variable(tf.random_uniform(shape=[hidden_layer_nodes]))
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]))
    b2 = tf.Variable(tf.random_uniform(shape = [1]))

    hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))

    loss = tf.reduce_mean(tf.square(y_target-final_output))

    my_opt = tf.train.GradientDescentOptimizer(0.005)
    train_step = my_opt.minimize(loss)

    sess.run(tf.global_variables_initializer())

    #training
    loss_vec = [] 
    test_loss = [] 
    for i in range(500):    
        # First we select a random set of indices for the batch.    
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)    
        # We then select the training values    
        rand_x = x_vals_train[rand_index]    
        rand_y = np.transpose([y_vals_train[rand_index]]) 
        
        sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})    
        # We save the training loss    
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})    
        loss_vec.append(np.sqrt(temp_loss))
        # Finally, we run the test-set loss and save it.    
        test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})    
        test_loss.append(np.sqrt(test_temp_loss))
    #plot
    plt.plot(loss_vec, 'k-', label='Train Loss') 
    plt.plot(test_loss, 'r--', label='Test Loss') 
    plt.title('Loss (MSE) per Generation') 
    plt.xlabel('Generation') 
    plt.ylabel('Loss') 
    plt.legend(loc='upper right') 
    plt.show()
def differentNNLayers1D():
    #convolution
    data_size = 25 
    data_1d = np.repeat(range(1,6),5)#np.random.normal(size=data_size) 
    print(data_1d)
    x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size]) 

    def conv_layer_1d(input_1d,my_filter):
            input_2d = tf.expand_dims(input_1d,0)
            input_3d = tf.expand_dims(input_2d,0)
            input_4d = tf.expand_dims(input_3d,3)
            print(sess.run(tf.shape(input_2d)))
            print(sess.run(tf.shape(input_3d)))
            print(sess.run(tf.shape(input_4d)))
            print(sess.run(input_1d,feed_dict={x_input_1d:data_1d}))
            print(sess.run(input_2d,feed_dict={x_input_1d:data_1d}))
            print(sess.run(input_3d,feed_dict={x_input_1d:data_1d}))
            print("input_before_conv",sess.run(input_4d,feed_dict={x_input_1d:data_1d}))

            convolution_output = tf.nn.conv2d(input_4d,filter=my_filter,strides=[1,1,1,1],padding="VALID")
            print("convolution",sess.run(convolution_output,feed_dict={x_input_1d:data_1d}))
            print("filter",sess.run(my_filter))
            conv_output_1d = tf.squeeze(convolution_output)
            return conv_output_1d
    my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
    sess.run(tf.global_variables_initializer())
    my_convolution_output = tf.nn.relu(conv_layer_1d(x_input_1d,my_filter))
    #sess.run(my_convolution_output,feed_dict={x_input_1d:data_1d})
    
    #maxpool
    def max_pool(input_1d,width):
        input_2d = tf.expand_dims(input_1d,0)
        input_3d = tf.expand_dims(input_2d,0)
        input_4d = tf.expand_dims(input_3d,3)

        pool_output = tf.nn.max_pool(input_4d,ksize=[1,1,width,1],strides = [1,1,1,1],padding="VALID")
        return tf.squeeze(pool_output)
    
    my_maxpool_out = max_pool(my_convolution_output,width=5)
    sess.run(tf.global_variables_initializer())


    #fully connected
    def fullyConnected(input_layer,num_outputs):
            weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer),[num_outputs]]))#stack the tensors into 1 #https://www.tensorflow.org/api_docs/python/tf/stack
            weight = tf.random_normal(weight,stddev=0.1)#probably should be a variable to learn but who cares
            bias = tf.random_normal(shape=[num_outputs])

            #to multiply turn into 3d
            input_layer_2d = tf.expand_dims(input_layer,0)

            full_output  = tf.add(tf.matmul(input_layer_2d,weight),bias)
            return tf.squeeze(full_output)
    
    my_full_output = fully_connected(my_maxpool_out,5)


    pass

if __name__=="__main__":
    differentNNLauyers1D()