import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from supportingFunctions import *
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.Session()
def simpleNearestN():
    #loading data
    try:
        housing_file = open("../data/housing.data")
    except IOError:
        print("no file opened")
    pass
    housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    num_features = len(cols_used)
    housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.read(-1).split('\n') if len(y)>=1]
    
    #preparing the data
    y_vals = np.transpose([np.array([y[13] for y in housing_data])]) 
    x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])
    x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)#ptp is max-min on the axis#https://docs.scipy.org/doc/numpy/reference/generated/numpy.ptp.html
    x_vals_train,x_vals_test,y_vals_train,y_vals_test = splitTrainTest(x_vals,y_vals)

    #batch and k value
    k = 4 
    batch_size=len(x_vals_test)


    #model
    x_data_train = tf.placeholder(shape=[None,num_features],dtype = tf.float32)
    x_data_test = tf.placeholder(shape=[None,num_features],dtype= tf.float32)

    y_target_train = tf.placeholder(shape=[None,1],dtype = tf.float32)
    y_target_test = tf.placeholder(shape=[None,1],dtype= tf.float32)

    #distance
    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train,tf.expand_dims(x_data_test,1))),axis = 2)

    #prediction
    top_k_xvals,top_k_indecies = tf.nn.top_k(tf.negative(distance),k=k)
    x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals,1),1)

    x_sums_repeated = tf.matmul(x_sums,tf.ones(shape=[1,k],dtype = tf.float32))#stack the sums colomns into a matrix

    x_val_weights = tf.expand_dims(tf.divide(top_k_xvals,x_sums_repeated),1)

    top_k_yvals = tf.gather(y_target_train,top_k_indecies)#gather to get slices of the tensor#https://www.tensorflow.org/api_docs/python/tf/gather

    prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals),axis = [1])#squeeze is used to remove all dimension of size 1#https://www.tensorflow.org/api_docs/python/tf/squeeze
    mse = tf.divide(tf.reduce_sum(tf.square(tf.subtract(prediction,y_target_test))),batch_size)

    num_loops = int(np.ceil(len(x_vals_test)/batch_size))

    for i in range(num_loops):
        min_index = i*batch_size
        max_index = min((i+1)*batch_size,len(x_vals_train))
        x_batch = x_vals_test[min_index:max_index]
        y_batch = y_vals_test[min_index:max_index] 

        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})    
        batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})
        print('Batch #' + str(i+1) + " MSE: " + str(np.round(batch_mse,3)))

def nearestNeighbourAndText():
    hypothesis = list("Bear")
    truth = list("Beers")
    h1 = tf.SparseTensor([[0,0,0],[0,0,1],[0,0,2],[0,0,3]],hypothesis,[1,1,1])
    t1 = tf.SparseTensor([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4]],truth,[1,1,1])
    
    print(sess.run(tf.edit_distance(h1,t1,normalize=False)))
    
    hypothesis2 = list('bearbeer') 
    truth2 = list('beersbeers') 
    h2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1], [0,1,2], [0,1,3]], hypothesis2, [1,2,4]) 
    t2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0], [0,1,1], [0,1,2], [0,1,3], [0,1,4]], truth2, [1,2,5])
    
    print(sess.run(tf.edit_distance(h2,t2,normalize=True)))
    
    hypothesis_words = ['bear','bar','tensor','flow'] 
    truth_word = ['beers'] 
    num_h_words = len(hypothesis_words)
    
    h_indices = [[xi, 0, yi] for xi,x in enumerate(hypothesis_words) for yi,y in enumerate(x)] 
    h_chars = list(''.join(hypothesis_words)) 
    h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words,1,1]) 
    truth_word_vec = truth_word*num_h_words 
    t_indices = [[xi, 0, yi] for xi,x in enumerate(truth_word_vec) for yi,y in enumerate(x)] 
    t_chars = list(''.join(truth_word_vec)) 
    t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words,1,1])
    print(sess.run(tf.edit_distance(h3, t3, normalize=True))) 
    
    pass
    # the same as previous but with sparseTensorValue
    def create_sparse_vec(word_list):    
        num_words = len(word_list)    
        indices = [[xi, 0, yi] for xi,x in enumerate(word_list) for yi,y in enumerate(x)]    
        chars = list(''.join(word_list))    
        return(tf.SparseTensorValue(indices, chars, [num_words,1,1]))#if we need sparse tensor outside the graph#https://www.tensorflow.org/api_docs/python/tf/SparseTensorValue
    
    hyp_string_sparse = create_sparse_vec(hypothesis_words) 
    truth_string_sparse = create_sparse_vec(truth_word*len(hypothesis_words))
    hyp_input = tf.sparse_placeholder(dtype=tf.string) 
    truth_input = tf.sparse_placeholder(dtype=tf.string)
    edit_distances = tf.edit_distance(hyp_input, truth_input, normalize=True)
    feed_dict = {hyp_input: hyp_string_sparse,truth_input: truth_string_sparse}             
    print(sess.run(edit_distances, feed_dict=feed_dict)) 

def mixedDistanceNearestNeighbor():
    #loading data
    try:
        housing_file = open("../data/housing.data")
    except IOError:
        print("no file opened")
    pass
    housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    num_features = len(cols_used)
    housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.read(-1).split('\n') if len(y)>=1]
    y_vals = np.transpose([np.array([y[13] for y in housing_data])]) 
    x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])
   
    #min-max scaling
    x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0) 


    #weight diagonal matrix
    weight_diagonal = x_vals.std(0)#compute the diagonal of the 0 axis#https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.std.html
    weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype=tf. float32) 
   
    x_vals_train,x_vals_test,y_vals_train,y_vals_test = splitTrainTest(x_vals,y_vals)

    #parameters
    k = 4 
    batch_size=len(x_vals_test) 
    
    #model
    x_data_train = tf.placeholder(shape=[None,num_features],dtype=tf.float32)
    x_data_test = tf.placeholder(shape=[None,num_features],dtype=tf.float32)
    y_target_train = tf.placeholder(shape=[None,1],dtype=tf.float32)
    y_target_test = tf.placeholder(shape=[None,1],dtype=tf.float32)

    #distance
    subtraction_term =  tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)) 
    first_product = tf.matmul(subtraction_term, tf.tile(tf.expand_dims(weight_matrix,0), [batch_size,1,1])) 
    second_product = tf.matmul(first_product,tf.transpose(subtraction_term,perm = [0,2,1]))#https://www.tensorflow.org/api_docs/python/tf/transpose
    distance = tf.sqrt(tf.matrix_diag_part(second_product))# returns the diagonals#https://www.tensorflow.org/api_docs/python/tf/linalg/diag_part
    #top k NN
    top_k_xvals,top_k_indices = tf.nn.top_k(tf.negative(distance),k=k)
    x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals,1),1)
    x_sums_repeated = tf.matmul(x_sums,tf.ones([1,k],tf.float32))
    x_val_weights = tf.expand_dims(tf.divide(top_k_xvals,x_sums_repeated), 1) 
    top_k_yvals = tf.gather(y_target_train, top_k_indices) 
    prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), squeeze_dims=[1]) 
    #mse
    mse = tf.divide(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

    #loop through the batches
    num_loops = int(np.ceil(len(x_vals_test)/batch_size)) 
    for i in range(num_loops):    
        min_index = i*batch_size
        max_index = min((i+1)*batch_size,len(x_vals_train))    
        x_batch = x_vals_test[min_index:max_index]    
        y_batch = y_vals_test[min_index:max_index]    
        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})    
        batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})    
        print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3))) 

    pass

def testingExpandDim():
    start_vec = np.array((83,69,45))
    print(start_vec)
    a = tf.expand_dims(start_vec, 0)
    b = tf.expand_dims(start_vec, 1)
    ab_sum = a + b
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        a = sess.run(a)
        b = sess.run(b)
        ab_sum = sess.run(ab_sum)

    print(a)
    print(b)
    print(ab_sum)

def mnistClassification():
    #loading the data
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    train_size = 1000
    test_size = 102
    rand_train_indices = np.random.choice(len(mnist.train.images), train_size, replace=False) 
    rand_test_indices = np.random.choice(len(mnist.test.images), test_size, replace=False) 
    x_vals_train = mnist.train.images[rand_train_indices] 
    x_vals_test = mnist.test.images[rand_test_indices] 
    y_vals_train = mnist.train.labels[rand_train_indices] 
    y_vals_test = mnist.test.labels[rand_test_indices]


    #params
    k = 4 
    batch_size=6
    #model
    x_data_train = tf.placeholder(shape=[None, 784], dtype=tf.float32) 
    x_data_test = tf.placeholder(shape=[None, 784], dtype=tf.float32) 
    y_target_train = tf.placeholder(shape=[None, 10], dtype=tf. float32) 
    y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32) 

    print(np.shape(x_vals_train))
    print(np.shape(x_vals_test))

    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train,tf.expand_dims(x_data_test,1))),axis=2)#expand dimensions so that we calc distance with each point
    
    #finding top distance
    top_k_xvals,top_k_indices = tf.nn.top_k(tf.negative(distance),k=k)
    prediction_indecies = tf.gather(y_target_train,top_k_indices)
    count_of_predictions = tf.reduce_sum(prediction_indecies,axis=1)
    prediction = tf.argmax(count_of_predictions,axis=1)

    num_loops = int(np.ceil(len(x_vals_test)/batch_size)) 
    test_output = [] 
    actual_vals = [] 
    for i in range(num_loops):   
        print(i) 
        min_index = i*batch_size    
        max_index = min((i+1)*batch_size,len(x_vals_train))    
        x_batch = x_vals_test[min_index:max_index]    
        y_batch = y_vals_test[min_index:max_index]    
        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})    
        
        test_output.extend(predictions)    
        actual_vals.extend(np.argmax(y_batch, axis=1))

    accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]]) 
    print('Accuracy on test set: ' + str(accuracy)) 
    pass
if __name__=="__main__":
    mnistClassification()