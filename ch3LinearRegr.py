import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
import requests

sess = tf.Session()

def generatingData():
    #generating data
    x_vals = np.linspace(0,10,100)

    y_vals = x_vals + np.random.normal(0,1,100)

    #creating the matrices needed
    x_vals_col = np.transpose(np.matrix(x_vals))
    ones = np.transpose(np.matrix(np.repeat(1,100)))

    A = np.column_stack((x_vals,ones))
    b= np.transpose(np.matrix(y_vals))

    #turning data into tensors
    A_tensor = tf.constant(A)
    b_tensor = tf.constant(b)
    return A_tensor,b_tensor

def linRegrMatrInv():
    A_tensor,b_tensor = generatingData()

    #calculating regression
    AtrA = tf.matmul(tf.transpose(A_tensor),A_tensor)
    solution = tf.matmul(tf.matmul(tf.matrix_inverse(AtrA),
                tf.transpose(A_tensor)),b_tensor)
    solEval = sess.run(solution)
    print("slope: ",solEval[0][0])
    print("intercept: ",solEval[1][0])

    pass

def linRegrCholeskyDecomposition():
    A_tensor,b_tensor = generatingData()
    #Cholesky decomposition
    L = tf.cholesky(tf.matmul(tf.transpose(A_tensor),A_tensor))
    
    tA_b = tf.matmul(tf.transpose(A_tensor),b_tensor)
    sol1 = tf.matrix_solve(L,tA_b)
    sol2 = tf.matrix_solve(tf.transpose(L),sol1)

    solEval = sess.run(sol2)
    print("slope: ",solEval[0][0])
    print("intercept: ",solEval[1][0])

def tensorflowReg():
    iris = datasets.load_iris()

    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([y[0] for y in iris.data])

    lr =0.05
    batch_size = 25
    x_data = tf.placeholder(shape=[None,1],dtype = tf.float32)
    y_target = tf.placeholder(shape=[None,1],dtype = tf.float32)
    
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    #model
    out= tf.add(tf.matmul(x_data,A),b)

    loss = tf.reduce_mean(tf.square(y_target-out))

    #initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    myopt = tf.train.GradientDescentOptimizer(lr)
    train_step =myopt.minimize(loss)


    #train
    for i in range(100):
        ind = np.random.choice(len(x_vals),size=batch_size)
        rand_x = np.transpose([x_vals[ind]])
        rand_y = np.transpose([y_vals[ind]])

        sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
        if (i+1)%25==0:
            print(sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y}))

def logisticRegression():
    bFile = requests.get('http://www.umass.edu/statdata/statdata/data/lowbwt.dat')#data link is not valid
    bData = bFile.text.split('\r\n')[5:] 
    bData = [[float(x) for x in y.split('') if len(x)>=1] for y in bData[1:] if len(y)>=1]
    
    y_vals = np.array([x[1] for x in bData])
    x_vals = np.array([x[2:9] for x in bData])

    #split the train test data
    indTr = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
    indTest = np.array(list(set(range(len(x_vals)))-set(indTr)))
    print(bData)
    xTr = x_vals[indTr]
    xTest = x_vals[indTest]
    yTr = y_vals[indTr]
    yTest = y_vals[indTest]

    #normalizing columns
    def norm_cols(m):
        col_max = m.max(axis=0)
        col_min = m.min(axis=0)
        return (m-col_min)/(col_max-col_min)

    x_vals_train = np.nan_to_num(norm_cols(xTr))#nan_to_num to ensure we have a number sisnce dividing
    x_vals_test  = np.nan_to_num(norm_cols(xTest))

    #model
    batch_size = 25
    x_data = tf.placeholder(shape=[None,7],dtype= tf.float32)
    y_target = tf.placeholder(shape=[None,1],dtype = tf.float32)
    A = tf.Variable(tf.random_normal(shape=[7,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    out = tf.add(tf.matmul(x_data,A),b)
    
    #loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out,y_target))
    #init
    sess.run(tf.global_variables_initializer())
    
    myopt = tf.train.GradientDescentOptimizer(0.01)
    train_step = myopt.minimize(loss)

    #collecting predictions
    prediction = tf.round(tf.sigmoid(out))
    pred_corr = tf.cast(tf.equal(prediction,y_target),tf.float32)
    accuracy = tf.reduce_mean(pred_corr)

    for i in range(1500):
        batchInd = np.random.choice(range(len(xTr)),size=batch_size)
        batchX = x_vals_train[batchInd]
        batchY = np.transpose(yTr[batchInd])

        sess.run(train_step,feed_dict={x_data:batchX,y_target:batchY})
        if (i+1)%500==0:
            print(sess.run(accuracy,feed_dict={x_data:batchX,y_target:batchY}))



if __name__=="__main__":
    #linRegrMatrInv()
    #linRegrCholeskyDecomposition()
    #tensorflowReg()
    logisticRegression()