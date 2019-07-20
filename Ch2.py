import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

sess  = tf.Session()

def multiple_layers():
    #1
    x_shape = [1,4,4,1]
    x_val = np.random.uniform(size=x_shape)
    #2
    x_data = tf.placeholder(tf.float32,shape = x_shape)
    #3,4
    my_filt = tf.constant(0.25,shape=[2,2,1,1])
    my_strides = [1,2,2,1]
    mov_avg_layer = tf.nn.conv2d(x_data,my_filt,my_strides,padding="SAME",name="Moving_AVG")
    #5
    def custom_layer(input_mat):
        iMsq = tf.squeeze(input_mat)
        A = tf.constant([[1.,2.],[-1.,3.]])
        b = tf.constant(1.,shape=[2,2])
        temp1 = tf.matmul(A,iMsq)
        temp2 = tf.add(b,temp1)
        return (tf.sigmoid(temp2))
    #6
    with tf.name_scope("Custom_Layer") as scope:
        custom_layer1= custom_layer(mov_avg_layer)
    #7
    print(sess.run(custom_layer1,feed_dict={x_data:x_val}))

def losses():
    x_vals = tf.linspace(-1.,1.,500)
    target = tf.constant(0.)
    #regression losses
    #1 L2
    l2_vals = tf.square(target-x_vals)
    l2_out = sess.run (l2_vals)
    #2 L1
    l1_vals = tf.abs(target-x_vals)
    l1_out = sess.run(l1_vals)
    #3 pseudo-huber
    delta = tf.constant(0.25)
    phuber_vals = tf.mul(tf.square(delta),tf.sqrt(1. + tf.square((target-x_vals)/delta))-1.)

    #classification losses
def backProp():
    global sess
    def regr():
        batchSize = 20
        #3 create data and placeholders
        x_vals = np.random.normal(1,0.1,100)
        y_vals = np.repeat(10.,100)
        x_data = tf.placeholder(shape = [1,None],dtype = tf.float32)
        target = tf.placeholder(shape=[1,None],dtype = tf.float32)
        A = tf.Variable(tf.random_normal(shape=[batchSize,1]))
        #4
        mult_out  = tf.matmul(x_data,A)
        #5 lodd
        loss = tf.reduce_mean(tf.square(target-mult_out))
        #6 initialize the vatiables
        init = tf.global_variables_initializer()
        sess.run(init)
        #7 define the optimization algorithm and target
        opt= tf.train.GradientDescentOptimizer(learning_rate=0.02)
        train_step = opt.minimize(loss)
        #8 traininf
        for i in range(100):
            rand_index  = np.random.choice(100,size=batchSize)
            rand_x = [x_vals[rand_index]]
            rand_y = [y_vals[rand_index]]

            sess.run(train_step,feed_dict={x_data:rand_x,target:rand_y})
            if (i+1)%25==0:
                #print("A=",str(sess.run(A)))
                print("Loss",str(sess.run(loss,feed_dict={x_data:rand_x,target:rand_y})))
    def categ():
        #11
        x_vals = np.concatenate((np.random.normal(-1,1,50),np.random.normal(3,1,50)))
        y_vals = np.concatenate((np.repeat(0.,50),np.repeat(1.,50)))

        x_data = tf.placeholder(shape=[1],dtype=tf.float32)
        y_target = tf.placeholder(shape=[1],dtype=tf.float32)
        
        A = tf.Variable(tf.random_normal(mean=10,shape=[1]))
        #12
        my_out = tf.add(x_data,A)
        #13
        my_out_expanded = tf.expand_dims(my_out,0)
        y_target_expanded = tf.expand_dims(y_target,0)
        # #14
        sess.run(tf.global_variables_initializer())
        #15
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = my_out_expanded,labels = y_target_expanded)
        #16
        my_opt = tf.train.GradientDescentOptimizer(0.05)
        train_step = my_opt.minimize(xentropy)
        #17
        for i in range(1400):
            rand_index = np.random.choice(100)
            rand_x = [x_vals[rand_index]]
            rand_y = [y_vals[rand_index]]

            sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})

            if (i+1)%200==0:
                print("Acateg:",str(sess.run(A)))
                print("LosssCateg",str(sess.run(xentropy,feed_dict={x_data:rand_x,y_target:rand_y})))
        pass
    regr()

    #cleanup the graph
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    sess = tf.Session()

    categ()
def flowerData():
    #2
    iris  = datasets.load_iris()
    bin_target = np.array([1. if x==0 else 0. for x in iris.target])
    iris_features = np.array([[x[2],x[3]] for x in iris.data])
    #3
    batch_size=20
    x1_data = tf.placeholder(shape=[None,1],dtype = tf.float32)
    x2_data = tf.placeholder(shape=[None,1],dtype = tf.float32)
    y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    sess.run(tf.global_variables_initializer())
    #4
    my_mult = tf.matmul(x2_data,A)
    ma_add = tf.add(my_mult,b)
    my_out = tf.math.subtract(x1_data,ma_add)
    #5
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = my_out,labels=y_target)
    #6
    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)
    
    for i in range(1000):
        rand_i = np.random.choice(np.shape(bin_target)[0],size=batch_size)
        x1_rand =np.array([[x[0]] for x in iris_features[rand_i]])
        x2_rand =np.array([[x[1]] for x in iris_features[rand_i]])
        y_rand =np.array([[y] for y in bin_target[rand_i]]) 
        sess.run(train_step,feed_dict={x1_data:x1_rand,x2_data:x2_rand,y_target:y_rand})
        if (i+1)%200==0:
            print(str(sess.run(A)))

    #plotting
    [[slope]] = sess.run(A) 
    [[intercept]] = sess.run(b) 
    x = np.linspace(0, 3, num=50) 
    ablineValues = [] 
    for i in x:  
        ablineValues.append(slope*i+intercept) 
    setosa_x = [a[1] for i,a in enumerate(iris_features) if bin_target[i]==1] 
    setosa_y = [a[0] for i,a in enumerate(iris_features) if bin_target[i]==1] 
    non_setosa_x = [a[1] for i,a in enumerate(iris_features) if bin_target[i]==0]
    non_setosa_y = [a[0] for i,a in enumerate(iris_features) if bin_target[i]==0] 
    plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa''') 
    plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa') 
    plt.plot(x, ablineValues, 'b-') 
    plt.xlim([0.0, 2.7]) 
    plt.ylim([0.0, 7.1]) 
    plt.suptitle('Linear Separator For I.setosa', fontsize=20) 
    plt.xlabel('Petal Length') 
    plt.ylabel('Petal Width') 
    plt.legend(loc='lower right') 
    plt.show()
def evaluateModels():
    global sess
    def regr():
        #data
        sess  = tf.Session()
        x_vals = np.random.normal(1,0.1,100)
        y_vals = np.repeat(10.,100)
        x_data = tf.placeholder(shape=[None,1],dtype = tf.float32)   
        y_target = tf.placeholder(shape=[None,1],dtype = tf.float32)

        batch_size=25
        train_indecies = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace = False)
        test_indecies  = np.array(list(set(range(len(x_vals)))-set(train_indecies)))
        x_vals_tr = x_vals[train_indecies]
        y_vals_tr = y_vals[train_indecies]
        x_vals_test = x_vals[test_indecies]
        y_vals_test = y_vals[test_indecies]

        A = tf.Variable(tf.random_normal(shape=[1,1]))
        

        #model
        my_out = tf.matmul(x_data,A)
        loss = tf.reduce_mean(tf.square(my_out-y_target))

        init = tf.global_variables_initializer()
        sess.run(init)

        my_opt = tf.train.GradientDescentOptimizer(0.02)
        train_step = my_opt.minimize(loss)

        #training
        for  i in range(100):
            rand_index = np.random.choice(len(x_vals_tr),size = batch_size)
            rand_x = np.transpose([x_vals_tr[rand_index]])
            rand_y = np.transpose([y_vals_tr[rand_index]])
            sess.run(train_step,feed_dict = {x_data:rand_x,y_target:rand_y})
            if (i+1)%25==0:
                print("Loss=", str(sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})))

        #evaluating the model
        mse_test = sess.run(loss,feed_dict = {x_data:np.transpose([x_vals_test]),y_target:np.transpose([y_vals_test])})
        mse_train = sess.run(loss,feed_dict = {x_data:np.transpose([x_vals_tr]),y_target:np.transpose([y_vals_tr])})
        
        print("MSE test: ",str(np.round(mse_test,2)))
        print("MSE train: ",str(np.round(mse_train,2)))

    def classif():

        pass

    regr()
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    sess = tf.Session()
    classif()


if __name__=="__main__":
    #multiple_layers()

    #backProp()#backprop and bath
    #flowerData()
    evaluateModels()
    pass