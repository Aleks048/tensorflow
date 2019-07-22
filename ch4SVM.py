import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from supportingFunctions import *
sess = tf.Session()

def SVM():
    
    def linearSVM():
        def plot(args):
            data = args["data"]
            s = args["session"]
            m = args["measures"]

            [[a1],[a2]]= s["sess"].run(s["A"])
            [[b]] = s["sess"].run(s["b"])
            slope = -a2/a1
            y_intercept = b/a1

            x1_vals = [d[1] for d in data["x_vals"]]

            best_fit = [] 
            for i in x1_vals:  
                best_fit.append(slope*i+y_intercept)

            setosa_x = [d[1] for i,d in enumerate(data["x_vals"]) if data["y_vals"][i]==1] 
            setosa_y = [d[0] for i,d in enumerate(data["x_vals"]) if data["y_vals"][i]==1]
            not_setosa_x = [d[1] for i,d in enumerate(data["x_vals"]) if data["y_vals"][i]==-1] 
            not_setosa_y = [d[0] for i,d in enumerate(data["x_vals"]) if data["y_vals"][i]==-1] 

            plt.plot(setosa_x, setosa_y, 'o', label='I. setosa') 
            plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa') 
            plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3) 
            plt.ylim([0, 10]) 
            plt.legend(loc='lower right') 
            plt.title('Sepal Length vs Pedal Width') 
            plt.xlabel('Pedal Width') 
            plt.ylabel('Sepal Length') 
            plt.show()
            plt.plot(m["test_acc"], 'r--', label='Test Accuracy')
            plt.title('Train and Test Set Accuracies') 
            plt.xlabel('Generation') 
            plt.ylabel('Accuracy') 
            plt.legend(loc='lower right') 
            plt.show()
            plt.plot(m["loss_vec"], 'k-') 
            plt.title('Loss per Generation') 
            plt.xlabel('Generation') 
            plt.ylabel('Loss') 
            plt.show()


            pass
        iris = datasets.load_iris()
        #loading data
        x_vals = np.array([[x[0],x[3]] for x in iris.data])
        y_vals = np.array([1 if y==0 else -1 for y in iris.target])
        #train test split
        x_valsTr,x_valsTest,y_valsTr,y_valsTest = splitTrainTest(x_vals,y_vals)
        #print(x_valsTr,x_valsTest,y_valsTr,y_valsTest)
        #model
        x_data = tf.placeholder(shape=[None,2],dtype = tf.float32)
        y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

        A = tf.Variable(tf.random_normal(shape=[2,1]))
        b = tf.Variable(tf.random_normal(shape=[1,1]))

        model_out = tf.subtract(tf.matmul(x_data,A),b)#hyperplane equation
        #loss
        l2_norm = tf.reduce_sum(tf.square(A))
        alpha = tf.constant(0.1)
        classification_term = tf.reduce_mean(tf.maximum(0.,tf.subtract(1.,tf.multiply(model_out,y_target))))

        loss = tf.add(classification_term,tf.multiply(alpha,l2_norm))

        #prediction and accuracy measures
        prediction = tf.sign(model_out)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,y_target),tf.float32))
        
        #optimizer and train step
        myopt = tf.train.GradientDescentOptimizer(0.01)
        train_step = myopt.minimize(loss)
        #init
        sess.run(tf.global_variables_initializer())

        #training    
        args = {
            "data":{
                "x_vals":x_vals,
                "y_vals":y_vals,
                "x_valsTr":x_valsTr,
                "y_valsTr":y_valsTr,
                "x_valsTest" :x_valsTest,
                "y_valsTest":y_valsTest
                },
            "session":{
                "sess":sess,
                "train_step":train_step,
                "x_data":x_data,
                "y_target":y_target,
                "loss":loss,
                "accuracy":accuracy,
                "A":A,
                "b":b
                },
            "batch_size":100,
            "epochs":500,
            "measures":{
                "loss_vec":[],
                "train_acc":[],
                "test_acc":[],
                "modNumMeasurements":100
            }
        }
        train(args)
        plot(args)

    def svmAsLinReg():
        def plot(args):
            data = args["data"]
            s = args["session"]
            m = args["measures"]

            [[slope]] = s["sess"].run(s["A"]) 
            [[y_intercept]] = s["sess"].run(s["b"]) 
            [width] = s["sess"].run(s["epsilon"])
            best_fit = [] 
            best_fit_upper = [] 
            best_fit_lower = []
            for i in data["x_vals"]:
                best_fit.append(slope*i+y_intercept)  
                best_fit_upper.append(slope*i+y_intercept+width)  
                best_fit_lower.append(slope*i+y_intercept-width) 
            plt.plot(data["x_vals"], data["y_vals"], 'o', label='Data Points') 
            plt.plot(data["x_vals"], best_fit, 'r-', label='SVM Regression Line', linewidth=3)
            plt.plot(data["x_vals"], best_fit_upper, 'r--', linewidth=2) 
            plt.plot(data["x_vals"], best_fit_lower, 'r--', linewidth=2) 
            plt.ylim([0, 10]) 
            plt.legend(loc='lower right') 
            plt.title('Sepal Length vs Pedal Width') 
            plt.xlabel('Pedal Width') 
            plt.ylabel('Sepal Length') 
            plt.show() 
            plt.plot(m["loss_vec"], 'k-', label='Train Set Loss') 
            plt.title('L2 Loss per Generation') 
            plt.xlabel('Generation') 
            plt.ylabel('L2 Loss') 
            plt.legend(loc='upper right') 
            plt.show()
            
        iris = datasets.load_iris()
        x_vals = np.array([[x[3]] for x in iris.data])
        y_vals = np.array([y[0] for y in iris.data])
        x_valsTr,x_valsTest,y_valsTr,y_valsTest = splitTrainTest(x_vals,y_vals)
        
        x_data = tf.placeholder(shape=[None,1],dtype = tf.float32,name = "xIn")
        y_target = tf.placeholder(shape = [None,1],dtype = tf.float32)

        A = tf.Variable(tf.random_normal(shape=[1,1]))
        b = tf.Variable(tf.random_normal(shape=[1,1]))

        modelOut = tf.add(tf.matmul(x_data,A),b)
        #loss
        eps = tf.constant([0.5])
        loss = tf.reduce_mean(tf.maximum(0.,tf.subtract(tf.abs(tf.subtract(modelOut,y_target)),eps)))

        myOpt = tf.train.GradientDescentOptimizer(0.075)
        train_step = myOpt.minimize(loss)
        #init
        sess.run(tf.global_variables_initializer())

        args = {
            "data":{
                "x_vals":x_vals,
                "y_vals":y_vals,
                "x_valsTr":x_valsTr,
                "y_valsTr":y_valsTr,
                "x_valsTest" :x_valsTest,
                "y_valsTest":y_valsTest
                },
            "session":{
                "sess":sess,
                "train_step":train_step,
                "x_data":x_data,
                "y_target":y_target,
                "loss":loss,
                #"accuracy":accuracy,
                "A":A,
                "b":b,
                "epsilon":eps
                },
            "batch_size":50,
            "epochs":200,
            "measures":{
                "loss_vec":[],
                # "train_acc":[],
                # "test_acc":[],
                "modNumMeasurements":50
            }
        }
        train(args)
        plot(args)
        pass

    def kernelSVM():
        
        #creating dataset
        (x_vals,y_vals) = datasets.make_circles(n_samples=500,factor=.5,noise=.1)
        
        y_vals = np.array([1 if y==1 else -1 for y in y_vals])
        class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
        class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
        class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
        class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]


        #model
        batch_size=250
        x_data = tf.placeholder(shape=[None,2],dtype=tf.float32)
        y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)
        prediction_grid = tf.placeholder(shape=[None,2],dtype=tf.float32)
        b = tf.Variable(tf.random_normal(shape=[1,batch_size]))
        
        #gaussian kernel
        gamma = tf.constant(-50.0)
        dist = tf.reduce_mean(tf.square(x_data),1)
        dist = tf.reshape(dist,[-1,1])#-1 for shape not known#https://www.tensorflow.org/api_docs/python/tf/reshape
        sq_dists = tf.add(tf.subtract(dist,tf.multiply(2.,tf.matmul(x_data,tf.transpose(x_data)))),tf.transpose(dist))
        my_kernel = tf.exp(tf.multiply(gamma,tf.abs(sq_dists)))

        #stating the dual problem
        model_output = tf.matmul(b,my_kernel)
        f_term = tf.reduce_sum(b)
        b_vec_c = tf.matmul(tf.transpose(b),b)
        y_targ_c = tf.matmul(y_target,tf.transpose(y_target))
        s_term = tf.reduce_sum(tf.multiply(my_kernel,tf.multiply(b_vec_c,y_targ_c)))
        loss = tf.negative(tf.subtract(f_term,s_term))

        #prediction and acc functions
        rA = tf.reshape(tf.reduce_sum(tf.square(x_data),1),[-1,1])
        rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid),1),[-1,1])

        pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB)) 
        pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
        prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target),b), pred_kernel) 
        prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output)) 
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))
        #optimizer and init
        my_opt = tf.train.GradientDescentOptimizer(0.001)
        trainStep = my_opt.minimize(loss)
        sess.run(tf.global_variables_initializer())

        loss_vec = [] 
        batch_accuracy = [] 
        for i in range(500):    
            rand_index = np.random.choice(len(x_vals), size=batch_size)    
            rand_x = x_vals[rand_index]    
            rand_y = np.transpose([y_vals[rand_index]])    
            sess.run(trainStep, feed_dict={x_data: rand_x, y_target: rand_y})        
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            loss_vec.append(temp_loss)        
            acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,y_target: rand_y,prediction_grid:rand_x})
            batch_accuracy.append(acc_temp)
            if (i+1)%100==0:        
                print('Step #' + str(i+1))        
                print('Loss = ' + str(temp_loss))
        pass
   # linearSVM()
    # svmAsLinReg()
    kernelSVM()

    pass

if __name__=="__main__":
    SVM()