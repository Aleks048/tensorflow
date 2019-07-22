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
        iris = datasets.load_iris()
        x_vals = np.array([x[3] for x in iris.data])
        y_vals = np.array([y[0] for y in iris.data])
        x_valsTr,x_valsTest,y_valsTr,y_valsTest = splitTrainTest(x_vals,y_vals)
        
        x_data = tf.placeholder(shape=[None,1],dtype = tf.float32)
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
                "b":b
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
        pass

   # linearSVM()
    svmAsLinReg()

    pass

if __name__=="__main__":
    SVM()