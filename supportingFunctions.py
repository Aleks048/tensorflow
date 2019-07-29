import numpy as np
import os  
import csv 
import string 

def splitTrainTest(x_vals,y_vals):
    '''
    splits the data into train and test in 0.8/0.2
    '''
    trInd = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace =False)
    testInd = np.array(list(set(range(len(x_vals)))-set(trInd))) 
    return x_vals[trInd],x_vals[testInd],y_vals[trInd],y_vals[testInd]

def train(args):
    data = args["data"]
    s = args["session"]
    m = args["measures"]

   
    for i in range(args["epochs"]):
        randInd = np.random.choice(len(data["x_valsTr"]),size=args["batch_size"])
        rand_x = data["x_valsTr"][randInd]
        rand_y = np.transpose([data["y_valsTr"][randInd]])

        feed_dict = {s["x_data"]:rand_x,s["y_target"]:rand_y}

        s["sess"].run(s["train_step"],feed_dict=feed_dict)

        #measures
        m["loss_vec"].append(s["sess"].run(s["loss"],feed_dict=feed_dict))
        if "train_acc" in m.keys():
            m["train_acc"].append(s["sess"].run(s["accuracy"],feed_dict=feed_dict))
        if "test_acc" in m.keys():
            m["test_acc"].append(s["sess"].run(s["accuracy"],feed_dict ={s["x_data"]:data["x_valsTest"],s["y_target"]:np.transpose([data["y_valsTest"]])}))

        if (i+1)%m["modNumMeasurements"]==0:
            print("Step :",i+1)
            if "A" in s.keys():
                print("A :",s["sess"].run(s["A"]))
            if "b" in s.keys():
                print("b :",s["sess"].run(s["b"])) 
            print("Loss", s["sess"].run(s["loss"],feed_dict=feed_dict))   

#natural language
def load_word_data():
    text_data = []
    if os.path.isfile('../data/SMSSpamCollection.csv'):
        print("file exists")
        with open('../data/SMSSpamCollection.csv', 'r') as temp_output_file:        
            reader = csv.reader(temp_output_file)        
            for row in reader:            
                text_data.append(row)
    return text_data 
def load_review_data():
    pos_data = []
    neg_data = []     
    
    for (_,_,filenames) in os.walk("../data/txt_sentoken/pos/"):
        for f in filenames:
            pos_data.append(open(os.path.join('../data/txt_sentoken/pos/',f)).read())
    for (_,_,filenames) in os.walk("../data/txt_sentoken/neg/"):
        for f in filenames:
            neg_data.append(open(os.path.join('../data/txt_sentoken/neg/',f)).read())
    target = [1]*len(pos_data)+[0]*len(neg_data)
    return pos_data,neg_data,target
    
def preprocess_word_data(texts):
    #data formatting
    texts = [x.lower() for x  in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x  in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x  in texts]
    texts = [' '.join(x.split()) for x in texts]
    return texts