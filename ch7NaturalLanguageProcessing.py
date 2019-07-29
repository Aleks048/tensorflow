import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

import string 
import requests 
import io 
from zipfile import ZipFile 
from tensorflow.contrib import learn 

import nltk
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from supportingFunctions import *

from tensorflow.python.framework import ops

import collections
sess = tf.Session() 

def bagOfWords():
    #loading the data
    text_data = load_word_data()
    text_data = [x[0].split('\t') for x in text_data if len(x)==1]
    texts = [x[1] for x in text_data]
    target = [0 if "spam" in x[0] else 1 for x in text_data]
    texts = preprocess_word_data(texts)

    sentence_size = 25 
    min_word_freq = 3

    #preprocessing data with tf
    vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size,min_frequency=min_word_freq)
    vocab_processor.fit_transform(texts)
    embeding_size = len(vocab_processor.vocabulary_)
    #train test split
    train_indices = np.random.choice(len(texts), round(len(texts)*0.8), replace=False) 
    test_indices = np.array(list(set(range(len(texts))) - set(train_indices))) 
    texts_train = [x for ix, x in enumerate(texts) if ix in train_indices] 
    texts_test = [x for ix, x in enumerate(texts) if ix in test_indices] 
    target_train = [x for ix, x in enumerate(target) if ix in train_indices] 
    target_test = [x for ix, x in enumerate(target) if ix in test_indices]
    #identity matrix used to look up sparse vector for each word
    identity_mat = tf.diag(tf.ones(shape=[embeding_size]))

    #model
    A = tf.Variable(tf.random_normal(shape = [embeding_size,1]))
    b = tf.Variable(tf.random_normal(shape = [1,1]))
    x_data = tf.placeholder(shape=[sentence_size],dtype=tf.int32)
    y_target = tf.placeholder(shape=[1,1],dtype=tf.float32)

    x_embed = tf.nn.embedding_lookup(identity_mat,x_data)#basicly 1-hot encoding #https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
    x_col_sums = tf.reduce_sum(x_embed,0)

    x_col_sums_2d = tf.expand_dims(x_col_sums,0)
    model_output  = tf.add(tf.matmul(x_col_sums_2d,A),b)
    #loss,pdrediction
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output,labels = y_target))
    prediction = tf.sigmoid(model_output)
    #optimizer
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #init
    sess.run(tf.global_variables_initializer())
    #training
    loss_vec = [] 
    train_acc_all = [] 
    train_acc_avg = [] 
    for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):    
        y_data = [[target_train[ix]]]        
        sess.run(train_step, feed_dict={x_data: t, y_target: y_data})        
        temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})  
        loss_vec.append(temp_loss)
        if (ix+1)%10==0:        
            print('Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))
    pass
def tf_idf():
    #param
    batch_size= 200 
    max_features = 1000
    #loading the data
    text_data = load_word_data()

    text_data = [x[0].split('\t') for x in text_data if len(x)==1]
    texts = [x[1] for x in text_data]
    target = [0 if "spam" in x[0] else 1 for x in text_data]
    texts = preprocess_word_data(texts)
    #tokenizing sentences
    def tokenizer(text):    
        words = nltk.word_tokenize(text)    
        return words 
    # Create TF-IDF of texts 
    tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features) 
    sparse_tfidf_texts = tfidf.fit_transform(texts) 
    #train/test split
    train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
    test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices))) 
    texts_train = sparse_tfidf_texts[train_indices] 
    texts_test = sparse_tfidf_texts[test_indices] 
    target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices]) 
    target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
    #model
    A = tf.Variable(tf.random_normal(shape=[max_features,1])) 
    b = tf.Variable(tf.random_normal(shape=[1,1])) 
    x_data = tf.placeholder(shape=[None, max_features], dtype=tf. float32) 
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    model_output = tf.add(tf.matmul(x_data, A), b) 

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits =model_output, labels=y_target))
    prediction = tf.round(tf.sigmoid(model_output)) 
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32) 
    accuracy = tf.reduce_mean(predictions_correct) 
    
    my_opt = tf.train.GradientDescentOptimizer(0.0025) 
    train_step = my_opt.minimize(loss) 
    # Intitialize Variables 
    sess.run(tf.global_variables_initializer()) 

    #train
    for i in range(10000):    
        rand_index = np.random.choice(texts_train.shape[0], size=batch_size)    
        rand_x = texts_train[rand_index].todense()    
        rand_y = np.transpose([target_train[rand_index]])    
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    pass
def skip_gram():
    sess = tf.Session()

    ''' MODEL PARAMETERS '''

    batch_size = 100#size of the batch of data fed into the model
    embedding_size = 200# the  dimension of the space we put the words to 
    vocabulary_size = 10000# how many words we will consider when create the voc for the model
    generations = 1#num of training iterations
    print_loss_every = 2000#used for loss printing

    num_sampled = int(batch_size/2) # Number of negative examples to sample???????????.
    window_size = 2 # How many words to consider left and right.
    top_k = 5 # top_k nearest words are printed for the validation word

    # Declare stop words
    stops = stopwords.words('english')#the words not considered in the dictionary

    # We pick five test words. We are expecting synonyms to appear
    # this are the words used for validation of the algorithm# the selected nearby words will be printed
    print_valid_every = 5000
    valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']
    # Later we will have to transform these into indices



    ''' LOAD THE DATA '''
    pos_data,neg_data,target = load_review_data() 
    texts = pos_data + neg_data

    ''' DATA PREPROCESSING '''
    # Normalize text# to lower/ remove punctuation,numbers,stopwords,extra whitespace
    def normalize_text(texts, stops):
        # Lower case
        texts = [x.lower() for x in texts]
        # Remove punctuation
        texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
        # Remove numbers
        texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
        # Remove stopwords
        texts = [' '.join([word for word in x.split() if word not in stops]) for x in texts]
        # Trim extra whitespace
        texts = [' '.join(x.split()) for x in texts]
        return texts
    texts = normalize_text(texts, stops)

    # Remove all the texts that have less then 3 words
    target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
    texts = [x for x in texts if len(x.split()) > 2]

    # Build dictionary of words
    def build_dictionary(sentences, vocabulary_size):

        # Turn sentences (list of strings) into lists of words
        split_sentences = [s.split() for s in sentences]#turn each sentence into array of words
        words = [x for sublist in split_sentences for x in sublist]#an array consisting of the words all of the sentences

        # Initialize list of [word, word_count] for each word, starting with unknown
        count = [['RARE', -1]]
        # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
        #the first element would be ['RARE',-1], the rest would be (word,num_of_occurances) decreasing#https://docs.python.org/2/library/collections.html
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        
        # Turn the count into the dictionary
        word_dict = {}
        # we rank the words by the number of appearances but  discard the number of appearances with rare being at 0 as exception
        for word, word_count in count:
            word_dict[word] = len(word_dict)
        return word_dict

    # Turn text data(sentences) into the list of integers
    def text_to_numbers(sentences, word_dict):

        data = []

        for sentence in sentences:
            sentence_data = []
            # For each word, either use selected index or rare word index
            for word in sentence.split(' '):
                if word in word_dict:
                    word_ix = word_dict[word]
                else:
                    word_ix = 0
                sentence_data.append(word_ix)
            data.append(sentence_data)
        return data

    ''' VOCALBULARY '''
    # Build our data set and dictionaries
    word_dictionary = build_dictionary(texts, vocabulary_size)
    word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))#we switch the keys and values in the dictionary to have the voc code as the key/representation purpose
    text_data = text_to_numbers(texts, word_dictionary)
    
    # Get validation words codes in the vocabulary
    valid_examples = [word_dictionary[x] for x in valid_words]

    # Generate data randomly (N words behind, target, N words ahead)
    def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):

        ''' Fill up data batch '''
        batch_data = []
        label_data = []

        while len(batch_data) < batch_size:#until we filled the batch_data

            # select random sentence
            rand_sentence = np.random.choice(sentences)

            # Generate consecutive windows to look at
            #the end will be cutof automaticaly/ex:[,,,,],[,,,],[,,]
            window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in enumerate(rand_sentence)]
            
            # Denote which element of each window is the word of interest /ex [0,1,2,2,2,2.....]
            label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

            if method == 'skip_gram':
                '''here we are looking for a bunch of words that surround the target.'''
                # separate into the tuple of the form (word_of_interest,[surrounding_words])
                batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x, y in zip(window_sequences, label_indices)]

                # Make it in to a list of tuples (target_word, surrounding_word)
                tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
            
            elif method == 'cbow':
                '''here we are looking for 1 word that is connected to the surrounding target so we flip surrounding and target from skip_gram'''
                # separate into the tuple of the form ([surrounding_words],word_of_interest)
                batch_and_labels = [(x[:y] + x[(y + 1):], x[y]) for x, y in zip(window_sequences, label_indices)]

                # Make it in to a list of tuples (target word, surrounding word)
                tuple_data = [(x_, y) for x, y in batch_and_labels for x_ in x]

            else:

                raise ValueError('Method {} not implemented yet.'.format(method))

            # extract batch and labels
            #given a list of tupples we put the first element into the batch and second into the labels
            batch, labels = [list(x) for x in zip(*tuple_data)]#*list to get the elements of the list we use it to pass them as parameters

            #add the data from the sentence into the output data
            batch_data.extend(batch[:batch_size])
            label_data.extend(labels[:batch_size])

        # Convert to numpy array
        batch_data = np.array(batch_data)
        label_data = np.transpose(np.array([label_data]))#transpose the target so that we have a column vector
        
        return batch_data, label_data

    ''' PLACEHIOLDERS '''
    x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(shape=valid_examples, dtype=tf.int32)

    ''' EMBEDDINGS '''
    # Embeddings for the words in the word_dict:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))# random_uniform from -1 to 1
    # Lookup the input word embedding:
    embed = tf.nn.embedding_lookup(embeddings, x_inputs)#we are given the the words and get the embeddings for them

    ''' LOSS needs to be verified'''
    #NCE
    #stackexchange#https://datascience.stackexchange.com/questions/13216/intuitive-explanation-of-noise-contrastive-estimation-nce-loss
    #article explaining theory#https://leimao.github.io/article/Noise-Contrastive-Estimate/
    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # Get loss from prediction
    loss = tf.reduce_mean(tf.nn.nce_loss(#https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss
                                        weights=nce_weights,

                                        biases=nce_biases,

                                        labels=y_target,

                                        inputs=embed,

                                        num_sampled=num_sampled,

                                        num_classes=vocabulary_size))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
    
    ''' MEASURE SIMILARITY '''
    #Cosine similarity between words/we measure the distance between/we use it to print the words that the model thinks should be together
    #normalize embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    #get embeddings for the validation words
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    #measure similarity
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)#note: we transpose the normalized embedding matrix 

    # Add variable initializer.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the skip gram model.

    loss_vec = []
    loss_x_vec = []

    for i in range(generations):

        batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)#get the batch of data
        feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}#create feed dict

        # Run the train step
        sess.run(optimizer, feed_dict=feed_dict)

        # Return the loss /representation
        if (i + 1) % print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i+1)
            print('Loss at step {} : {}'.format(i+1, loss_val))        

        # Validation: Print some random words and top 5 related words
        if (i+1) % print_valid_every == 0:

            sim = sess.run(similarity, feed_dict=feed_dict)

            for j in range(len(valid_words)):

                valid_word = valid_words[j]

                nearest = (-sim[j, :]).argsort()[1:top_k+1]#????

                log_str = "Nearest to {}:".format(valid_word)

                for k in range(top_k):

                    close_word = word_dictionary_rev[nearest[k]]

                    log_str = '{} {},'.format(log_str, close_word)

                print(log_str)

if __name__=="__main__":
    skip_gram()
    ops.reset_default_graph()