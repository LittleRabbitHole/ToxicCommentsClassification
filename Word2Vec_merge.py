#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:36:29 2018

@author: angli
"""
import pickle
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# get word vec
def get_W2V(word):
    #word = 'ferguson'
    #print 'querying: {}'.format(word)
    try:
        vec = np.array(model[word])
    
        #print 'shape: {}'.format(vec.shape)
        #print 'sample: '
        #print vec[1:10]
    except Exception:
        #print ('word: "{}" is not in the vocabulary of the model located at: "{}"'.format(word, model_path))
        vec = np.zeros((200,))
    
    return vec 

if __name__ == "__main__":
    # load model
    model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
    model_path = '{}/w2vmodel'.format(model_dir_path)
    model = Word2Vec.load('{}/cleantxt_200.w2v'.format(model_path))
    
    # get comments
    all_comments = pd.read_csv("/home/ang/Comments/test_cleaned.csv")
    #all_comments['w2v'] = np.nan
    #all_comments['w2v'] = all_comments['w2v'].astype(object)
    
    #store in dict
    comment2vec = {}
    
    n=0 #recording # of comments that has no vecs
    m=0 #recording # of empty comments
    for index, row in all_comments.iterrows():
        comment_id = row['id']
        cleancomment = row['clean_comment']
        
        if index % 10000 == 0: print (index)
        
        #check nan
        if cleancomment is not np.nan:
            wordlist = cleancomment.split(" ")
            
            if len(wordlist)>0:
                #get the tweet vec as mean
                comment_vec = np.zeros((200,))
                
                nwords = 0
                for word in wordlist:
                    try:
                        word_vec = np.array(model[word])
                        comment_vec += word_vec
                        nwords += 1
                    except Exception:
                        #print ('word: "{}" is not in the vocabulary of the model located at: "{}"'.format(word, model_path))
                        word_vec = np.zeros((200,))
                #mean
                comment_vec_mean = comment_vec/nwords
                
                if not np.isnan(comment_vec_mean[0]): 
                    #set value
                    comment2vec[comment_id] = comment_vec_mean
                else:
                    n+=1
                    print ("index " + str(index)+ " has not result ...")
        else:
            m += 1
    #save to pickle
    with open('/home/ang/Comments/test_c2v.pkl', 'wb') as f:
        pickle.dump(comment2vec, f)
    #comment2vec.to_pickle("/home/ang/Comments/train_c2v.pkl")
    print ('num of comments that has no vecs is: '+str(n) + '\n' + "# of empty comments: "+str(m))
    #df = pd.read_pickle("/home/ang/BLM_hashtags/aggre_data/aggre_tweet_hashtag_t2v.pkl")
    #clean train: 159505

