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
    
    # get tweets
    #all_word_lst = pickle.load( open( "aggre_tweet_hashtag_cleantweet.pkl", "rb" ) )
    all_tweets = pd.read_csv("/home/ang/BLM_hashtags/aggre_data/aggre_tweet_hashtag_cleantweet.csv")
    all_tweets = all_tweets.drop_duplicates()
    all_tweets['w2v'] = np.nan
    all_tweets['w2v'] = all_tweets['w2v'].astype(object)
    
    n=0
    for index, row in all_tweets.iterrows():
        cleantweet = row['tweettext_clean']
        
        if index % 10000 == 0: print (index)
        
        #check nan
        if cleantweet is not np.nan:
            wordlist = cleantweet.split(" ")
            
            if len(wordlist)>0:
                #get the tweet vec as mean
                tweet_vec = np.zeros((200,))
                
                nwords = 0
                for word in wordlist:
                    try:
                        word_vec = np.array(model[word])
                        tweet_vec += word_vec
                        nwords += 1
                    except Exception:
                        #print ('word: "{}" is not in the vocabulary of the model located at: "{}"'.format(word, model_path))
                        word_vec = np.zeros((200,))
                        n+=1
                #mean
                tweet_vec_mean = tweet_vec/nwords
                if not np.isnan(tweet_vec_mean[0]): 
                    #set value
                    all_tweets = all_tweets.set_value(index, 'w2v', tweet_vec_mean)
                else:
                    print ("index " + str(index)+ " has not result ...")

    #save to pickle
    all_tweets.to_pickle("/home/ang/BLM_hashtags/aggre_data/aggre_tweet_hashtag_t2v.pkl")
    print (n)
    #df = pd.read_pickle("/home/ang/BLM_hashtags/aggre_data/aggre_tweet_hashtag_t2v.pkl")
    

