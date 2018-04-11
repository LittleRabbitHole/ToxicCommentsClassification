#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:21:25 2018

@author: angli
"""

import os
import logging
import argparse
import glob
import copy
import re
import string
import html
from nltk.corpus import stopwords
import nltk
#nltk.download('popular')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('get_CleanComments')


#define some patterns
URL_PATTERN=re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
SMILEYS_PATTERN = re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE)
NUMBERS_PATTERN = re.compile(r"(^|\s)(\-?\d+(?:\.\d)*|\d+)")


#input as training data   
#output the ([list of IDs], [list of cleaned comments])
#length should be same as input lines
def get_cleanlines(tweettext):
    #match out the tweetID following by ",": "12345678912345678",
    tweetID_raw = re.findall(r'\"[0-9]{16,19}\",', tweettext)
    #clean out the last ","
    tweetID_lst = [x[:-1].replace(" ","") for x in tweetID_raw]
    
    #tweet text
    tweetText_raw = re.split(r'\"[0-9]{16,19}\",', tweettext)[1::]
    #clean out
    tweetText_lst = [html.unescape(y.strip().replace('"','').replace('\n','')) for y in tweetText_raw]
    
    #check
    if len(tweetID_lst) == len(tweetText_lst):
        #return
        return(len(tweetID_lst), tweetID_lst, tweetText_lst)
    else:
        raise ValueError('Error processing files: lines not match')



#clean tweet text
def clean_tweet(Nlines, tweetID_lst, tweetText_lst):
    updated_cleantweet_lst = []
    for n in range(Nlines):
        if n%1000 ==0: print (n)
        tweetID =  tweetID_lst[n]
        #clean url, emoji, mention, smiley
        cleantweet_Url = re.sub(URL_PATTERN, '', tweetText_lst[n])
        cleantweet_UrlMention = re.sub(MENTION_PATTERN, '', cleantweet_Url)
        cleantweet_UrlMentionRev = re.sub(RESERVED_WORDS_PATTERN, '', cleantweet_UrlMention)
        cleantweet_UrlMentionRevEmoji = re.sub(EMOJIS_PATTERN, '', cleantweet_UrlMentionRev)
        cleantweet_final = re.sub(SMILEYS_PATTERN, '', cleantweet_UrlMentionRevEmoji)
        #lower case
        tweetText = str(cleantweet_final).replace("N/A","").lower()
        #into list
        cleantweet_lst = tweetText.split(' ')
        cleantweet_lst = list(filter(None, cleantweet_lst))
        #if not none
        if len(cleantweet_lst) >= 1 and cleantweet_lst[0] != '':
            #remove ending hashtags
            filtered_hashtags = remove_EndHashtags(cleantweet_lst)
            #final remove leading/trailing puctuation each word
            tweet_word_lst1 = [s.strip("`~()?:!.,;'""&*<=+ >#|-/{}%$^@[]") for s in filtered_hashtags]
            #remove non-letter in middle
            tweet_word_lst = [re.sub(r'[^a-zA-Z0-9]+', '', s) for s in tweet_word_lst1]
            #remove stop words
            filtered_words = [word for word in tweet_word_lst if word not in stopwords.words('english')]
            #remove all non-letters
            filtered_stopwords = [re.sub(r'^[^a-zA-Z]+$', '', s) for s in filtered_words]
            #remove none
            final_tweet_word_lst = list(filter(None, filtered_stopwords))
            #rejoin into sentence
            cleanedtweettext = ' '.join(final_tweet_word_lst)
            #add tweetID
            final_tweet_word_lst = [tweetID, cleanedtweettext]
            #update cleantweet list
            updated_cleantweet_lst.append(final_tweet_word_lst)
        else:
            #cleantweet_lst = [""]
            #rejoin into sentence
            cleanedtweettext = ' '.join(cleantweet_lst)
            final_tweet_word_lst = [tweetID, cleanedtweettext]
            updated_cleantweet_lst.append(final_tweet_word_lst)
    
    #return result  
    return (updated_cleantweet_lst)
