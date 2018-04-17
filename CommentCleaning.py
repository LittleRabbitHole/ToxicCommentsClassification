#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:21:25 2018

@author: angli
"""
#import pandas as pd
#import os
import logging
#import argparse
import re
from nltk.corpus import stopwords
import nltk
#nltk.download('popular')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import HTMLParser
#import html
#import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('get_CleanComments')


#define some patterns
#URL_PATTERN=re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
URL_PATTERN=re.compile(r'''(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))''')
#SMILEYS_PATTERN = re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE)
NUMBERS_PATTERN = re.compile(r"(^|\s)(\-?\d+(?:\.\d)*|\d+)")
IP_PATTERN = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

wordnet_lemmatizer = WordNetLemmatizer()

#mapping nltk lemmatizer pos taggs
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


#clean comment text
def cleanComment(comment):
    #clean url, emoji, mention, smiley
    clean_Url = re.sub(URL_PATTERN, '', comment)
    clean_UrlSmile = clean_Url.replace(":-)" , " ").replace(":)"," ").replace(";-)", " ").replace(";)"," ") #':-)', ':)', ';-)', ';)'
    clean_UrlSmileIP = re.sub(IP_PATTERN, '', clean_UrlSmile)
    clean_final = re.sub(NUMBERS_PATTERN, '', clean_UrlSmileIP)
    #lower case
    commentText = clean_final.replace("\n"," ").lower()
    #into list
    comment_lst = commentText.split(" ")
    comment_lst = list(filter(None, comment_lst))
    #if not none
    if len(comment_lst) >= 1 and comment_lst[0] != '':
        #comment_lst = [HTMLParser.HTMLParser().unescape(y.strip().replace('"','').replace('\n','')) for y in comment_lst]
        comment_lst = [html.unescape(y.strip().replace('"','').replace('\n','')) for y in comment_lst]
        #final remove leading/trailing puctuation each word
        comment_lst1 = [s.strip("`~()?:!.,;'""&*<=+ >#|-/{}%$^@[]") for s in comment_lst]
        #remove non-letter in middle
        comment_lst = [re.sub(r'[^a-zA-Z0-9]+', '', s) for s in comment_lst1]
        #remove stop words
        filtered_words = [word for word in comment_lst if word not in stopwords.words('english')]
        #remove none
        filtered_words = list(filter(None, filtered_words))  
        #lemm
        tagged = nltk.pos_tag(filtered_words)
        lemed_filtered_words = [wordnet_lemmatizer.lemmatize(x[0], get_wordnet_pos(x[1])) for x in tagged] #journalists
        #remove all non-letters
        filtered_stopwords = [re.sub(r'^[^a-zA-Z]+$', '', s) for s in lemed_filtered_words]
        #remove none
        final_word_lst = list(filter(None, filtered_stopwords))  
        #rejoin into sentence
        cleanedtext = ' '.join(final_word_lst)
    else:
        #cleantweet_lst = [""]
        #rejoin into sentence
        cleanedtext = ' '.join(comment_lst)
    
    #return result  
    return (cleanedtext)



if __name__ == "__main__":
    #train = pd.read_csv("/Users/Ang/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/train.csv")
    #all_comments_lst = list(train["comment_text"])
    #with open('/Users/angli/ANG/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/all_toxic_comments_lst.pkl', 'wb') as f:
        #pickle.dump(all_comments_lst, f, protocol=2)

    # read file    
    #comments_lst = pickle.load( open( "/home/ang/Comments/all_toxic_comments_lst.pkl", "rb" ) )
    train = pd.read_csv("/home/ang/Comments/test.csv")
    _log.info("Reading file ...")
    
    clean_comments = []
    n=0
    for ind, row in train.iterrows():
        n+=1
        if n % 2000 == 0: 
            _log.info("process line {}...".format(n))
            
        clean_comment = cleanComment(row["comment_text"])
        clean_comments.append(clean_comment)
    
    train["clean_comment"] = clean_comments
    
    _log.info("writing file ...")    
    #with open('/home/ang/Comments/clean_comments.pkl', 'wb') as f:
        #pickle.dump(clean_comments, f)
    train.to_csv("/home/ang/Comments/test_cleaned.csv", index=False)
    
    _log.info("done")    

