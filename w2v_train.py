#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:16:44 2018

@author: angli
"""
import pickle
import os
import logging
import sys
from argparse import ArgumentParser
from gensim.models import Word2Vec

def loadt(log):
    text = []
    
    i = 0
    #flist = glob.glob('{}*'.format(path))
    #for f_name in flist:
    print('\nprocessing file')
    comments = pickle.load( open( "/home/ang/Comments/train_clean_comments.pkl", "rb" ) )
    for line in comments:
        i += 1
        if type(line) != float:
            t = line.split(" ")           
            text.append(t)

        sys.stdout.write('\r')
        sys.stdout.write("text loaded: {}".format(i))
        sys.stdout.flush()
    
    log.info('{} tweets loaded in total'.format(i))
    
    return text


if __name__ == "__main__":
    #dir_path = os.path.dirname(os.path.realpath(__file__))


    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    #parser = ArgumentParser(description='input file dir')
    #parser.add_argument('--loc', '-l', help='file location', required=True)
    #args = parser.parse_args()
    #location = str(args.loc)

    # read text into memory
    log.info('source loading...')

    text = loadt(log)

    # train word2vec

    log.info('start training w2v')
    model = Word2Vec(min_count=0, window=10, size=200, sg=1, workers=30)

    log.info('building vocab')
    model.build_vocab(text)

    log.info('training')
    model.train(text)

    log.info('saving trained model')
    model_path = '/home/ang/Comments/w2vmodel'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save('{}/cleantxt_200.w2v'.format(model_path))
