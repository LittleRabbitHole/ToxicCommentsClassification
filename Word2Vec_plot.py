#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 20:53:05 2018

@author: Ang
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
model_path = '{}/w2vmodel'.format(model_dir_path)
model = Word2Vec.load('{}/cleantxt_200.w2v'.format(model_path))


def tsne_plot(vocabs_df):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    fig = plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #plt.show()
    fig.savefig('/home/ang/Comments/x_tsne.png')
