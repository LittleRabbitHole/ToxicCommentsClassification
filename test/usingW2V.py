#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 23:22:39 2018

@author: Ang
"""
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
model_path = '{}/w2vmodel'.format(model_dir_path)
model = Word2Vec.load('{}/cleantxt_200.w2v'.format(model_path))
print (model.most_similar('memory'))



def tsne_plot(vocabs_df):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word, row in vocabs_df:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    


df = pd.read_csv("/Users/Ang/Documents/GitHub/ToxicCommentsClassification/x_tsne.csv", index_col=0)


#all words
vocab = model[model.vocab]

vocab = list(model.vocab)
X = model[vocab] #this is vector

vocab = vocab[0:3000]
#>>> len(X) 383628
x = X[0:3000]

x_tsne = tsne.fit_transform(x)

df = pd.DataFrame(x_tsne, index=vocab, columns=['x', 'y'])
df = df[0:200]

import matplotlib.pyplot as plt
df = pd.read_csv("/Users/Ang/Documents/GitHub/ToxicCommentsClassification/x_tsne.csv", index_col=0)
df = df[100:200]




fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)
    
plt.show()
fig.savefig('x_tsne.png')