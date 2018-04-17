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


model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
import os
model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
model_path = '{}/w2vmodel'.format(model_dir_path)
model = Word2Vec.load('{}/cleantxt_200.w2v'.format(model_path))
print (model.most_similar('memory'))

#all words
vocab = model[model.vocab]

vocab = list(model.vocab)
X = model[vocab] #this is vector


vocab = vocab[0:3000]
#>>> len(X) 383628
x = X[0:3000]
x_tsne = tsne.fit_transform(x)

df = pd.DataFrame(x_tsne, index=vocab, columns=['x', 'y'])

