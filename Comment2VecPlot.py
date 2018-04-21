#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:35:30 2018

@author: Ang
"""

import pickle
import os
from gensim.models import Word2Vec
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

model_dir_path = os.path.dirname(os.path.realpath("/home/ang/Comments/w2vmodel"))
model_path = '{}/w2vmodel'.format(model_dir_path)
model = Word2Vec.load('{}/cleantxt_200.w2v'.format(model_path))
#print (model.most_similar('revert'))
#w2v_model.vocab

train = pd.read_csv("train_cleaned.csv")
rowsums=train.iloc[:,1:6].sum(axis=1)
train['clean']=(rowsums==0)

extr_toxic = train[train['severe_toxic'] == 1]
extr_toxic_ids = extr_toxic['id'].tolist()

extr_toxic_ids = extr_toxic_ids[0:1000]

clean = train[train['clean'] == 1]
clean_ids = clean['id'].tolist()
clean_ids = clean_ids[0:1000]

train_c2v = pickle.load( open( "train_c2v.pkl", "rb" ) )

clean_vec = []
for idx in clean_ids:
    clean_vec.append(train_c2v[idx])

toxic_vec = []
for idx in extr_toxic_ids:
    toxic_vec.append(train_c2v[idx])

vec = clean_vec + toxic_vec
pickle.dump( vec, open( "vec200_plot.p", "wb" ) )

#plot the 2000 vec
with open('/Users/angli/Ang/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/data/vec200_plot.p', 'rb') as f:
        vec = pickle.load(f, encoding='bytes')
    
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
tsne_model = TSNE(n_components=2)
new_values = tsne_model.fit_transform(vec)   
new_values_df = pd.DataFrame(new_values) 
label = [1]*1000 + [2]*1000
new_values_df['label'] = label


fig = plt.scatter(new_values_df[0], new_values_df[1], alpha=0.2,
            s=50, c=new_values_df.label, cmap='viridis')

sns.pairplot(x_vars=[0], y_vars=[1], data=new_values_df, hue="label", size=5)




