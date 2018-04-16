#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:08:04 2016

@author: angli
"""


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.misc import imread

if __name__ == "__main__":


    train_cleaned = pd.read_csv("/Users/angli/ANG/OneDrive/Documents/Pitt_PhD/Class/2018Spring/ML/finalProject/train_cleaned.csv")
    train_cleaned.columns.values
    rowsums=train_cleaned.iloc[:,2:].sum(axis=1)
    train_cleaned['clean']=(rowsums==0)
    train_cleaned['clean'].sum()
    
    clean  = list(train_cleaned.query('clean == 1 ')['clean_comment'].dropna())
    clean = [x.strip() for x in clean]
    clean_txt = " ".join(clean)
    
    toxic  = list(train_cleaned.query('toxic == 1 ')['clean_comment'])
    toxic = [x.strip() for x in toxic]
    toxic_txt = " ".join(toxic)
    
    severe = list(train_cleaned.query('severe_toxic == 1 ')['clean_comment'])
    severe = [x.strip() for x in severe]
    severe_txt = " ".join(severe)
    
    obscene  = list(train_cleaned.query('obscene == 1 ')['clean_comment'])
    obscene = [x.strip() for x in obscene]
    obscene_txt = " ".join(obscene)
    
    threat  = list(train_cleaned.query('threat == 1 ')['clean_comment'])
    threat = [x.strip() for x in threat]
    threat_txt = " ".join(threat)
    
    insult  = list(train_cleaned.query('insult == 1 ')['clean_comment'])
    insult = [x.strip() for x in insult]
    insult_txt = " ".join(insult)
    
    identity_hate  = list(train_cleaned.query('identity_hate == 1 ')['clean_comment'])
    identity_hate = [x.strip() for x in identity_hate]
    identity_hate_txt = " ".join(identity_hate)
    
    #generate the wordcloud
    twitter_mask = imread('/Users/angli/Documents/GitHub/ToxicCommentsClassification/word_cloud_pngs/twitter_mask.png', flatten=True)
    wordcloud = WordCloud(
                          font_path='/Users/angli/Documents/GitHub/ToxicCommentsClassification/cabin-sketch-v1.02/CabinSketch-Bold.ttf',
                          background_color='white',
                          width=1800,
                          height=1400,
                          mask=twitter_mask
                ).generate(clean_txt)
    
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('/Users/angli/Documents/GitHub/ToxicCommentsClassification/word_cloud_pngs/clean_comments_wordcloud.png', dpi=300)
    plt.show()

