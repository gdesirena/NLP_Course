#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:13:42 2021

@author: gaddiel
"""

import nltk                                # Plibrería para NLP
from nltk.corpus import twitter_samples    # Ejemplo de conjunto de datos de Twitter de NLTK
import matplotlib.pyplot as plt            # biblioteca para visualización
import numpy as np  
from utilss import Utilities as prep

# select the lists of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = all_positive_tweets + all_negative_tweets

# let's see how many tweets we have
print("Number of tweets: ", len(tweets))
 #make a numpy array representing labels of the tweets
labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

# create a object of the class
preproces = prep()

# create frequency dictionary

freqs = preproces.build_freqs(tweets, labels)

# check data type
print(f'type(freqs) = {type(freqs)}')

# check length of the dictionary
print(f'len(freqs) = {len(freqs)}')
