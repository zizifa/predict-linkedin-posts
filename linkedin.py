
import pandas as pd
import numpy as np
from hazm import *
from keras.layers import Dense
from keras.models import Sequential 
from keras.activations import relu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import json
import sys

data=[]
with open('',encoding='utf-8') as f:
    post = json.load(f)

for p in post :
    temporarypost = p['post']
    normal = Normalizer()
    tokenize =WordTokenizer()
    
    
    normaliztext=normal.normalize(temporarypost)
    token =tokenize.tokenize(normaliztext)
    
    
    cleantoken=[tk for tk in token if tk.isalnum()]
    cleanpost = " ".join(cleantoken)
    print(cleanpost)
    
    data.append({'post': cleanpost , 'reaction' : p['reactions'], 'comments' : p['comments']})
    
    

