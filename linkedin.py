
import pandas as pd
import numpy as np
from hazm import *
from keras.layers import Dense
from keras.models import Sequential 
from keras.activations import relu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import MinMaxScaler
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
    
    

#df =pd.DataFrame(data)
#df.to_csv('linkedin.csv',index=False)

df=pd.read_csv('linkedin.csv')
#print(len(df))

print(df.columes)
x_raw=df['post']
vector = TfidfVectorizer()
x_vectorized =vector.fit_transform(x_raw)
y = df['reaction']


scaler =MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape[-1 , 1])

x_train , x_test , y_train , y_test =train_test_split(x_vectorized ,y_scaled ,test_size =0.2)