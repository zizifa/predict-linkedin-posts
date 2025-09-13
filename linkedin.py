
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
""""
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
    
    """

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
y_scaled = scaler.fit_transform(y.values.reshape[-1 , 1])

x_train , x_test , y_train , y_test =train_test_split(x_vectorized ,y_scaled ,test_size =0.2)


model = Sequential([
    Dense(120 ,activation =relu),
    Dense(64 ,activation =relu),
    Dense(1)
])

model.compile(loss='mean_squared_error' , optimazer='adam' , metrics ='accuracy')
model.fit(x_train , y_train ,epochs =14 , batch_size=16)



new_post="""
طوری رفتار نکنید که همه خوششون بیاد !
قرار نیست همه خوششون بیاد ، اصلا اگه همه خوششون اومد باید به خودتون شک کنید .
خودتون‌ باشید .

"""

normalizenewtext = Normalizer()
tokenizenewpost =WordTokenizer()
nnp=normalizenewtext.normalize(new_post)
token =tokenizenewpost.tokenize(nnp)

    
cleantokennewpost=[t for t in token if tk.isalnum()]
cleannewpost = " ".join(cleantokennewpost)
print(cleannewpost)
    
xxx=model.predict(vector.transform([cleannewpost]).toarray())
scaled_predict=scaler.inverse_transform(xxx.reshape(-1,1))
scaled_predict = int(scaled_predict[0][0])
print(scaled_predict)

