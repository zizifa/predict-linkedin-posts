import pandas as pd
import numpy as np
from hazm import *
from keras.layers import Dense
from keras.models import Sequential 
from keras.activations import relu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import json



with open('',encoding='utf-8') as f:
    post = json.load(f)

