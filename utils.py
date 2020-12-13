# %%
## import library for this project
import numpy as np
import pandas as pd
import re
import os, sys
from spello.model import SpellCorrectionModel  

import importlib
import requests,zipfile,io
import urllib.request

from spello.model import SpellCorrectionModel


from numpy import array
from numpy import asarray
from numpy import zeros

import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense,LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D,Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, GRU
from tensorflow.keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from tensorflow.keras.models import load_model
from pickle import dump,load

################################################
### SECTION1: preprocessing step for all dataset 
#################################################
# %%

## define a function to clean up the data 
## 1. drop empty row and remove duplicates
## 2. remove ratings which are string

def clean_up(df,target,cols):
    new_df = df[~pd.isnull(df[target])]
    new_df.drop_duplicates(subset=cols, inplace=True)
    new_df.reset_index(drop= True,inplace=True)
    if 'rating' in new_df.columns:
        if new_df.rating.dtype == 'object':
            new_df = new_df[new_df.rating.str.len()<=1]
        new_df['rating'] = new_df['rating'].astype('str')
    return new_df

# %%
## define a function to clean review column 
## 1. remove any html tags
## 2. remove punctuations and numbers
## 3. remove single character 
## 4. remove multiple space
## 5. convert review to low case

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # convert to low case
    sentence = sentence.lower()

    return sentence

# %%

## create a preprocessing function to transform input data

def preprocessing(df,spell_check_path_url):
    import os
    if 'rating' in df.columns:
        df_1 = clean_up(df,'review',['review','rating'])
    else:
        df_1 = clean_up(df,'review',['review'])
    
    df_1['review'] = df_1['review'].apply(lambda x: preprocess_text(x))
    ## now load a pretrained model to correct spelling mistakes

    ## geting spell check zip file into your local
    request = requests.get(spell_check_path_url)
    file = zipfile.ZipFile(io.BytesIO(request.content))
    file.extractall(os.getcwd())

    sp = SpellCorrectionModel(language='en')  
    sp.load(os.getcwd()+'/en.pkl')
    df_1['review'] = df_1['review'].apply(lambda x: sp.spell_correct(x)['spell_corrected_text'])

    return df_1


# %%
#########################################
##### SECTION 2: FOR NLP MODEL
#########################################

## nltk text processing to filter any non-relevant words
def text_process(df):
    nopunc = [i.lower() for i in df if i not in string.punctuation]
    nopunc_text = ''.join(nopunc)
    return [i for i in nopunc_text.split() if i not in stopwords.words('english')]


# %%
#########################################
##### SECTION 3: FOR DL MODEL
#######################################

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# %%
## vetorize and create sequence for dev/test data


def dl_model_inference(train_df,test_df,max_features,maxlen,dl_model_path):
    tokenizer = Tokenizer(num_words= max_features)
    tokenizer.fit_on_texts(list(train_df['review'].values))
    vocab_size = len(tokenizer.word_index) + 1
    X_vector = tokenizer.texts_to_sequences(test_df['review'].values)
    X_test = pad_sequences(X_vector, padding='post', maxlen=maxlen)
    
    request = requests.get(dl_model_path)
    file = zipfile.ZipFile(io.BytesIO(request.content))
    file.extractall(os.getcwd())

    model_dl = load_model(os.getcwd() + '/chata_cnn_model.h5')
    dl_model_pred = model_dl.predict(X_test)
    return X_test,dl_model_pred,model_dl

