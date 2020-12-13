# %%
## import library for data processing & data exploration
import numpy as np
import pandas as pd
import re
from spello.model import SpellCorrectionModel  

from numpy import array
from numpy import asarray
from numpy import zeros

# %%
## define data path and spell check pretrained data 

url = 'https://raw.githubusercontent.com/hxchua/datadoubleconfirm/master/datasets/arrivals2018.csv'



df = pd.read_csv(url, error_bad_lines=False)
train_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\data\data\sentiment_dataset_train.csv'
spell_check_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\en.pkl\en.pkl'
# %%
## read training data and exploring training data structure
train_df0 = pd.read_csv(train_path)
print(train_df0.dtypes)
print(train_df0.head(5))
print(train_df0.shape)

# %%
## transform traning data 
train_df = preprocessing(train_df0,spell_check_path)