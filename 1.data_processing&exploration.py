# %%
## import library for data processing & data exploration
import importlib
moduleName = 'utils'
importlib.import_module(moduleName)
# %%
from utils import *

# %%
## define data path and spell check pretrained data 
train_path = 'https://raw.githubusercontent.com/JackZhang0224/Chata_AI_Assignment/main/sentiment_dataset_train.csv'
spell_check_path_url = 'https://haptik-website-images.haptik.ai/spello_models/en.pkl.zip'

# %%
## read training data and exploring training data structure
train_df0 = pd.read_csv(train_path)
print(train_df0.dtypes)
print(train_df0.head(5))
print(train_df0.shape)

# %%
## transform traning data 
train_df = preprocessing(train_df0,spell_check_path_url)

