# %%

from utils import *

# %%
## define parameters
train_path = 'https://raw.githubusercontent.com/JackZhang0224/Chata_AI_Assignment/main/sentiment_dataset_train.csv'
holdout_path = 'https://raw.githubusercontent.com/JackZhang0224/Chata_AI_Assignment/main/sentiment_dataset_test.csv'

dl_model_path = 'https://raw.githubusercontent.com/JackZhang0224/Chata_AI_Assignment/main/chata_cnn_model.zip'

spell_check_path_url = 'https://haptik-website-images.haptik.ai/spello_models/en.pkl.zip'

max_features = 35000
maxlen = 2000

# %% 
## preprocess and evaluate on dev data

train_df0 = pd.read_csv(train_path)
## transform traning data 
train_df = preprocessing(train_df0,spell_check_path_url)

# %%
holdout_df0 = pd.read_csv(holdout_path)
holdout_df = preprocessing(holdout_df0,spell_check_path_url)


############################ following is prediction for dl model ###########

# %%

x_holdout,holdout_pred_dl,model_dl = dl_model_inference(train_df,holdout_df,max_features,maxlen,dl_model_path)


# %%
## combine prediction into dataframe

predict_class = np.argmax(holdout_pred_dl, axis=1)
holdout_df['pred_dl'] = predict_class+1

