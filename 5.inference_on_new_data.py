# %%
## define parameters
train_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\data\data\sentiment_dataset_train.csv'
holdout_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\data\data\sentiment_dataset_test.csv'

dl_model_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\final\chata_cnn_weights.hdf5'

spell_check_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\en.pkl\en.pkl'


max_features = 35000
maxlen = 2000

# %%
%run 0.utils.py

# %% 
## preprocess and evaluate on dev data

train_df0 = pd.read_csv(train_path)
## transform traning data 
train_df = preprocessing(train_df0,spell_check_path)

holdout_df0 = pd.read_csv(holdout_path)
holdout_df = preprocessing(holdout_df0,spell_check_path)


############################ following is prediction for dl model ###########

# %%

x_holdout,holdout_pred_dl,model_dl = dl_model_inference(train_df,holdout_df,max_features,maxlen,dl_model_path)


# %%
## combine prediction into dataframe

predict_class = np.argmax(holdout_pred_dl, axis=1)
holdout_df['pred_dl'] = predict_class+1

