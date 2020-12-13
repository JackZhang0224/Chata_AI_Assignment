# %%
%run 0.utils.py

# %%
## define data path for dev and holdout data
train_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\data\data\sentiment_dataset_train.csv'
dev_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\data\data\sentiment_dataset_dev.csv'
holdout_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\data\data\sentiment_dataset_test.csv'

baseline_model_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\baseline_nlp_model.pkl'
dl_model_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\final\chata_cnn_weights.hdf5'

spell_check_path = r'C:\Users\jackzhang\OneDrive - Intelius AI\Desktop\personal\previous_work_project\chata\en.pkl\en.pkl'


max_features = 35000
maxlen = 2000


# %% 
## preprocess and evaluate on dev data

train_df0 = pd.read_csv(train_path)
## transform traning data 
train_df = preprocessing(train_df0,spell_check_path)

dev_df0 = pd.read_csv(dev_path)
dev_df = preprocessing(dev_df0,spell_check_path)

# %%
# load model for baseline and generate prediction for dev dataset
model_nlp = load(open(baseline_model_path, 'rb'))
dev_pred_nlp = model_nlp.predict(dev_df['review'])
print(classification_report(dev_df['rating'], dev_pred_nlp))


############################ following is prediction for dl model ###########

# %%

x_dev,dev_pred_dl,model_dl = dl_model_inference(train_df,dev_df)


# %%

class_names = ['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']
# Splitting off my y variable

dev_df1 = pd.get_dummies(dev_df, columns = ['rating'])

y_dev = dev_df1[class_names].values
# %%


loss, accuracy = model_dl.evaluate(x_dev, y_dev, verbose=False)
print("Dev data Accuracy:  {:.4f}".format(accuracy))



# %%
## combine prediction into dataframe and calulate confusion matrix

predict_class = np.argmax(dev_pred_dl, axis=1)
print(classification_report(dev_df['rating'].astype(int), predict_class+1))


dev_df_final = dev_df.copy()
dev_df_fianl['pred_NLP'] =dev_pred_nlp
dev_df_final['pred_dl'] = predict_class+1
