# %%

from utils import *

# %%
## define parameters

model_nlp_path = os.getcwd() + '/baseline_nlp_model.pkl'


# %%
## create a pipeline to train model with training data

from pickle import dump,load
pipeline = Pipeline([
    ('Tf-Idf', TfidfVectorizer(ngram_range=(1,2), analyzer=text_process)),
    ('classifier', MultinomialNB())
])
X = train_df['review']
y = train_df['rating']
review_train, review_test, label_train, label_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(review_train, label_train)

dump(pipeline, open(model_nlp_path, 'wb'))
# %%
## generate prediction on validation dataset and check model accuracy
baseline_model = load(open(model_nlp_path, 'rb'))
pip_pred = baseline_model.predict(review_test)
print(classification_report(label_test, pip_pred))
