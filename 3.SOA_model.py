# %%
from utils import *
# %%
## transform target variables into matrix

train_df_dl = pd.get_dummies(train_df, columns = ['rating'])

class_names = ['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']

# Splitting off my y variable
y = train_df_dl[class_names].values

train_X = list(train_df_dl['review'].values)

# %%
## split training data into validation dataset
X_train, X_test, y_train, y_test = train_test_split(train_X, y, test_size=0.20, random_state=100)

# %%
## decide word lenth
print("Number of words: ")
print(len(np.unique(np.hstack(train_X))))

print("Review length: ")
result = [len(x) for x in train_X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
# plot review length
plt.boxplot(result)
plt.show()

# %%
## now we decide
maxlen = 2000
max_features = 35000
embedding_dim= 100


# %%
## vetorize and create sequence for training/validation data
tokenizer = Tokenizer(num_words= max_features)
tokenizer.fit_on_texts(train_X)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# %%

vocab_size = len(tokenizer.word_index) + 1

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# %%
## train cnn model 

model = Sequential()

model.add(Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(5, activation='sigmoid'))

earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5)
checkpoint = ModelCheckpoint(monitor = 'val_loss', save_best_only = True, filepath = 'chata_cnn_weights.hdf5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=128,
                    callbacks=[earlystop, checkpoint])


# %%
## model evaluation on train/val dataset

loss, accuracy = model.evaluate(X_train,y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Validation Accuracy:  {:.4f}".format(accuracy))

## training and validation dataset accurary is around 90%

# %%
## look at tain/validation accuracy and loss graph

plot_history(history)

## validiation loss increased at after epoc 4
