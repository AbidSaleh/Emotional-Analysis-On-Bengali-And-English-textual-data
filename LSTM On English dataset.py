# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 01:58:55 2019

@author: ABID
"""

#!/usr/bin/env python


import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

categories = ["joy","fear", "anger", "sadness", "disgust", "shame", "guilt"]


data1=pd.read_excel("E:/8th semester/thesis/final project and docs/labeled english dataset.xlsx")
    

num_class = len(np.unique(data1.label.values))
y = data1['target'].values

MAX_LENGTH = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data1.status_text.values)
post_seq = tokenizer.texts_to_sequences(data1.status_text.values)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.05)

vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)


inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)

x = LSTM(64)(embedding_layer)
x = Dense(32, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

filepath="weights.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=True, epochs=40, callbacks=[checkpointer])

df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')


model.load_weights('weights.hdf5')
predicted = model.predict(X_test)
#print("predicted  = ")
#print(predicted)
#matrix = metrics.confusion_matrix(X_test.argmax(axis=1), predicted.argmax(axis=1))
#print(matrix);
prediction = np.argmax(predicted, axis=1)


print("accuracy score= \n")
print(accuracy_score(y_test, prediction))
print("confusionmatrix\n")
print(confusion_matrix(y_test, prediction))

print("Classification summary for RNN(LSTM) Classifier:\n")
print(metrics.classification_report(y_test, prediction, target_names=categories))

####