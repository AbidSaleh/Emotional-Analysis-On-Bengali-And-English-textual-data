# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 03:15:32 2019

@author: ABID
"""

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



    


####

ps = PorterStemmer()
categories = ["joy","fear", "anger", "sadness", "disgust", "shame", "guilt"]
stopwords = stopwords.words("english")

#text will contain the tweets1
text = []
#cat will contain the emotion category of tweets
cat =[]
data=pd.read_excel("E:/8th semester/thesis/final project and docs/labeled english dataset.xlsx")

for row in data.status_text:
    text.append(row);
for row2 in data.label:
    cat.append(row2);
#Preprocessing task
temp = []
for i in range(7660):
    temp=word_tokenize(text[i], language="english")
    text[i] = ""
    for j in range(len(temp)):
        if temp[j] not in stopwords:
            temp[j] = ps.stem(temp[j])
            text[i] += temp[j] + " "
data1 = pd.DataFrame(
    {'status_text': text,
     'label': cat
    })
data1['target'] = data1.label.astype('category').cat.codes
####


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
          shuffle=True, epochs=30, callbacks=[checkpointer])

df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')


model.load_weights('weights.hdf5')
predicted = model.predict(X_test)

prediction = np.argmax(predicted, axis=1)


print("accuracy score= \n")
print(accuracy_score(y_test, prediction))
print("confusionmatrix\n")
print(confusion_matrix(y_test, prediction))

print("Classification summary for RNN(LSTM):\n")
print(metrics.classification_report(y_test, prediction, target_names=categories))

####