# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 02:51:30 2019

@author: ABID
"""

#importing necessary modules
import numpy as np
import nltk
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
import pandas as pd

#creating stemmer for stemming
ps = PorterStemmer()
categories = ["joy","fear", "anger", "sadness", "disgust", "shame", "guilt"]
stopwords = stopwords.words("english")

#text will contain the tweets1
text = []
#cat will contain the emotion category of tweets
cat =[]

#reading data from csv file

data1=pd.read_excel("E:/8th semester/thesis/final project and docs/labeled english dataset.xlsx")

for row in data1.status_text:
    text.append(row);
for row2 in data1.label:
    cat.append(row2);
#Preprocessing task
temp = []
for i in range(7665):
    temp=word_tokenize(text[i], language="english")
    text[i] = ""
    for j in range(len(temp)):
        if temp[j] not in stopwords:
            temp[j] = ps.stem(temp[j])
            text[i] += temp[j] + " "

#Splitting data into 80% and 20%
trainText = text[800:]
trainCat = cat[800:]
testText = text[:800]
testCat = cat[:800]

data = []
# Building the pipeline for sequential tasks
# This pipeline will first vectorize the text as Bigram
# then find the IF_IDF of the bigram terms


#using Support vector machine
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer()),
             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                alpha=1e-3, n_iter=10, random_state=42)),])
text_clf= text_clf.fit(trainText, trainCat)
predicted = text_clf.predict(testText)
print("\nSVM Accuracy: ",np.mean(predicted == testCat)*100,"%")

print("Classification summary for SVM:\n")
print(metrics.classification_report(testCat, predicted, target_names=categories))
print("Confusion Matrix for Classification using SVM:\n")
print(metrics.confusion_matrix(testCat, predicted))
