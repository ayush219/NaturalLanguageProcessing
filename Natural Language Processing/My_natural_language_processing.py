# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:08:32 2019

@author: Ayush
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset
dataset= pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#cleaning
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(0,1000):    
    review=re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()

Y=dataset.iloc[:,1].values

#Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

#Classifier
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, Y_train) 

#Prediction
Y_pred= classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)