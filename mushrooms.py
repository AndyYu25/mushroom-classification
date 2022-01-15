# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import print_function

import math

#from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.data import Dataset
from sklearn.preprocessing import LabelBinarizer
import logging

#Setting up Environment and randomizing data
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
mushrooms = pd.read_csv("mushrooms.csv", sep=",")
mushrooms = mushrooms.reindex(np.random.permutation(mushrooms.index))
# Converting values in dataset to numbers
mushrooms = mushrooms.drop(mushrooms[mushrooms['stalk-root']=='?'].index)
mushrooms = mushrooms.drop('veil-type', axis=1)
binarizer = preprocessing.LabelBinarizer()
for feature in mushrooms.columns:
    if len(mushrooms[feature].unique()) == 2:
        mushrooms[feature] = binarizer.fit_transform(mushrooms[feature])
featuresTemp = []
for feature in mushrooms.columns[1:]:
    if len(mushrooms[feature].unique()) > 2:
        featuresTemp.append(feature)
x = pd.get_dummies(mushrooms[featuresTemp])
mushrooms = mushrooms.join(x)
mushrooms = mushrooms.drop(featuresTemp, axis=1)
def process_target(mushrooms):
     target = pd.DataFrame()
     target["class"] = mushrooms["class"]
     return target
#Dividing into training and validation datasets
trainingExamples = mushrooms.head(6500)
trainingTargets = process_target(mushrooms.head(6500))
validationExamples = mushrooms.tail(1625)
validationTargets = process_target(mushrooms.tail(1625))

trainingExamples = np.asarray(trainingExamples).astype('float32')
trainingTargets = np.asarray(trainingTargets).astype('float32')
validationExamples = np.asarray(validationExamples).astype('float32')
validationTargets = np.asarray(validationTargets).astype('float32')


def construct_feature_columns(input_features):
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def trainModel(model, epochs, trainingExamples, trainingTargets):
  model.fit(trainingExamples, trainingTargets, epochs = epochs)
  trainPrediction= model.predict(trainingExamples)
  scores = model.evaluate(trainingExamples, trainingTargets, verbose=0)
  print('Final Accuracy on training data: {}% \n Final error on training data: {}'.format(scores[1] * 100, 1 - scores[1]))   


def evaluateModel(model, validationExamples, validationTargets):
  testPredictions = model.predict(validationExamples)
  scores = model.evaluate(validationExamples, validationTargets, verbose=0)
  print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores[1] * 100, 1 - scores[1]))   


model = Sequential()
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


trainModel(model, 50, trainingExamples, trainingTargets)
evaluateModel(model, validationExamples, validationTargets)