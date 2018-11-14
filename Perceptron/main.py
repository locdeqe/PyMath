import pandas as ps
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

def writeAnswer(path, answer):
    file = open(path, 'w', encoding='utf-8')
    file.write(str(answer))
    file.close()

columns = ['Target', 'Value 1', 'Value 2']

trainData = ps.read_csv('./data/train.csv', header = None)
trainData.columns = columns

testData = ps.read_csv('./data/test.csv', header = None)
testData.columns = columns

model = Perceptron(random_state = 241)
model.fit(trainData[['Value 1', 'Value 2']], np.array(trainData[['Target']]).ravel())
accuracy = accuracy_score(np.array(testData[['Target']]).ravel(), model.predict(testData[['Value 1', 'Value 2']]))

scaler = StandardScaler()
scaler.fit(trainData[['Value 1', 'Value 2']])

prepocessedTrainData = scaler.transform(trainData[['Value 1', 'Value 2']])
prepocessedTestData = scaler.transform(testData[['Value 1', 'Value 2']])

model.fit(prepocessedTrainData, np.array(trainData[['Target']]).ravel())
postProcessedAccuracy = accuracy_score(np.array(testData[['Target']]).ravel(), model.predict(prepocessedTestData))

tunningResult = postProcessedAccuracy - accuracy

tunningResult =  "%.3f" % tunningResult

if not os.path.exists('./results'):
    os.makedirs('./results')
    os.makedirs('./results/partOne')

writeAnswer('./results/partOne/result.txt', tunningResult)