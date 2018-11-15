import pandas as ps
import numpy as np
from sklearn.svm import SVC
import os

def writeAnswer(path, answer):
    file = open(path, 'w', encoding='utf-8')
    file.write(str(answer))
    file.close()

data = ps.read_csv('./data/data.csv',  header = None, names = ['Target', 'Value 1', 'Value 2']);

model = SVC(C = 100000, random_state = 241)

model.fit(data[['Value 1', 'Value 2']], np.array(data[['Target']]).ravel())

answer = ' '.join(str(x + 1) for x in model.support_)

if not os.path.exists('./results'):
    os.makedirs('./results/partOne')

writeAnswer('./results/partOne/answer.txt', answer)