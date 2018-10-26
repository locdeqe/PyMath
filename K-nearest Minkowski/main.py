import pandas as ps
import numpy as np
from sklearn import *
import os

def writeAnswer(path, answer):
    file = open(path, 'w', encoding='utf-8')
    file.write(str(answer))
    file.close()

raw_data = datasets.load_boston()

data = ps.DataFrame(preprocessing.scale(raw_data.data))

data.columns = raw_data.feature_names

results_vector = raw_data.target

all_params = np.linspace(1, 10, 200)

kFold = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 42)

results = {}

for i in all_params:
    model = neighbors.KNeighborsRegressor(n_neighbors = 5, p = i, weights = 'distance')
    model.fit(data, results_vector)
    score = np.mean(model_selection.cross_val_score(model, data, results_vector, cv = kFold))
    results.update({i: score})

pIndex = max(results, key=results.get)
partTwo = "%.2f" % results.get(pIndex)
pIndex = "%.2f" % pIndex

if not os.path.exists('./results'):
    os.makedirs('./results')
    os.makedirs('./results/partOne')
    os.makedirs('./results/partTwo')

writeAnswer('./results/partOne/answer.txt', pIndex)