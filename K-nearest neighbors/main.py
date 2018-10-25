import pandas as ps
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
import os

def writeAnswer(path, answer):
    file = open(path, 'w', encoding='utf-8')
    file.write(str(answer))
    file.close()

wineData = ps.read_csv('./data/wine.csv')

wineData.columns = ['Class',
                    'Alcohol', 
                    'Malic acid', 
                    'Ash', 
                    'Alcalinity of ash', 
                    'Magnesium', 
                    'Total phenols', 
                    'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity',
                    'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']

wineDataResults =  np.array(wineData[['Class']]).ravel()
wineData = scale(wineData.drop(columns=['Class']))

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

result = {}

for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors = i)

    neigh.fit(wineData, wineDataResults)

    score = np.mean(cross_val_score(neigh, wineData, wineDataResults, cv=kf))  
      
    result.update({i: score})

partOne = max(result, key=result.get)
partTwo = "%.2f" % result.get(partOne)

print(partOne, partTwo)


if not os.path.exists('./results'):
    os.makedirs('./results')
    os.makedirs('./results/partOne')
    os.makedirs('./results/partTwo')

writeAnswer('./results/partOne/answer.txt', partOne)
writeAnswer('./results/partTwo/answer.txt', partTwo)