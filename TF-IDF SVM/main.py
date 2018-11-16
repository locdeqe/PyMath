import pandas as ps
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
import os

def writeAnswer(path, answer):
    file = open(path, 'w', encoding='utf-8')
    file.write(str(answer))
    file.close()

newGroups = datasets.fetch_20newsgroups(subset = 'all', categories = ['alt.atheism', 'sci.space'])
vectorizer  = TfidfVectorizer(stop_words = 'english')

X = vectorizer.fit_transform(newGroups.data)
y = newGroups.target
'''
grid = {'C': np.power(10.0, np.arange(-5, 5))}
cv = KFold(n_splits = 5, shuffle = True, random_state = 241)
model = SVC(kernel = 'linear', random_state = 241)
gs = GridSearchCV(model, grid, scoring = 'accuracy', cv = cv)

gs.fit(X, y)

coef = gs.best_params_
'''
newModel = SVC(C = 10, kernel = 'linear', random_state = 241)
newModel.fit(X, y)

afterFitCoefs = newModel.coef_.toarray()

for x in afterFitCoefs:
    x = abs(x)


'''
N = 10

idx = np.ravel(X.sum(axis=0).argsort(axis=1))[::-1][:N]
top10Words = np.array(vectorizer.get_feature_names())[idx].tolist()

answer = ' '.join(top10Words)

if not os.path.exists('./result'):
    os.makedirs('./result')

writeAnswer('./result/answer.txt', answer)
'''