import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('./data/titanic.csv', index_col="PassengerId")
selectedData = data[['Pclass', 'Fare', 'Sex', 'Age']]
selectedData = selectedData[selectedData['Age'] > 0]

selectedData['Sex'] = selectedData['Sex'].map({'female': 1, 'male': 0})

targetData = data[['Survived']]
targetData = targetData[data['Age'] > 0]

clf = DecisionTreeClassifier(random_state=241)

clf.fit(selectedData, targetData)

print(clf.feature_importances_)