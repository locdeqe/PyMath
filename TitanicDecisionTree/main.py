import pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def visualize_tree(tree, feature_names):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

def preprocess_data(data):
    selectedData = data[['Pclass', 'Fare', 'Sex', 'Age']]
    selectedData = selectedData[selectedData['Age'] > 0]
    selectedData['Sex'] = selectedData['Sex'].map({'female': 0, 'male': 1})

    return selectedData

data = pandas.read_csv('./data/titanic.csv', index_col="PassengerId")

preprocessed_data = preprocess_data(data)
feature_names = list(preprocessed_data.columns[:])

targetData = data[['Survived']]
targetData = targetData[data['Age'] > 0]

clf = DecisionTreeClassifier(random_state=241)

clf.fit(preprocessed_data, targetData)

visualize_tree(clf, feature_names)
