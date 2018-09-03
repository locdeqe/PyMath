import pandas
import numpy as np
import os

data = pandas.read_csv('./data/titanic.csv', index_col="PassengerId")

#passangers in total
totalCount = len(data.index)

#males in total
males = data[data['Sex'] == 'male']
totalMalesCount = len(males.index)

#females in total
females = data[data['Sex'] == 'female']
totalFemalesCount = len(females.index)

#survived
survived = data[data['Survived'] == 1]
survivedCount = len(survived.index)
surviedPercent = "%.2f" % round(((survivedCount / totalCount) * 100) , 2) 

#first class passangers
firstClassPassangers = data[data['Pclass'] == 1]
firstClassPassangersCount = len(firstClassPassangers.index)
firstClassPassangersPercent = "%.2f" % round(((firstClassPassangersCount / totalCount) * 100), 2)


#mean age and median age
meanAge = "%.2f" % (round(data['Age'].mean(), 2)) 
medianAge = data['Age'].median()

#Pearson correlation
correlationData = "%.2f" % round((data[['SibSp', 'Parch']].corr(method='pearson').iloc[0].iloc[1]) * 100)

#mostPopularFemaleName
femaleNames = females[['Name']]
results = {}

for row in femaleNames.iterrows():
    nameString = str(row[1])
    nameString = nameString[:nameString.find('Name:')]
    
    if nameString.find('Miss.') > -1:
        name = nameString[nameString.find('Miss.') + 5 :]
    else: 
        name = nameString[nameString.find('Mrs.') + 4 :]
    
    if name.find('(') > -1:
        name = name[name.find('(') + 1 : name.find(')')]

    if name[0] == "\"" :
        name = name[1:-1]

    resultArray = name.strip().split()

    if len(resultArray) > 1:
       resultArray = resultArray[:-1]
    
    for name in resultArray:
        if results.get(name) == None:
            results[name] = 1
        else:
            results[name] = results.get(name) + 1

mostPopularName = sorted(results.items(), key=lambda x: (-x[1], x[0]))[0][0]


if not os.path.exists('./results'):
    os.makedirs('./results')
    os.makedirs('./results/partOne')
    os.makedirs('./results/partTwo')
    os.makedirs('./results/partThree')
    os.makedirs('./results/partFour')
    os.makedirs('./results/partFive')
    os.makedirs('./results/partSix')

#males, females print
answer = open('./results/partOne/answer.txt', 'w', encoding='utf-8')
answer.write(str(totalMalesCount) + ' ' + str(totalFemalesCount))
answer.close()

#survivedPrint
answer = open('./results/partTwo/answer.txt', 'w', encoding='utf-8')
answer.write(surviedPercent)
answer.close()

#first class print
answer = open('./results/partThree/answer.txt', 'w', encoding='utf-8')
answer.write(firstClassPassangersPercent)
answer.close()

#mean and median age print
answer = open('./results/partFour/answer.txt', 'w', encoding='utf-8')
answer.write(str(meanAge) + ' ' + str(medianAge))
answer.close()

#Pearson correlation print
answer = open('./results/partFive/answer.txt', 'w', encoding='utf-8')
answer.write(correlationData)
answer.close()

#mostPopularFemaleName print
answer = open('./results/partSix/answer.txt', 'w', encoding='utf-8')
answer.write(mostPopularName)
answer.close()