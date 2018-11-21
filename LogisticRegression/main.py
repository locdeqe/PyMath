import pandas
import numpy as np

data = pandas.read_csv('./data/data.csv', header = None, names = ['Target', 'Value 1', 'Value 2'])

print(data.head())