import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def read_dataset(fname):
    # First column is column header
    data = pd.read_csv(fname, index_col=0)
    # Drop useless column
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # Gender data
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    # Boarding harbour info
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n))
    #print(labels)
    # Process loss data
    data = data.fillna(0)
    return data

#print(train1[:10])
train = read_dataset('titanic//test.csv')

Xtest = train.values

from sklearn.externals import joblib
clf = joblib.load('titanic.pkl')

Ypred = clf.predict(Xtest)
Ypred = pd.Series(Ypred)
ID = pd.read_csv('titanic//test.csv')
ID = ID['PassengerId']
result = pd.concat([ID, Ypred], axis=1)
result.columns = ['PassengerId', 'Survived']
result.to_csv('titanic//result.csv', index=False)
