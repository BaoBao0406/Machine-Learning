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

train = read_dataset('titanic//train.csv')

from sklearn.model_selection import train_test_split

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='entropy')
clf.fit(X_train, y_train)
tr_score = clf.score(X_train, y_train)
cv_score = clf.score(X_test, y_test)

print('tr param: {0}; cv score: {1}'.format(tr_score, cv_score))
