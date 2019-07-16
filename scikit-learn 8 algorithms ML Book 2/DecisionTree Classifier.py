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

#train1 = pd.read_csv('titanic//train.csv', index_col=0)
#print(train1[:10])
train = read_dataset('titanic//train.csv')
#print(train[:10])

from sklearn.model_selection import train_test_split

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('train dataset: {0}; test dataset: {1}'.format(X_train.shape, X_test.shape))

from sklearn.tree import DecisionTreeClassifier

#clf = DecisionTreeClassifier()
#clf.fit(X_train, y_train)
#train_score = clf.score(X_train, y_train)
#test_score = clf.score(X_test, y_test)
#print('train score: {0}; test score: {1}'.format(train_score, test_score))
"""
# Choose variable max_depth
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)

depths = range(2, 15)
scores = [cv_score(d) for d in depths]
#print(scores)
tr_scores = [s[0] for s in scores]
#print(tr_scores)
cv_scores = [s[1] for s in scores]

# Find cross data validation score
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))

plt.xlabel('max depth of decision tree')
plt.ylabel('score')
plt.plot(depths, cv_scores, '.g-', label='cross-validation score')
plt.plot(depths, tr_scores, '.r--', label='training score')
plt.legend()
"""
# Train model with score
def cv_score(val):
    clf = DecisionTreeClassifier(criterion='entropy', min_impurity_split=val)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)

# Set variable range, and use model to train
values = np.linspace(0.0, 1.0, 50)
scores = [cv_score(v) for v in values]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# Find cross data validation score
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = values[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))

# Draw model variable and model score
plt.figure(figsize=(6, 4))
plt.grid()
plt.xlabel('threshold of entropy')
plt.ylabel('score')
plt.plot(values, cv_scores, '.g-', label='cross-validation score')
plt.plot(values, tr_scores, '.r--', label='training score')
plt.legend()

from sklearn.model_selection import GridSearchCV

def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(6, 4))
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, '.--', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, '.-', color='g', label='Cross-validation score')
    plt.legend(loc='best')

thresholds = np.linspace(0, 0.5, 50)
# Set variable matrix
param_grid = {'min_impurity_split': thresholds}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X, y)
print('best param: {0}\nbest score: {1}'.format(clf.best_params_, clf.best_score_))
plot_curve(thresholds, clf.cv_results_, xlabel='gini thresholds')

entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.5, 50)

# Set variable matrix
param_grid = [{'criterion': ['entropy'], 'min_impurity_split': entropy_thresholds}, 
               {'criterion': ['gini'], 'min_impurity_split': gini_thresholds},
               {'max_depth': range(2, 10)},
               {'min_samples_split': range(2, 30, 2)}]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X, y)
print('best param: {0}\nbest score: {1}'.format(clf.best_params_, clf.best_score_))
