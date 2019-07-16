import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('D:\\Python\\Machine Learning\\Book 3\\Indians-Diabetes\\diabetes.csv')
#print('dataset shape {}'.format(data.shape))
#data.head()
#print(data.groupby('Outcome').size())

X = data.iloc[:, 0:8]
Y = data.iloc[:, 8]
print('shape of X {}; shape of Y {}'.format(X.shape, Y.shape))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

# Build 3 modules
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=2)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=2, weights='distance')))
models.append(('Radius Neighbors', RadiusNeighborsClassifier(n_neighbors=2, radius=500.0)))
"""
# Train 3 modules and score each of them
results = []
for name, model in models:
    model.fit(X_train, Y_train)
    results.append((name, model.score(X_test, Y_test)))
for i in range(len(results)):
    print('name: {}; score: {}'.format(results[i][0], results[i][1]))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Use Cross Validation to train and score three modules
results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X, Y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print('name: {}; cross val score: {}'.format(results[i][0], results[i][1].mean()))
"""
# Choose a module
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
train_score = knn.score(X_train, Y_train)
test_score = knn.score(X_test, Y_test)
print('train score: {}; test score: {}'.format(train_score, test_score))

# Draw Score diagram
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
#from common.utils import plot_learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

knn = KNeighborsClassifier(n_neighbors=2)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(10, 6), dpi=200)
plot_learning_curve(knn, 'Learning Curve for KNN Diabetes', X, Y, ylim=(0.00, 1.01), cv=cv)

from sklearn.feature_selection import SelectKBest
# Choose the best K features
selector = SelectKBest(k=3)
X_new = selector.fit_transform(X, Y)
print(X_new[Y==0])

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_new, Y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print('name: {}; cross val score: {}'.format(results[i][0], results[i][1].mean()))

# Draw Diagram
plt.figure(figsize=(10, 6))
plt.ylabel('BMI')
plt.xlabel('Glucose')
# Draw Y == 0 as sample
plt.scatter(X_new[Y==0][:, 0], X_new[Y==0][:, 1], c='r', s=20, marker='o')
# Draw Y == 1 as smaple
plt.scatter(X_new[Y==1][:, 0], X_new[Y==1][:, 1], c='g', s=20, marker='^')
plt.scatter(X_new[Y==1][:, 0], X_new[Y==1][:, 2], c='b', s=20, marker='^')