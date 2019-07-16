from sklearn.datasets import load_breast_cancer
import numpy as np
from matplotlib import pyplot as plt

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

"""
print('data shape: {0}; no. positive: {1}; no. negative: {2}'.format(X.shape, y[y==1].shape[0], y[y==0].shape[0]))
print(cancer.data[0])
print(cancer.feature_names)
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
"""
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print('train score: {train_score:.6f}; test score: {test_score:.6f}'.format(train_score=train_score, test_score=test_score))

# Sample predict
y_pred = model.predict(X_test)
print('matchs: {0}/{1}'.format(np.equal(y_pred, y_test).shape[0], y_test.shape[0]))

# Predict probability : Find the prob lower than 90%
y_pred_proba = model.predict_proba(X_test)
# Print first sample of data
print('sample of predict probability: {0}'.format(y_pred_proba[0]))

# Find first row, if negative prob is bigger than 0.1
y_pred_proba_0 = y_pred_proba[:, 0] > 0.1
result = y_pred_proba[y_pred_proba_0]

# Find second row, if positive prob is bigger than 0.1
y_pred_proba_1 = result[:, 1] > 0.1
print(result[y_pred_proba_1])
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Add degree for preprocessing
def polynomial_model(degree=1, **kwarg):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    logistic_regression = LogisticRegression(**kwarg)
    pipeline = Pipeline([('polybomial_features', polynomial_features), ('logistic_regression', logistic_regression)])
    return pipeline

import time

model = polynomial_model(degree=2, penalty='l1')

start = time.clock()
#model.fit(X_train, y_train)

#train_score = model.score(X_train, y_train)
#cv_score = model.score(X_test, y_test)
#print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.clock()-start, train_score, cv_score))

#logistic_regression = model.named_steps['logistic_regression']
#print('model parameters shape: {0}; count of non-zero element: {1}'.format(logistic_regression.coef_.shape, np.count_nonzero(logistic_regression.coef_)))

# Plot Learning curve
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.legend(loc='best')
    return plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
title = 'Learning Curves (degree={0}, penalty={1})'
degrees = [1, 2]

# L1 penalty
penalty = 'l1'

start = time.clock()
plt.figure(figsize=(12, 4))
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learning_curve(polynomial_model(degree=degrees[i], penalty=penalty), title.format(degrees[i], penalty), X, y, ylim=(0.8, 1.01), cv=cv)

print('elaspe: {0:.6f}'.format(time.clock()-start))

# L2 penalty
penalty = 'l2'

start = time.clock()
plt.figure(figsize=(12, 4))
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i + 1)
    plot_learning_curve(polynomial_model(degree=degrees[i], penalty=penalty, solver='lbfgs'), title.format(degrees[i], penalty), X, y, ylim=(0.8, 1.01), cv=cv)

print('elaspe: {0:.6f}'.format(time.clock()-start))