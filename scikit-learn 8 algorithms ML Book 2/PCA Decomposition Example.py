import time
import logging
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from matplotlib import pyplot as plt
import sys

"""
# Download photo
logging.basicConfig(level=logging.INFO, format='%(asctime)s % (message)s')

data_home='datasets/'
logging.info('Start to load dataset')
faces = fetch_olivetti_faces(data_home=data_home)
logging.info('Done with load dataset')
"""

data_home='datasets/'
faces = fetch_olivetti_faces(data_home=data_home)
X = faces.data
y = faces.target
targets = np.unique(faces.target)
target_names = np.array(["c%d" % t for t in targets])
n_targets = target_names.shape[0]
n_samples, h, w = faces.images.shape
print('Sample count: {}\nTarget count: {}'.format(n_samples, n_targets))
print('Image size: {}x{}\nDataset shape: {}\n'.format(w, h, X.shape))

def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
    # Display photo array
    plt.figure(figsize=(2* n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.01)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.axis('off')

n_row = 2
n_col = 6

sample_images = None
sample_titles = []
for i in range(n_targets):
    people_images = X[y==i]
    people_sample_index = np.random.randint(0, people_images.shape[0], 1)
    people_sample_image = people_images[people_sample_index, :]
    if sample_images is not None:
        sample_images = np.concatenate((sample_images, people_sample_image), axis=0)
    else:
        sample_images = people_sample_image
    sample_titles.append(target_names[i])
    
plot_gallery(sample_images, sample_titles, h, w, n_row, n_col)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
"""
# Use SVM to train model
from sklearn.svm import SVC

start = time.clock()
print('Fitting train datasets.. ')
clf = SVC(class_weight='balanced')
clf.fit(X_train, y_train)
print('Done in {0:.2f}s'.format(time.clock()-start))

start = time.clock()
print('Predicting test dataset...')
y_pred = clf.predict(X_test)
print('Done in {0:.2f}s'.format(time.clock()-start))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print('confusion matrix: \n')
np.set_printoptions(threshold=sys.maxsize)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=target_names))
"""
# Use diagram to plot the Explained Variance ratio (Best k value)
from sklearn.decomposition import PCA
print('Exploring explained variance ratio for dataset...')
candidate_components = range(10, 300, 30)
explained_ratios = []
start = time.clock()
for c in candidate_components:
    pca = PCA(n_components=c)
    X_pca = pca.fit_transform(X)
    explained_ratios.append(np.sum(pca.explained_variance_ratio_))
print('Done in {0:.2f}s'.format(time.clock()-start))

plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(candidate_components, explained_ratios)
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained variance ratio for PCA')
plt.yticks(np.arange(0.5, 1.05, .05))
plt.xticks(np.arange(0, 300, 20))


n_row = 1
n_col = 5

sample_images = sample_images[0:5]
sample_titles = sample_titles[0:5]

def title_prefix(prefix, title):
    return "{}: {}".format(prefix, title)

plotting_images = sample_images
plotting_titles = [title_prefix('orig', t) for t in sample_titles]
candidate_components = [140, 75, 37, 19, 8]
for c in candidate_components:
    print('Fitting and projecting on PCA(n_components={})... '.format(c))
    start = time.clock()
    pca = PCA(n_components=c)
    pca.fit(X)
    X_sample_pca = pca.transform(sample_images)
    X_sample_inv = pca.inverse_transform(X_sample_pca)
    plotting_images = np.concatenate((plotting_images, X_sample_inv), axis=0)
    sample_title_pca = [title_prefix('{}'.format(c), t) for t in sample_titles]
    plotting_titles = np.concatenate((plotting_titles, sample_title_pca), axis=0)
    print('Done in {0:.2f}s'.format(time.clock() - start))
    
print('Plotting sample image with different number of PCA conponments... ')
plot_gallery(plotting_images, plotting_titles, h, w, n_row * (len(candidate_components) + 1), n_col)

# Select PCA best k to decomposit data
n_components =140

print('Fitting PCA by using training data...')
start = time.clock()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print('Done in {0:.2f}s'.format(time.clock() - start))

print('Projecting input data for PCA...')
start = time.clock()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('Done in {0:.2f}s'.format(time.clock() - start))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

print('Searching the best parameters for SVC...')
param_grid = {'C': [1, 5, 10, 50, 100], 
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=2, n_jobs=4)
clf = clf.fit(X_train_pca, y_train)
print('Best parameters found by grid search:')
print(clf.best_params_)

# Use Confusion Matrix to display
from sklearn.metrics import confusion_matrix

start = time.clock()
print('Predict test dataset...')
y_pred = clf.best_estimator_.predict(X_test_pca)
cm = confusion_matrix(y_test, y_pred, labels=range(n_targets))
print('Done in {0:.2f}.\n'.format(time.clock()-start))
print('confusion matrix:')
np.set_printoptions(threshold=sys.maxsize)
print(cm)

from sklearn.metrics import classification_report
target_names = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13',
                'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25',
                'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36']
print(classification_report(y_test, y_pred, target_names=target_names))
