from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np

A = np.array([[3, 2000], [2, 3000], [4, 5000], [5, 8000], [1, 2000]], dtype='float')

def std_PCA(**argv):
    scaler = MinMaxScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
    return pipeline

pca = std_PCA(n_components=1)
R2 = pca.fit_transform(A)
print(R2)
# Reverse PCA
print(pca.inverse_transform(R2))
