# Adjust Rand Index
from sklearn import metrics
import numpy as np
"""
label_true = np.random.randint(1, 4, 6)
label_pred = np.random.randint(1, 4, 6)
print('Adjusted Rand-Index for random sample: %.3f' % metrics.adjusted_rand_score(label_true, label_pred))
label_true = [1, 1, 3, 3, 2, 2]
label_pred = [3, 3, 2, 2, 1, 1]
print('Adjusted Rand-Index for same structure sample: %.3f' % metrics.adjusted_rand_score(label_true, label_pred))


# Homogeneity and Completeness
label_true = [1, 1, 2, 2]
label_pred = [2, 2, 1, 1]
print('Homogeneity score for same structure sample: %.3f' % metrics.homogeneity_score(label_true, label_pred))

label_true = [1, 1, 2, 2]
label_pred = [0, 1, 2, 3]
print('Homogeneity score for each cluster come from only one class: %.3f' % metrics.homogeneity_score(label_true, label_pred))

label_true = [1, 1, 2, 2]
label_pred = [1, 2, 1, 2]
print('Homogeneity score for each cluster come from two classes: %.3f' % metrics.homogeneity_score(label_true, label_pred))


# Completeness
label_true = [1, 1, 2, 2]
label_pred = [2, 2, 1, 1]
print('Completeness score for same structure sample: %.3f' % metrics.completeness_score(label_true, label_pred))

label_true = [0, 1, 2, 3]
label_pred = [1, 1, 2, 2]
print('Completeness score for each class assign to only one cluster: %.3f' % metrics.completeness_score(label_true, label_pred))

label_true = [1, 1, 2, 2]
label_pred = [1, 2, 1, 2]
print('Completeness score for each class assign to two classes: %.3f' % metrics.completeness_score(label_true, label_pred))

label_true = np.random.randint(1, 4, 6)
label_pred = np.random.randint(1, 4, 6)
print('Completeness score for random sample: %.3f' % metrics.completeness_score(label_true, label_pred))
"""

# V-measure
label_true = [1, 1, 2, 2]
label_pred = [2, 2, 1, 1]
print('V-measure score for same structure sample: %.3f' % metrics.v_measure_score(label_true, label_pred))

label_true = [0, 1, 2, 3]
label_pred = [1, 1, 2, 2]
print('V-measure score for each classes assign to only one cluster: %.3f' % metrics.v_measure_score(label_true, label_pred))
print('V-measure score for each classes assign to only one cluster: %.3f' % metrics.v_measure_score(label_pred, label_true))

label_true = [1, 1, 2, 2]
label_pred = [1, 2, 1, 2]
print('V-measure score for each classes assign to two classes: %.3f' % metrics.v_measure_score(label_true, label_pred))

