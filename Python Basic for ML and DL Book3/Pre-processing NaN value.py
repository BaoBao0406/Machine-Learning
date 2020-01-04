import pandas as pd
from io import StringIO
import numpy as np

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)
"""
# Find all null value
print(df.isnull().sum())

# Drop row contain null
print(df.dropna(axis=0))

# Drop column contain null
print(df.dropna(axis=1))

# Drop row with all value as null
print(df.dropna(how='all'))

# Drop row that have less than 4 values
print(df.dropna(thresh=4))

# Drop row if specific column contain null
print(df.dropna(subset=['C']))
"""

# Fill the missing value with mean imputation
from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

imr = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=100)
imr = imr.fit(df.values)
imputed_data1 = imr.transform(df.values)
print(imputed_data1)

