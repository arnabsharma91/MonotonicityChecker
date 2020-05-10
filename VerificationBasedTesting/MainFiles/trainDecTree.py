
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn import tree
import numpy as np

from sklearn.tree import DecisionTreeClassifier


def functrainDecTree():
    np.random.seed(0)
    df = pd.read_csv('OracleData.csv')
    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]
    model = DecisionTreeClassifier()
    model = model.fit(X, Y)


    return model
   