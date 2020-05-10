
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def functrainDecTree():

    df = pd.read_csv('OracleData.csv')
    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]
    model = DecisionTreeClassifier()
    model = model.fit(X, Y)



    return model
    