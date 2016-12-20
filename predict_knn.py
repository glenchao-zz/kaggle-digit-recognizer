# http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.neighbors import KNeighborsClassifier
# https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii

import pandas as pd
import numpy as np

train = pd.read_csv("./train.csv", header=0)
test = pd.read_csv("./test.csv", header=0)

target = train["label"]
train.drop("label", axis=1, inplace=True)

train_data = train.values
test_data = test.values

knn = KNeighborsClassifier(n_neighbors=10)
knn = knn.fit(train_data, target)
output = knn.predict(test_data)

df_output = pd.DataFrame()
df_output["ImageId"] = range(1, len(output)+1)
df_output["Label"] = output
df_output[["ImageId", "Label"]].to_csv("./prediction_knn.csv", index=False)