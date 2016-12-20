# http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
# https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii

import pandas as pd
import numpy as np

train = pd.read_csv("./train.csv", header=0)
test = pd.read_csv("./test.csv", header=0)

target = train["label"]
train.drop("label", axis=1, inplace=True)

train_data = train.values
test_data = test.values

forest = RandomForestClassifier(n_estimators = 210, max_depth=4, criterion='gini')
forest = forest.fit(train_data, target)
output = forest.predict(test_data)

df_output = pd.DataFrame()
df_output["ImageId"] = range(1, len(output)+1)
df_output["Label"] = output
df_output[["ImageId", "Label"]].to_csv("./prediction_randomforest.csv", index=False)