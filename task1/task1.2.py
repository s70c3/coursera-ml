import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pandas.read_csv('titanic.csv', index_col='PassengerId')
df = df[['Pclass', 'Fare','Age', 'Sex', 'Survived']]
df=df.dropna(axis=0)
X = df[['Pclass', 'Fare','Age', 'Sex']]
X = X.replace(to_replace=['male', 'female'], value=[1, 0])


y = df['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_
print(importances)