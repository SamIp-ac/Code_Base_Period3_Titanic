import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_importance, XGBRFClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# import data
dfl = pd.read_csv(r'../Titanic_train.csv')


df_1 = dfl[dfl['Sex'] == 'male']

df_bin = df_1['Age']
median_male = df_bin.median()
df_1['Age'].fillna(median_male, inplace=True)

df_female = dfl[dfl['Sex'] == 'female']

df_bin = df_female['Age']
median_female = df_bin.median()
df_female['Age'].fillna(median_female, inplace=True)

frames = [df_1, df_female]

df1 = pd.concat(frames, axis=0)

label_map0 = {'C': 1, 'Q': 2, 'S': 3}
df1['Embarked'] = df1['Embarked'].map(label_map0)

label_map1 = {'male': 1, 'female': 0}
df1['Sex'] = df1['Sex'].map(label_map1)

df1.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name', 'Embarked', 'Fare'], inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# Checking NaN
nan_values = df1[df1.isna().any(axis=1)]
print(nan_values)

df1 = df1.drop([61, 829])

# Convert to value###
X = df1[features].values
y = df1['Survived'].values

# data imbalance
over = SMOTE()
X, y = over.fit_resample(X, y)

# stacking
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC()),
              ('xgboost', XGBRFClassifier(eval_metric='logloss', use_label_encoder=False))]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stackingCl.fit(X, y)

cv = cross_val_score(stackingCl, X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of stacking is : ', score)

# Do it
df = pd.read_csv(r'../Titanic_test.csv')
df_1 = df[df['Sex'] == 'male']

df_bin = df_1['Age']
median_male = df_bin.median()
df_1['Age'].fillna(median_male, inplace=True)

df_female = df[df['Sex'] == 'female']

df_bin = df_female['Age']
median_female = df_bin.median()
df_female['Age'].fillna(median_female, inplace=True)

frames = [df_1, df_female]

df1 = pd.concat(frames, axis=0)

label_map1 = {'male': 1, 'female': 0}
df1['Sex'] = df1['Sex'].map(label_map1)

df1.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name', 'Fare', 'Embarked'], inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# Checking NaN
nan_values = df1[df1.isna().any(axis=1)]
print(nan_values)

# Convert to value
X0 = df1[features].values
# pred.
Prediction = stackingCl.predict(X0)
counter_p = Counter(Prediction)
print('The number of 0,1 : ', counter_p)

Ans = pd.DataFrame()
Ans['PassengerId'] = df['PassengerId']
Ans['Survived'] = Prediction
print(Ans)

Ans.to_csv('Ans_titanic0003.csv', index=False)

