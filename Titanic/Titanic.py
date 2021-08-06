import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_importance, XGBRFClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import metrics

# import data
df0 = pd.read_csv(r'../Titanic_train.csv')

df_1 = df0[df0['Sex'] == 'male']

df_bin = df_1['Age']
median_male = df_bin.median()
df_1['Age'].fillna(median_male, inplace=True)

df_female = df0[df0['Sex'] == 'female']

df_bin = df_female['Age']
median_female = df_bin.median()
df_female['Age'].fillna(median_female, inplace=True)

frames = [df_1, df_female]

df1 = pd.concat(frames, axis=0)

label_map0 = {'C': 1, 'Q': 2, 'S': 3}
df1['Embarked'] = df1['Embarked'].map(label_map0)

label_map1 = {'male': 1, 'female': 0}
df1['Sex'] = df1['Sex'].map(label_map1)

df1.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name'], inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']

# Checking NaN
nan_values = df1[df1.isna().any(axis=1)]
print(nan_values)

df1 = df1.drop([61, 829])

# Convert to value###
X = df1[features].values
y = df1['Survived'].values

# data imbalance
counter = Counter(y)
print('The number of 0,1 : ', counter)

# data imbalance
over = SMOTE()
X, y = over.fit_resample(X, y)

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=27)

# Cross_val
np.random.seed(27)
cv = cross_val_score(KNeighborsClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of Knn is : ', score)

np.random.seed(27)
cv = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of RandomForest is : ', score)

np.random.seed(27)
cv = cross_val_score(DecisionTreeClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of DecisionTree is : ', score)

# XGboosting
np.random.seed(27)
cv = cross_val_score(XGBRFClassifier(eval_metric='logloss', use_label_encoder=False), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of xgboost is : ', score)

# Stacking
np.random.seed(27)
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC()),
              ('xgboost', XGBRFClassifier(eval_metric='logloss', use_label_encoder=False))]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
######
cv = cross_val_score(stackingCl, X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of stacking is : ', score)

# predict---xgboost
np.random.seed(77)
xgboostModel = XGBRFClassifier(n_estimators=100, learning_rate=0.3)
xgboostModel.fit(X_train, y_train)

predicted1 = xgboostModel.predict(X_test)

print('The score XGBoost(test) is : ', xgboostModel.score(X_test, y_test))

# features importance
xgboostModel.fit(X, y)
plot_importance(xgboostModel)
plt.show()

# accuracy score
print('-----------------classification_report of xgboost-------------------')
print(metrics.classification_report(y_test, predicted1))
print('jaccard_similarity_score', metrics.jaccard_score(y_test, predicted1))
print('log_loss', metrics.log_loss(y_test, predicted1))
print('zero_one_loss', metrics.zero_one_loss(y_test, predicted1))
print('AUC&ROC', metrics.roc_auc_score(y_test, predicted1))
print('matthews_corrcoef', metrics.matthews_corrcoef(y_test, predicted1))

sns.heatmap(metrics.confusion_matrix(y_test, predicted1), annot=True)
plt.title("Confusion Matrix XG boosting")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# predict--stacking
np.random.seed(77)
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC()),
              ('xgboost', XGBRFClassifier(eval_metric='logloss', use_label_encoder=False))]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stackingCl.fit(X_train, y_train)

predicted2 = stackingCl.predict(X_test)

print('The score stacking(test) is : ', stackingCl.score(X_test, y_test))

# accuracy score
print('-----------------classification_report of stacking-------------------')
print(metrics.classification_report(y_test, predicted2))
print('jaccard_similarity_score', metrics.jaccard_score(y_test, predicted2))
print('log_loss', metrics.log_loss(y_test, predicted2))
print('zero_one_loss', metrics.zero_one_loss(y_test, predicted2))
print('AUC&ROC', metrics.roc_auc_score(y_test, predicted2))
print('matthews_corrcoef', metrics.matthews_corrcoef(y_test, predicted2))

sns.heatmap(metrics.confusion_matrix(y_test, predicted2), annot=True)
plt.title("Confusion Matrix Stacking")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Do it
df = pd.read_csv(r'../Titanic_test.csv')
df_1 = df[df['Sex'] == 'male']

df_bin = df_1['Age']
mean_male = df_bin.mean()
df_1['Age'].fillna(mean_male, inplace=True)

df_female = df[df['Sex'] == 'female']

df_bin = df_female['Age']
mean_female = df_bin.mean()
df_female['Age'].fillna(mean_female, inplace=True)

frames = [df_1, df_female]

df1 = pd.concat(frames, axis=0)

label_map0 = {'C': 1, 'Q': 2, 'S': 3}
df1['Embarked'] = df1['Embarked'].map(label_map0)

label_map1 = {'male': 1, 'female': 0}
df1['Sex'] = df1['Sex'].map(label_map1)

df1.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name'], inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']

# Checking NaN
nan_values = df1[df1.isna().any(axis=1)]
print(nan_values)

df_bin = df['Fare']
mean_fare = df_bin.mean()
df['Fare'].fillna(mean_fare, inplace=True)

# Convert to value
X0 = df1[features].values

# Normalize
scaler = MinMaxScaler()
X0 = scaler.fit_transform(X0)
# fit
xgboostModel.fit(X, y)

# Predicted
predicted = xgboostModel.predict(X0)
Ans = pd.DataFrame()
Ans['PassengerId'] = df['PassengerId']
Ans['Survived'] = predicted
print(Ans)

Ans.to_csv('Ans_titanic.csv001', index=False)
