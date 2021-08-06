import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout
import re

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

df1['Embarked'] = df1['Embarked'].fillna('S')

fare_median = df1['Fare'].median()
df1['Fare'] = df1['Fare'].fillna(fare_median)

df1['FamilySize'] = df1['SibSp'] + df1['Parch'] + 1

df1['Age_Range'] = pd.cut(df1['Age'], bins=[0, 12, 20, 40, 120], labels=['Children', 'Teenage', 'Adult',
                                                                         'Elder'])

df1['Fare_Range'] = pd.cut(df1['Fare'], bins=[-1, 7.91, 14.45, 31, 120, 10000],
                           labels=['Low_fare', 'median_fare', 'Average_fare', 'high_fare', 'Extremely_high'])


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# Create a new feature Title, containing the titles of passenger names
df1['Title'] = df1['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
df1['Title'] = df1['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df1['Title'] = df1['Title'].replace('Mlle', 'Miss')
df1['Title'] = df1['Title'].replace('Ms', 'Miss')
df1['Title'] = df1['Title'].replace('Mme', 'Mrs')
df1 = df1.drop(columns='Name')
df1 = df1.drop(columns='PassengerId')
df1 = df1.drop(columns=['Age', 'Fare', 'Ticket', 'Cabin'])

label_map1 = {'male': 1, 'female': 0}
df1['Sex'] = df1['Sex'].map(label_map1)

testdf = pd.get_dummies(df1, columns=['Sex', 'Age_Range', 'Embarked', 'Title', 'Fare_Range'],
                        prefix=['Sex', 'Age_type', 'Em_type', 'Title', 'Fare_type'])

testdf = testdf.drop(columns='Survived')
print(testdf.columns)
print(df1)
dataset = testdf


# Convert to value
sur = df1['Survived']
X = dataset.values
y = sur.values
print(X)
# data count
counter = Counter(y)
print('The number of 0,1 : ', counter)
# data imbalance
over = SMOTE()
X, y = over.fit_resample(X, y)

# DNN
# define the keras model
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=X.shape[1], units=14, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=11, kernel_initializer="uniform"))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="relu", units=11, kernel_initializer="uniform"))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()

history = classifier.fit(X, y, batch_size=3, validation_split=0.1, epochs=200, verbose=1, shuffle=True)
# fit the keras model on the dataset
score = classifier.evaluate(X, y, batch_size=5)
print('The DNN score is ', score)

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

df1['Embarked'] = df1['Embarked'].fillna('S')

fare_median = df1['Fare'].median()

df1['Fare'] = df1['Fare'].fillna(fare_median)

df1['FamilySize'] = df1['SibSp'] + df1['Parch'] + 1

df1['Age_Range'] = pd.cut(df1['Age'], bins=[0, 12, 20, 40, 120], labels=['Children', 'Teenage', 'Adult',
                                                                         'Elder'])

df1['Fare_Range'] = pd.cut(df1['Fare'], bins=[-1, 7.91, 14.45, 31, 120, 10000],
                           labels=['Low_fare', 'median_fare', 'Average_fare', 'high_fare', 'Extremely_high'])

# Create a new feature Title, containing the titles of passenger names
df1['Title'] = df1['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
df1['Title'] = df1['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df1['Title'] = df1['Title'].replace('Mlle', 'Miss')
df1['Title'] = df1['Title'].replace('Ms', 'Miss')
df1['Title'] = df1['Title'].replace('Mme', 'Mrs')

df1 = df1.drop(columns='Name')
df1 = df1.drop(columns='PassengerId')
df1 = df1.drop(columns=['Age', 'Fare', 'Ticket', 'Cabin'])

label_map1 = {'male': 1, 'female': 0}
df1['Sex'] = df1['Sex'].map(label_map1)

testdf = pd.get_dummies(df1, columns=["Sex", "Age_Range", "Embarked", 'Title', "Fare_Range"],
                        prefix=["Sex", "Age_type", "Em_type", 'Title', "Fare_type"])
# Convert to value
dataset0 = testdf
X0 = dataset0.values
print(dataset0.columns)
# Predicted
predicted0 = classifier.predict(X0, batch_size=5)
predicted0 = predicted0.round()
Ans0 = pd.DataFrame()
Ans0['PassengerId'] = df['PassengerId']
Ans0['Survived'] = predicted0

Ans0.to_csv('Ans_titanic000.csv', index=False)
