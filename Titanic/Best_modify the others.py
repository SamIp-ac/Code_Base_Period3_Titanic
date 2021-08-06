import pandas as pd
import re
import numpy as np
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('../Titanic_train.csv')

test = pd.read_csv("../Titanic_test.csv")


def clean_data(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
    data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1

    data['Embarked'] = data['Embarked'].fillna('S')
    data.loc[data["Embarked"] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2


clean_data(train)
clean_data(test)

drop_column = ['Cabin']
train.drop(drop_column, axis=1, inplace=True)
test.drop(drop_column, axis=1, inplace=True)

all_data = [train, test]

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in all_data:
    dataset['FamilySize_Range'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['FamilySize_Range'] = pd.cut(dataset['FamilySize_Range'], bins=[0, 2, 5, 7, 100],
                                         labels=['Small', 'Middle', 'Large', 'Extra'])


# Define function to extract titles from passenger names


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names


for dataset in all_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# create Range for age features
for dataset in all_data:
    dataset['Age_Range'] = pd.cut(dataset['Age'], bins=[0, 12, 20, 40, 120],
                                  labels=['Children', 'Teenage', 'Adult', 'Elder'])

# create RAnge for fare features
for dataset in all_data:
    dataset['Fare_Range'] = pd.cut(dataset['Fare'], bins=[0, 7.91, 14.45, 31, 120],
                                   labels=['Low_fare', 'median_fare', 'Average_fare', 'high_fare'])

traindf = train
testdf = test

all_dat = [traindf, testdf]
for dataset in all_dat:
    drop_column = ['Age', 'Fare', 'Name', 'Ticket', 'SibSp', 'Parch']
    dataset.drop(drop_column, axis=1, inplace=True)

# Removing the passenger id from trainning set
drop_column = ['PassengerId']
traindf.drop(drop_column, axis=1, inplace=True)

# Adding the extra feataure in Train data set
traindf = pd.get_dummies(traindf, columns=["Sex", "Title", "Age_Range", "Embarked", "Fare_Range", "FamilySize_Range"],
                         prefix=["Sex", "Title", "Age_type", "Em_type", "Fare_type", "FamilySize_type"])

# Adding the extra feature in test data set
testdf = pd.get_dummies(testdf, columns=["Sex", "Title", "Age_Range", "Embarked", "Fare_Range", "FamilySize_Range"],
                        prefix=["Sex", "Title", "Age_type", "Em_type", "Fare_type", "FamilySize_type"])

target = traindf['Survived'].values
features = traindf[['Pclass', 'Sex_0', 'Sex_1',
                    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                    'Age_type_Children', 'Age_type_Teenage', 'Age_type_Adult', 'Age_type_Elder',
                    'Em_type_0', 'Em_type_1', 'Em_type_2',
                    'Fare_type_Low_fare', 'Fare_type_median_fare',
                    'Fare_type_Average_fare', 'Fare_type_high_fare',
                    'FamilySize_type_Small', 'FamilySize_type_Middle', 'FamilySize_type_Large',
                    'FamilySize_type_Extra']].values

# data imbalance
over = SMOTE()
features, target = over.fit_resample(features, target)

# Model
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=features.shape[1], units=15, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=11, kernel_initializer="uniform"))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="relu", units=11, kernel_initializer="uniform"))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()

history = classifier.fit(features, target, batch_size=5, epochs=100,
                         validation_split=0.1, verbose=1, shuffle=True)

drop_column = ['PassengerId']
testdf.drop(drop_column, axis=1, inplace=True)
testdf.head()
# predicting the results
prediction = classifier.predict_classes(testdf)

Ans = pd.DataFrame()
Ans['PassengerId'] = test['PassengerId']
Ans['Survived'] = prediction

Ans.to_csv('Ans_titanic00033.csv', index=False)
