import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from xgboost import XGBClassifier

pd.options.mode.chained_assignment = None  # default='warn'

raw_train = pd.read_csv('train.csv')
raw_test  = pd.read_csv('test.csv')

changed_train = raw_train.copy(deep = True)
changed_test  = raw_test.copy(deep = True)

clean_data = [changed_train, changed_test]

for dataset in clean_data:
	dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
	dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

drop_column = ['PassengerId','Cabin', 'Ticket']
changed_train.drop(drop_column, axis=1, inplace = True)
changed_test.drop(drop_column, axis=1, inplace = True)

for dataset in clean_data:
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
	dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
	dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

title_names = changed_train['Title'].value_counts() < 10
changed_train['Title'] = changed_train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

label = LabelEncoder()
for dataset in clean_data:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

Target = ['Survived']

#define x variables for original w/bin features to remove continuous variables
changed_train_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

XGBClassifier()

alg = XGBClassifier()

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
cv_results = model_selection.cross_validate(alg, changed_train[changed_train_x_bin], pd.Series.ravel(changed_train[Target]), cv  = cv_split, return_train_score=True)

print(cv_results['train_score'].mean())
print(cv_results['test_score'].mean())