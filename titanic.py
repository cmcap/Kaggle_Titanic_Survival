import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation

def clean_data(titanic):
    # fill in all blank ages with median value
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    # female = 0, Male = 1
    titanic['Sex'] = titanic['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # All missing Embarked -> just make them embark from most common place
    if len(titanic.Embarked[titanic.Embarked.isnull() ]) > 0:
        titanic.Embarked[titanic.Embarked.isnull() ] = titanic.Embarked.dropna().mode().values
    titanic['Embarked'] = titanic['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    # Generating a familysize column
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

    # The .apply method generates a new series
    titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
    import re

    # A function to get the title from a name.
    def get_title(name):
        # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    # Get all the titles and print how often each one occurs.
    titles = titanic["Name"].apply(get_title)
    

    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {"Mr": 0, "Miss": 2, "Mrs": 3, "Master": 1, "Dr": 1, "Rev": 0, "Major": 0, "Col": 0, "Mlle": 2, "Mme": 3, "Don": 0, "Dona": 3,"Lady": 3, "Countess": 3, "Jonkheer": 0, "Sir": 0, "Capt": 0, "Ms": 2}
    for k,v in title_mapping.items():
        titles[titles == k] = v

    # Add in the title column.
    titanic["Title"] = titles
    import operator

    # A dictionary mapping family name to id
    family_id_mapping = {}

    # A function to get the id given a row
    def get_family_id(row):
        # Find the last name by splitting on a comma
        last_name = row["Name"].split(",")[0]
        # Create the family id
        family_id = "{0}{1}".format(last_name, row["FamilySize"])
        # Look up the id in the mapping
        if family_id not in family_id_mapping:
            if len(family_id_mapping) == 0:
                current_id = 1
            else:
                # Get the maximum id from the mapping and add one to it if we don't have an id
                current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
            family_id_mapping[family_id] = current_id
        return family_id_mapping[family_id]

    # Get the family ids with the apply method
    family_ids = titanic.apply(get_family_id, axis=1)

    # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
    family_ids[titanic["FamilySize"] < 3] = -1

    titanic["FamilyId"] = family_ids
    
    return titanic

# TRAIN AND PREDICT

titanic_train = pandas.read_excel("/Kaggle/Titanic/train.xlsx", sheetname='Sheet1')
titanic_test = pandas.read_excel("/Kaggle/Titanic/test.xlsx", sheetname='Sheet1')

titanic_train = clean_data(titanic_train)
titanic_test = clean_data(titanic_test)


# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize", "NameLength","Title","FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic_train[predictors], titanic_train["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare","FamilyId","Title","Fare"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ['Age', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title', 'FamilyId']],
    [LogisticRegression(random_state=1), ['Pclass', 'Sex', 'Age', 'Embarked', 'FamilySize', 'NameLength', 'Title']], 
    [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=10, min_samples_leaf=5), ['Pclass', 'Age', 'SibSp', 'Fare', 'Embarked', 'Title']],
    [SVC(random_state=1,kernel = 'rbf',C = 1.0,probability = True), ['Pclass', 'Sex', 'FamilySize', 'Title']]
]

# Create a new dataframe with only the columns Kaggle wants from the dataset.
full_predictions = []
avg_score = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.   
    alg.fit(titanic_train[predictors], titanic_train["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
    scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=6)
    avg_score.append(scores.mean())
print avg_score
print sum(avg_score)/len(avg_score)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 0 + full_predictions[1] * 0 + full_predictions[2] * 10 + full_predictions[3] * 0) / 10
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("/Kaggle/Titanic/Titanic_Submission.csv", index=False)
