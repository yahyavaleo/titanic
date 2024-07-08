# titanic

The [Titanic competition]("https://www.kaggle.com/competitions/titanic/overview") is based on the infamous shipwreck of Titanic in April 15, 1912. The goal of this competition is to create a model that predicts which passengers survived the Titanic shipwreck.

My solution to this competition is in the form of the Jupyter Notebook, `titanic.ipynb`. In this notebook:

- I did a detailed analysis of the input features in order to understand what impact does each feature has on the target.
- After selecting suitable features, I developed strategies to fill in missing values and created suitable encoding schemes for non-numerical features.
- I developed a complete pipeline to automatically perform the necessary data preprocessing.
- Then, I tested a lot of commonly used classification models. From which, I found out that Support Vector Machine Classifier and Random Forest Classifer are the most promising.
- Finally, I ran grid searches on these two classifiers to find the best set of parameters. The best model achieved roughly 82% accuracy on cross-validation, and I used this model to predict which passengers survived in the test set.

## Requirements

The notebook requires a Python environment with the following packages installed:

```txt
numpy==2.0.0
pandas==2.2.2
scikit_learn==1.3.0
matplotlib==3.8.0
seaborn==0.13.2
```

These packages can be installed by running the following command:

```sh
pip install -r requirements.txt
```

## Dataset

The training file, `train.csv`, consists of the following columns:

<!-- prettier-ignore -->
| Variable | Definition | Note |
| - | - | - |
| `PassengerId` | Dummy ID assigned to each passenger | |
| `Survived` | Whether the passenger survived | `1` means survived, `0` means didn't survive |
| `Name` | Name of the passenger | |
| `Pclass` | Ticket class - a proxy for socio-economic status | `1` means upper class, `2` means middle class, and `3` means lower class |
| `Sex` | Gender - male or female | |
| `Age` | Age in years | For children younger than 1 y/o, the age is fractional. If age is estimated, it has `.5` added. |
| `SibSp` | Count of siblings and/or spouses abroad the Titanic | Siblings include brothers, sisters, stepbrothers, and stepsisters. Spouses include husband or wife. |
| `Parch` | Count of parents and/or children abroad the Titanic | Parents include father and mother. Children include son, daughter, stepson, and stepdaughter. |
| `Ticket` | Ticket number | |
| `Fare` | Passenger fare (in USD) | |
| `Cabin` | Cabin number | |
| `Embarked` | Name of the port from which the passenger went on board to Titanic | `C` means Cherbourg, `Q` means Queenstown, `S` means Southampton |

The test file, `test.csv`, consists of similar columns, except `Survived`, which is the target variable to be predicted.

## Observations

After analyzing the relationships between the given features and the survival status, I found that:

- Upper class passengers (`Pclass=1`) were very likely to survive (survival likelihood of $70\%$).
- Middle class passengers (`Pclass=2`) were slightly more likely to **not** survive (surival likelihood of $-10\%$).
- Lower class passengers (`Pclass=3`) were very likely to **not** survive (survival likelihood of $-68\%$).
- Female passengers (`Gender="female"`) were very likely to survive (survival likelihood of $188\%$).
- Male passengers (`Gender="male"`) were very likely to **not** survive (survival likelihood of $-77\%$).
- Older passengers (in terms of `Age`) were slightly more likely to **not** survive.
- Passengers with no siblings/spouses (`SibSp=0`) were more likely to **not** survive.
- Passengers with just one siblings/spouses (`SibSp=1`) were very likely to survive.
- Passengers with two siblings/spouses (`SibSp=2`) were slightly more likely to survive.
- Passengers with more than two siblings/spouses (`SibSp>2`) were more likely to **not** survive.
- Passengers with no parents/children (`Parch=0`) were more likely to **not** survive.
- Passengers with one or more parents/children (`Parch>0`) were generally more likely to survive.
- Passengers with larger ticket fares (in terms of `Fare`) were more likely to survive.
- Passengers who embarked from Cherbourg (`Embarked="C"`) were more likely to survive (survival likelihood of $24\%$).
- Passengers who embarked from Queenstown (`Embarked="Q"`) were more likely to **not** survive (survival likelihood of $-36\%$).
- Passengers who embarked from Southampton (`Embarked="S"`) were more likely to **not** survive (survival likelihood of $-49\%$).

## Data Imputation

In the training set, the columns `Age` and `Embarked` had missing values, while in the test set, apart from `Age` and `Embarked`, the `Fare` column also had missing values.

I developed these strategies to fill in the missing values:

- For `Age`, I discovered a linear relationship with other column values, and used that relationship to fill in the missing values.
- For `Fare`, I used the mean value of the column.
- For `Embarked`, I used the most frequent value of the column.

In order to discover the linear relationship between `Age` and other columns, I analyzed that only features which can meaningfully relate to age are:

- `Pclass`: Rich people are usually older (it takes time to build a large fortune).
- `SibSp`: Older people are more likely to travel without siblings and usually have at most one spouse. And if they are younger, they are likely travelling with their parents, who paid for them, and therefore, likely have siblings, who are also travelling with them.
- `Parch`: Younger people will likely travel with both their parents and have no children. Middle aged people (~30) will travel without their parents and would no children. Older people will have children, who they have brought with them to travel.

I verified for correlation among these features used linear regression to find the relationship.

## Feature Engineering

All columns except for `Sex` and `Embarked` are numeric, so they work fine with machine learning algorithms. These two columns are both categorical, so I one-hot encoded them.

I also removed the `PassengerId` and `Name` columns, since they do not hold any useful information. I also removed the `Ticket` and `Cabin` columns, since they are complex strings which are hard to process.

## Preprocessing Pipeline

To make preprocessing manageable, I created a pipeline that contains the following steps:

1. The specified features are selected from the pandas dataframe.
2. The `Age` column is imputed using the linear regression model.
3. The `Fare` column is imputed using the mean value.
4. The `Embarked` column is imputed using the most frequent value.
5. The `Sex` column is one-hot encoded.
6. The `Embarked` column is one-hot encoded.
7. The feature vector is converted to a numpy array.
8. The feature vector is scaled using a `StandardScaler`.

## Model Selection

I trained many commonly used classification models and found out that Support Vector Machine Classifiers and Random Forest Classifiers achieved the highest accuracy. I then used grid search to search for the best parameters for these models.

Finally, I used the best model to make predictions on the test set, which achieved an accuracy score of $79.4\%$.
