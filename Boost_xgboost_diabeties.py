## Adaboosting model
## Sol for 1st Q
import pandas as pd
import numpy as np
diabeties = pd.read_csv("C:/RAVI/Data science/Assignments/Module 20 AdaBoost-Extreme Gradient Boosting/Archive/Diabetes_RF.csv")

diabeties.columns
diabeties.columns = ['Number_of_times_pregnant', 'Plasma_glucose_concentration', 'Diastolic_blood_pressure', 'Triceps_skin_fold_thickness', 'two_Hour_serum_insulin', 'Body_mass_index', 'Diabetes_pedigree_function', 'Age', 'Class_variable']

## EDA
diabeties.describe()
diabeties.shape
diabeties.dtypes
diabeties.isnull().sum()

## Histogram
import matplotlib.pyplot as plt
plt.hist(diabeties.Number_of_times_pregnant)
plt.hist(diabeties.Plasma_glucose_concentration)
plt.hist(diabeties.Diastolic_blood_pressure)
plt.hist(diabeties.Triceps_skin_fold_thickness)
plt.hist(diabeties.two_Hour_serum_insulin)
plt.hist(diabeties.Body_mass_index)
plt.hist(diabeties.Diabetes_pedigree_function)
plt.hist(diabeties.Age)
plt.hist(diabeties.Class_variable)

## Boxplot
plt.boxplot(diabeties.Number_of_times_pregnant)
plt.boxplot(diabeties.Plasma_glucose_concentration)
plt.boxplot(diabeties.Diastolic_blood_pressure)
plt.boxplot(diabeties.Triceps_skin_fold_thickness)
plt.boxplot(diabeties.two_Hour_serum_insulin)
plt.boxplot(diabeties.Body_mass_index)
plt.boxplot(diabeties.Diabetes_pedigree_function)
plt.boxplot(diabeties.Age)
plt.boxplot(diabeties.Class_variable)

## Skeewness
diabeties.Number_of_times_pregnant.skew()
diabeties.Plasma_glucose_concentration.skew()
diabeties.Diastolic_blood_pressure.skew()
diabeties.Triceps_skin_fold_thickness.skew()
diabeties.two_Hour_serum_insulin.skew()
diabeties.Body_mass_index.skew()
diabeties.Diabetes_pedigree_function.skew()
diabeties.Age.skew()


# Input and Output Split
predictors = diabeties.loc[:, diabeties.columns!="Class_variable"]
type(predictors)

target = diabeties["Class_variable"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


## Model buielding using adaboosting
from sklearn.ensemble import AdaBoostClassifier
model_ada = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

## Model fitting on train data
model_ada.fit(x_train, y_train)

## Model predicting on test data
pred_ada = model_ada.predict(x_test)

## Confusion matrics for test data
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, pred_ada)

## Accuracy for test data
accuracy_score(y_test, pred_ada)
## 0.79870129

# Evaluation on Training Data
pred_adatr = model_ada.predict(x_train)

##Confussion matrics
confusion_matrix(y_train, pred_adatr)

## ACuuracy
accuracy_score(y_train, pred_adatr)
## 0.84201954

##\/.\/\/\/\/\/\/\/\/\/\\/\

## For XGBoostong model
# Input and Output Split
predictors = diabeties.loc[:, diabeties.columns!="Class_variable"]
type(predictors)

target = diabeties["Class_variable"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


## Model buielding
## pip install xgboost
import xgboost as xgb
model = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

## FItting the model to train data
model.fit(x_train, y_train)

## Model evoluting on test data
pred = model.predict(x_test)

## Confusion matrics for test data
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, pred)

## Accuracy for test data
accuracy_score(y_test, pred)
## 0.772727272

## Finding better 'subsample', 'colsample_bytree', 'gamma', 'rag_alpha' for buield model
xgb.plot_importance(model)

model_n = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model_n, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')
grid_search.fit(x_train, y_train)
cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test))
grid_search.best_params_

## Predicting on train data
pred_tr = model.predict(x_train)

## Confussion matrics
confusion_matrix(y_train, pred_tr)

## Accuracy
accuracy_score(y_train, pred_tr)
## 1.0

