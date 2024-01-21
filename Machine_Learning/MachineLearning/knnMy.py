################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("Machine_Learning/datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

#######################
# 2. Data Preprocessing & Feature Engineering
#######################
# If the variables are standard, the results will be faster or more successful.
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Variable values come standardized
X_scaled = StandardScaler().fit_transform(X)

#To add column names
X = pd.DataFrame(X_scaled, columns=X.columns)

#######################
#3. Modeling & Prediction
#######################
# We use knn as distance based method
# Don't fit
knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)
# Don't guess
knn_model.predict(random_user)

#######################
#4. Model Evaluation
#######################

# y_pred for confusion matrix:
y_pred = knn_model.predict(X)

# y_prob for AUC:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
#acc 0.83
#f1 0.74
# AUC
roc_auc_score(y, y_prob)
#0.90

# Cross validation
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


#0.73
#0.59
#0.78

# 1. Sample size can be increased.
# 2. Data preprocessing can be detailed
#3. Feature engineering new variables can be derived
#4. Optimizations can be made for the relevant algorithm.

# Neighborhood count hypermeter can be changed. Let's look at the parameters.
knn_model.get_params()
# Parameter: These are the weights that the models learn from the data.
#-The weights are the estimators of those parameters. The parameters are learned from the data.
# Hyperparameters: These are external parameters that must be defined by the user and cannot be learned from the data set.


#######################
#5. Hyperparameter Optimization
#######################

knn_model = KNeighborsClassifier()
knn_model.get_params()

# We create the number of parameters
knn_params = {"n_neighbors": range(2, 50)}

# For the search process (we evaluate the error looking process 5 times)
knn_gs_best = GridSearchCV(knn_model,
                            knn_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X, y)

knn_gs_best.best_params_

#######################
#6. Final Model
#######################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                             x,
                             y,
                             cv=5,
                             scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# If we want to check the diabetes status of any other user

random_user = X.sample(1)
#let's apply it to the model and see the result
knn_final.predict(random_user)