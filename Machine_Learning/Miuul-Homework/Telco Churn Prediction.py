#################
# Telco Churn Prediction (Creating a Customer Abandonment Prediction Model)
#################

####################
# Business Problem
####################
# It is expected to develop a machine learning model that can predict customers who will leave the company.

#######################
# Dataset Story
#######################

# Telco customer churn data includes information about a fictitious telecom company that provided home phone and Internet services to 7,043 customers in California in the third quarter.
# Shows which customers left, stayed, or signed up for their service.
#21 Variable 7043 Observations 977.5 KB
# CustomerId : Customer ID
# Gender
# SeniorCitizen : Whether the customer is elderly (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents: Whether the customer has dependents (Yes, No
# tenure: Number of months the customer stays with the company
# PhoneService: Whether the customer has phone service (Yes, No)
# MultipleLines: Whether the customer has more than one line (Yes, No, No phone service)
# InternetService: Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, no Internet service)
# OnlineBackup: Whether the customer has an online backup (Yes, No, no Internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, no Internet service)
# TechSupport: Whether the customer receives technical support (Yes, No, no Internet service)
# StreamingTV : Whether the customer has broadcast TV (Yes, No, no Internet service)
# StreamingMovies: Whether the customer is streaming movies (Yes, No, No internet service)
# Contract: Customer's contract period (Month to month, One year, Two years)
# PaperlessBilling: Whether the customer has a paperless bill (Yes, No)
# PaymentMethod: Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: Amount collected from the customer monthly
# TotalCharges: Total amount collected from the customer
# Churn: Whether the customer uses it (Yes or No)

####################
# Task 1: Exploratory Data Analysis
####################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 300)

df= pd.read_csv("Machine_Learning/datasets/Telco-Customer-Churn-2.csv")
df.head()
df.shape
df.info()
df["Contract"].value_counts()

# We convert this variable into a numerical variable with the Numeric method

df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

# We update the variable consisting of Yes and No to 1 and 0. We express the model in an understandable way.
df["Churn"]=df["Churn"].apply(lambda x :1 if x == "Yes" else 0)


#############################
# General situation
##########################

def check_df(dataframe, head=5):
     print("################################## Shape #################")
     print(dataframe.shape)
     print("################################## Types #################")
     print(dataframe.dtypes)
     print("################################## Head ##################")
     print(dataframe.head())
     print("################################## Tail ################")
     print(dataframe.tail())
     print("##################################NA ##################")
     print(dataframe.isnull().sum())
     print("################################## Quantiles #################")
     print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99,1]).T)

check_df(df)
# Step 2: Make the necessary arrangements. (such as variables with type errors)
# Step 3: Observe the distribution of numerical and categorical variables in the data.
# Step 4: Examine the target variable with categorical variables.
# Step 5: Examine if there are any unusual observations.
# Step 6: Check if there are any missing observations.
####################
# Task 2: Feature Engineering
####################
# Step 1: Take the necessary action for missing and contradictory observations.
# Step 2: Create new variables.
# Step 3: Perform the encoding operations.
# Step 4: Standardize numerical variables.
####################
# Task 3: Modeling
####################
# Step 1: Build models with classification algorithms and examine accuracy scores. Choose the 4 best models.
# Step 2: Perform hyperparameter optimization with the models you selected. Re-establish the model with the hyperparameters you found.
# Step 1: Capture numerical and categorical variables.

def grab_col_names(dataframe, cat_th=10, car_th=20):
# We determined the threshold values ourselves by checking the acceptance above these ranges
     #Important information###
     """

     It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
     Note: Categorical variables with numerical appearance are also included.

     parameters
     ------
         dataframe: dataframe
                 Dataframe from which variable names are to be taken
         cat_th: int, optional
                 Class threshold value for variables that are numeric but categorical
         car_th: int, optinal
                 class threshold for categorical but cardinal variables

     returns
     ------
         cat_cols: list
                 Categorical variable list
         num_cols: list
                 Numerical variable list
         cat_but_car: list
                 List of cardinal variables with categorical view

     examples
     ------
         import seaborn as sns
         df = sns.load_dataset("iris")
         print(grab_col_names(df))


     Notes
     ------
         cat_cols + num_cols + cat_but_car = total number of variables
         num_but_cat is inside cat_cols.
         The sum of the 3 lists that return equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

     """
     # cat_cols, cat_but_car
     cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
     num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                    dataframe[col].dtypes != "O"]
     cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                    dataframe[col].dtypes == "O"]
     cat_cols = cat_cols + num_but_cat
     cat_cols = [col for col in cat_cols if col not in cat_but_car]

     #num_cols
     num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
     num_cols = [col for col in num_cols if col not in num_but_cat]

     print(f"Observations: {dataframe.shape[0]}")
     print(f"Variables: {dataframe.shape[1]}")
     print(f'cat_cols: {len(cat_cols)}')
     print(f'num_cols: {len(num_cols)}')
     print(f'cat_but_car: {len(cat_but_car)}')
     print(f'num_but_cat: {len(num_but_cat)}')
     return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

# Categorical Variables Analysis

def cat_summary(dataframe, col_name, plot=False):
     print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
     print("####################################################"
     if plot:
         sns.countplot(x=dataframe[col_name], data=dataframe)
         plt.show()

for col in cat_cols:
     cat_summary(df, col, plot=True)


# Analysis of Numerical Variables

def num_summary(dataframe, numerical_col, plot=False):
     quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
     print(dataframe[numerical_col].describe(quantiles).T)

     if plot:
         dataframe[numerical_col].hist(bins=20)
         plt.xlabel(numerical_col)
         plt.title(numerical_col)
         plt.show()


for col in num_cols:
     num_summary(df, col, plot=True)


# Analysis of Numerical Variables According to Target

def target_summary_with_num(dataframe, target, numerical_col):
         print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

for col in num_cols:
     target_summary_with_num(df, "Churn", col)

# Analysis of Categorical Variables According to Target

def target_summary_with_cat(dataframe, target, categorical_col):
         print(categorical_col)
         print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                             "Count" : dataframe[categorical_col].value_counts(),
                             "Ratio" : 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
     target_summary_with_cat(df, "Churn", col)

#Correlation

df[num_cols].corr()

# Correlation Matrix
f, ax= plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalCharges appears to have high correlation with monthly charges and tenure.

df.corrwith(df["Churn"]).sort_values(ascending=False)

# Missing Value Analysis
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns= missing_values_table(df, na_name=True)
df["Contract"].value_counts()

# Let's fill in the totalcharge with the monthly payment amounts

df["TotalCharges"].fillna(df["TotalCharges"].median(),inplace=True)
df.isnull().sum()
df["Contract"].value_counts()

# BASE MODEL SETUP

dff=df.copy()

cat_cols=[col for col in cat_cols if col not in["Churn"]]
cat_cols
dff.head()


# One-Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
     return dataframe
dff= one_hot_encoder(dff, cat_cols, drop_first=True)



y= dff["Churn"]
X= dff.drop(["Churn","customerID"], axis=1)

models=[("LR",LogisticRegression(random_state=12345)),
         ("KNN",KNeighborsClassifier()),
         ("CART",DecisionTreeClassifier(random_state=12345)),
         ("RF",RandomForestClassifier(random_state=12345)),
         ("SVM",SVC(gamma="auto", random_state=12345)),
         ("XGB",XGBClassifier(random_state=12345)),
         ("LightGBM",LGBMClassifier(random_state=12345)),
         ("CatBoost",CatBoostClassifier(verbose=False,random_state=12345))]

for name, model in models:
     cv_results= cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
     print(f"#############{name} ###########")
     print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)})")
     print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)})")
     print(f"Recall: {round(cv_results['test_recall'].mean(), 4)})")
     print(f"Precision: {round(cv_results['test_precision'].mean(), 4)})")
     print(f"F1: {round(cv_results['test_f1'].mean(), 4)})")

df["Contract"].value_counts()

# Outliers Analysis


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
     quartile1 = dataframe[col_name].quantile(q1)
     quartile3 = dataframe[col_name].quantile(q3)
     interquantile_range = quartile3 - quartile1
     up_limit = quartile3 + 1.5 * interquantile_range
     low_limit = quartile1 - 1.5 * interquantile_range
     return low_limit, up_limit

def check_outlier(dataframe, col_name):
     low_limit, up_limit = outlier_thresholds(dataframe, col_name)
     if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
         return True
     else:
         return False

def replace_with_thresholds(dataframe, variable):
     low_limit, up_limit = outlier_thresholds(dataframe, variable)
     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Outlier analysis and Suppression Process
for col in num_cols:
     print(col, check_outlier(df,col))

     if check_outlier(df, col):
         replace_with_thresholds(df, col)

# Feature Extraction
df.loc[(df["tenure"]>=0) & (df["tenure"] <=12), "NEW_TENURE_YEAR"]= "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"] <=24), "NEW_TENURE_YEAR"]= "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"] <=36), "NEW_TENURE_YEAR"]= "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"] <=48), "NEW_TENURE_YEAR"]= "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"] <=60), "NEW_TENURE_YEAR"]= "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"] <=72), "NEW_TENURE_YEAR"]= "5-6 Year"

# Class distribution of categorical variables
df["NEW_TENURE_YEAR"].value_counts()

df["Contract"].value_counts()

# Specify customers with 1 or 2 year contracts as Engaged
df["NEW_Engaged"] = df["Contract"].apply(lambda x:1 if x in["One year","Two year"] else 0)

# People who do not receive any support, backup or protection
df["NEW_noProt"]= df.apply(lambda x:1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport" ] != "Yes") else 0, axis=1)

# Customers who have monthly contracts and are young
df["NEW_Young_Not_Engaged"]= df.apply(lambda x:1 if (x["NEW_Engaged"]== 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Total number of services received by the person
df["New_TotalServices"] = (df[["PhoneService", "InternetService", "OnlineSecurity",
                                "OnlineBackup","DeviceProtection","TechSupport",
                                "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

# Total number of services received by the person
df["NEW_FLAG_ANY_STREAMING"]= df.apply(lambda x:1 if(x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Does the person make automatic payments?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x:1 if x in ["Bank transfer(automatic)","Credit card (automatic)"] else 0)
df[df['tenure'] == 0]


# Average Monthly payment
df["New_AVG_Charges"]= df["TotalCharges"] / (df["tenure"]+1)

# Increase of the current price compared to the average price
df["NEW_Increase"]= df["New_AVG_Charges"] / df["MonthlyCharges"]

# Fee per service
df["New_AVG_Service_Fee"]= df["MonthlyCharges"] / (df["New_TotalServices"] + 1)

df.shape

# Encoding operations (We convert from categorical to numerical)
cat_cols, num_cols, cat_but_car= grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
     labelencoder = LabelEncoder()
     dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
     return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes =="O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
     label_encoder(df, col)

# One Hot Encoding Process
# update process of cat_cols list
cat_cols= [col for col in cat_cols if col not in binary_cols and col not in["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
     return dataframe

df= one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape


# Modelling

y= df["Churn"]
X= df.drop(["Churn","customerID"], axis=1)

models=[("LR",LogisticRegression(random_state=12345)),
         ("KNN",KNeighborsClassifier()),
         ("CART",DecisionTreeClassifier(random_state=12345)),
         ("RF",RandomForestClassifier(random_state=12345)),
         ("SVM",SVC(gamma="auto", random_state=12345)),
         ("XGB",XGBClassifier(random_state=12345)),
         ("LightGBM",LGBMClassifier(random_state=12345)),
         ("CatBoost",CatBoostClassifier(verbose=False,random_state=12345))]

for name, model in models:
     cv_results= cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
     print(f"#############{name} ###########")
     print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)})")
     print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)})")
     print(f"Recall: {round(cv_results['test_recall'].mean(), 4)})")
     print(f"Precision: {round(cv_results['test_precision'].mean(), 4)})")
     print(f"F1: {round(cv_results['test_f1'].mean(), 4)})")

#############################
# Random Forests
#############################

rf_model= RandomForestClassifier(random_state=17)

rf_params={"max_depth":[5,8, None],
            "max_features":[3,5,7,"auto"],
            "min_samples_split":[2,5,8,15,20],
            "n_estimators":[100, 200, 500]}

rf_best_grid= GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=0).fit(X,y)

rf_best_grid.best_params_
rf_best_grid.best_score_

rf_final =rf_model.set_params(**rf_best_grid.best_params_,random_state=17).fit(X,y)

# K-Fold Cross Validation uses the dataset to evaluate classification models and train the model.
# --is one of the partitioning methods.
cv_results= cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#############################
#XGBoost
#############################

xgboost_model= XGBClassifier(random_state=17)

xgboost_params={"learning_rate":[0.1,0.01, 0.001],
                 "max_depth":[5,8,12,15,20],
                 "n_estimators":[100, 500, 1000],
                 "colsample_bytree":[0.5, 0.7, 1]}


xgboost_best_grid= GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)
xgboost_final =xgboost_model.set_params(**xgboost_best_grid.best_params_,random_state=17).fit(X,y)

cv_results= cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#############################
#LightGBM Model
#############################

lgbm_model= LGBMClassifier(random_state=17)

lgbm_params= {"learning_rate":[0.01, 0.1, 0.001],
              "n_estimators":[100,300, 500, 1000],
              "colsample_bytree":[0,5, 0.7, 1]}

lgbm_best_grid= GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

lgbm_final =lgbm_model.set_params(**lgbm_best_grid.best_params_,random_state=17,).fit(X,y)

cv_results= cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#############################
#CatBoost
#############################

catboost_model= CatBoostClassifier(random_state=17, verbose=False)

catboost_params={"iterations":[200, 500],
                  "learning_rate":[0.01, 0.1],
                  "depth":[3, 6]}

# If you set the max_dept parameter to 5, the depth of the decision tree will be at most 5.
# In this way, you can make adjustments about the generality of the tree.
# Deeper trees are more prone to overlearning, while shallower trees are more generalists.


catboost_best_grid= GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_,random_state=17).fit(X,y)

cv_results= cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#######################
#FeatureImportance
#######################

def plot_importance(model, features, num=len(X), save=False):
     feature_imp= pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
     plt.figure(figsize=(10, 10))
     sns.set(font_scale=1)
     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num])

     plt.title("Features")
     plt.tight_layout()
     plt.show()
     if save:
         plt.savefig("importances.png")

# Orders of importance in the data

plot_importance(rf_final,X)
plot_importance(xgboost_final,X)
plot_importance(lgbm_final,X)
plot_importance(catboost_final,X)



