####################### ####
# Diabetes Prediction with Logistic Regression
####################### ####

# Business Problem:

# When the characteristics are specified, it is determined whether the people have diabetes or not.
# a machine learning that can predict if they are not
# can you improve the model?

# The data set is a large data set held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the USA.
# part. People aged 21 and over living in Phoenix, the 5th largest city in the State of Arizona in the USA
# Data used for diabetes research on Pima Indian women. 768 observations and 8 numerical
It consists of # independent variables. The target variable is specified as "outcome"; 1 diabetes test result
# indicates positive, 0 indicates negative.

# Variables
# Pregnancies: Number of pregnancies
# Glucose: Glucose.
#BloodPressure: Blood pressure.
# SkinThickness: Skin Thickness
# Insulin: Insulin.
# BMI: Body mass index.
# DiabetesPedigreeFunction: A function that calculates our probability of having diabetes based on people in our ancestry.
# Age: Age (years)
# Outcome: Information about whether the person has diabetes or not. Having the disease (1) or not (0)


#1. Exploratory Data Analysis
#2. Data Preprocessing
#3. Model & Prediction
#4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
#7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
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

####################### ####
# Exploratory Data Analysis
####################### ####

df = pd.read_csv("Machine_Learning/datasets/diabetes.csv")
df.head()
df.shape

#############################
# Analysis of Target
#############################

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

# Ratio of 1 and 0
100 * df["Outcome"].value_counts() / len(df)



#############################
# Analysis of Features
#############################

df.describe().T
df.head()

# Let's look at blood pressure with histogram
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

#For Glucose
df["Glucose"].hist(bins=20)
plt.xlabel("Glucose")
plt.show()

# for all numeric variables instead of looking at them one by one

def plot_numerical_col(dataframe, numerical_col):
     dataframe[numerical_col].hist(bins=20)
     plt.xlabel(numerical_col)
     plt.show(block=True)#to prevent consecutive graphics from overwhelming each other


for col in df.columns:
     plot_numerical_col(df, col)

# Select Outcome to exclude all irrelevant variables
cols = [col for col in df.columns if "Outcome" not in col]


# for col in cols:
# plot_numerical_col(df, col)

df.describe().T

#############################
# Target vs Features
#############################
# By taking the group into group according to the target and taking the average of the numerical variables
# Let's see how the target affects independent variables in terms of their classes.

df.groupby("Outcome").agg({"Pregnancies": "mean"})

# We made a breakdown between independent variables and dependent variables. And we took the averages.
def target_summary_with_num(dataframe, target, numerical_col):
     print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
     target_summary_with_num(df, "Outcome", col)


####################### ####
# Data Preprocessing
####################### ####
df.shape
df.head()

df.isnull().sum()

df.describe().T

# upper and lower threshold values are calculated and are there any outliers in the upper and lower threshold values?
for col in cols:
     print(col, check_outlier(df, col))

# For outliers that exist within insulin values
replace_with_thresholds(df, "Insulin")

# Variables need to be standardized.
# First, we need to ensure that the models treat the variables equally. We must state that the one with larger values is not superior to the smaller one.
# Secondly, standardization is preferred so that the parameter estimation methods used provide faster and more accurate estimates.

# Robust-->>Subtracts the median from all observation unit values and divides it by the range value.
# The difference between Robust and Standard Scaler is that Robust is not affected by outliers.
# The reason we chose Robust is that it is resistant to outliers and extracts the median rather than the average from each observation unit.
# -Median is not affected by outliers. Similarly, after subtracting the median from each observation value, it is not divided by the std deviation as in the standard scaler, but is divided by the ranger value.

for col in cols:
     df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


####################### ####
# Model & Prediction
####################### ####

y = df["Outcome"] # dependent variable

X = df.drop(["Outcome"], axis=1) # argument

# Setting up the model
log_model = LogisticRegression().fit(X, y)

# Constants of this model
log_model.intercept_

# Coefficients (weights) of variables
log_model.coef_

# For prediction model
y_pred = log_model.predict(X)

y_pred[0:10]

y[0:10]


####################### ####
# Model Evaluation
####################### ####
Let's visualize the function by giving # y real values, y_pred real values
def plot_confusion_matrix(y, y_pred):
     acc = round(accuracy_score(y, y_pred), 2)
     cm = confusion_matrix(y, y_pred)
     sns.heatmap(cm, annot=True, fmt=".0f")
     plt.xlabel('y_pred')
     plt.ylabel('y')
     plt.title('Accuracy Score: {0}'.format(acc), size=10)
     plt.show()

plot_confusion_matrix(y, y_pred)


# Let's bring the direct process where these calculations are made
print(classification_report(y, y_pred))
# Success metrics for class 1 that we are interested in are interpreted according to this class.
# The class we are interested in refers to diabetes.
# Precision: 74% of our predictions are successful.
# We were able to classify those with Recall Value: 1 with 58% success.
# Harmonic mean: (F1-Score) 65% ratio has been achieved.



# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

#ROC AUC-
# It is a general metric based on the successes that may occur according to different classification threshold values.
# To calculate this metric, we need the predicted values of class 1 of the dependent variable.
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
#0.83939


####################### ####
# Model Validation: Holdout
####################### ####
# Holdout works by splitting the data set into two, train with one and test the other.

# Partition operation of the data set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     test_size=0.20, random_state=17)
# Let's install the model on the train set
log_model = LogisticRegression().fit(X_train, y_train)

# We want the log model object to predict. In response to independent variables, these people are predicted according to their diabetes status.
y_pred = log_model.predict(X_test)

# Let's calculate the probabilities of belonging to a class
y_prob = log_model.predict_proba(X_test)[:, 1]

# Performance evaluation
print(classification_report(y_test, y_pred))

# Support values decreased and changed, this represents the number of observations for each class


# Before
# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Now
# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

# As a comment;
#1-There is not much difference with the previous one, but somehow the prediction results seem to change when the model touches the data it does not see.
# --Seems to fail more. Model validation is really needed in this case.

# Let's look at the model success with ROC curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

#AUC=0.88
# Previous value AUC = 0.84, now AUC = 0.88 After the changed random states, the following method support is taken for the accuracy of which one


####################### ####
# Model Validation: 10-Fold Cross Validation
####################### ####
# Let's do cross validation.

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# We will do it using all the data.
log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                             x,y,
                             cv=5,
                             scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# Take the average
cv_results["test_accuracy"].mean()


# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
#Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
#AUC: 0.8327

####################### ####
# Prediction for A New Observation
####################### ####

x.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)




