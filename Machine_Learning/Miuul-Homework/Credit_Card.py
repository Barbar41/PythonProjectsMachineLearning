#################
# Examining the Data Set
#################

# It is important for credit card companies to detect fraud, they do not want their customers to be charged incorrectly.
# It is desired to create a model that detects fraudulent credit cards using this data set.
# The data set consists of transactions made by credit card in Europe in September 2013, and these transactions are labeled as 1 if fraud (fraud), otherwise 0.
# Due to confidentiality, there is not much background information and except for the "Time", "Amount" variable, other variables have been transformed with PCA (Principal component analysis).
# "Time": seconds between the first transaction and each transaction
# "Amount": cost
# First, we will read the data set, observe whether there are any empty values, and look at the distribution of the Class variable.
# Afterwards, we standardize the "Amount" and "Time" variables.
# We separate the data with the hold out method and create the model with logistic regression, which is a classification model.
# We will go through a single model.

#############################
# Required library and functions
#############################

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report,f1_score,recall_score,roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc,rcParams
import itertools

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Reading the data set
df = pd.read_csv("../datasets/creditcard.csv")
df.head()

# Number of variables and observations in the data set
print("Number of observations: ",len(df))
print("Number of variables: ", len(df.columns))

# we want to observe the types of variables in the data set and whether they contain empty values
df.info()

# The presence of class 1 in the data set is 0.2%, and class 0 is 99.8%.
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Class'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('distribution')
ax[0].set_ylabel('')
sns.countplot('Class',data=df,ax=ax[1])
ax[1].set_title('Class')
plt.show()

# Standardize Time and Amount variables
rob_scaler = RobustScaler()
df['Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.head()

# We apply the hold out method and divide the data set into training and testing (80%, 20%).
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123456)

# Defining and training the model and success score
model = LogisticRegression(random_state=123456)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f"%(accuracy))

# Accuracy is the ratio of correct predictions made in the system to all predictions.
# The accuracy score of the model we created is 0.999. We can say that our model works perfectly, right?
# Let's look at the Confusion Matrix to examine its performance.

#######################
#ConfusionMatrix
#######################
# Confusion Matrix is a table used to describe the performance of the actual values of a classification model on test data.
# Contains 4 different combinations of estimated and actual values.
# True Positives (TP): Positive was predicted and it is true.
# True Negative (TN): Negative was predicted and it is true.
# False Positive (FP): Positive was predicted and it is wrong.
# False Negative (FN): Negative was predicted and it is wrong.

def plot_confusion_matrix(cm, classes,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):

     plt.rcParams.update({'font.size': 19})
     plt.imshow(cm, interpolation='nearest', cmap=cmap)
     plt.title(title,fontdict={'size':'16'})
     plt.colorbar()
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation=45,fontsize=12,color="blue")
     plt.yticks(tick_marks, classes,fontsize=12,color="blue")
     rc('font', weight='bold')
     fmt = '.1f'
     thresh = cm.max()
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="red")

     plt.ylabel('True label',fontdict={'size':'16'})
     plt.xlabel('Predicted label',fontdict={'size':'16'})
     plt.tight_layout()

plot_confusion_matrix(confusion_matrix(y_test, y_pred=y_pred), classes=['Non Fraud','Fraud'],
                       title='Confusion matrix')

# A total of 56875 predictions were made for the Non Fraud class, 56870 (TP) of which were correct and 5 (FP) were incorrect.
# A total of 87 predictions were made for the Fraud class, 31 (FN) wrong and 56 (TN) correct.
# The model tells us that it can predict the Fraud situation with an accuracy of 0.99. But when we examine the Confusion Matrix, the rate of incorrect predictions in the Fraud class is quite high. While it is successful in predicting the majority class, it is not successful in predicting the minority class. In other words, the model correctly predicts the non-fraud class with a rate of 0.99.
# The fact that the number of observations belonging to the Non Fraud class is more than the number of observations belonging to the fraud class causes the model to be successful in predicting the Non Fraud class.
# With this observation, we can say that accuarcy score is not a good performance measurement for classification models, especially if it contains imbalance as in the data set we have.
# We have examined the dataset, we can look at how we can deal with the imbalance and create a model, what methods can be applied and with what metrics we can measure its performance.

#######################
# Classification Report
#######################

# We saw that the accuracy value (Accuracy Score) was not sufficient. We need to look at different metrics to measure the performance of the model.
# Precision: Shows how much of what is predicted as positive is actually positive.
# --If precision is low, it means there are many false positives.
# Recall (Sensitivity): Shows how many of the values we should have predicted as positive.
# --If the recall is low, it means there are many false negatives. It should be as high as possible.
# F1 score: It is difficult to compare two models in case of low precision and high recall or vice versa.
# --Making it comparable F1 score helps measure precision and recall simultaneously.
# --Shows the harmonic mean of Precision and Sensitivity values.
# Let's examine these metrics for our model.
# The classification report shows a class-by-class representation of the main classification criteria.

print(classification_report(y_test, y_pred))


#### Let's examine the Precision measure for each class. ###

# Shows how many of the predictions made for the 0 (non fraud) class are correct.
# --Looking at the confusion matrix, 56870 + 31 = 56901 non fraud class predictions were made and 56870 of them were guessed correctly.
# --Precision value for class 0 is 1 (56870 / 56901)

Returns how many of the predictions made for class #1 (fraud) were correct.
# --Looking at the confusion matrix, 5 + 56 = 61 fraud class predictions were made and 56 of them were guessed correctly.
# --Precision value for class 0 is 0.92 (56 / 61)

### Let's examine the recall measure for each class.###

# It shows how many of the values we should have guessed for the 0 (non fraud) class we guessed correctly.
# --56870 + 5 = We have 56875 observations belonging to the non-fraud class and 56870 of them were predicted correctly.
# --The recall value for class 0 is 56870 / 56875 = 1.

# It shows how many of the values we should have predicted for class 1 (fraud) we guessed correctly.
# --31 + 56 = We have 87 fraud class observations and 56 of them were predicted correctly.
# --The recall value for class 1 is 56 / 87 = 0.64.

# When we look at the recall values, we can easily see the failure to predict 1 class.
# F1-score represents the harmonic mean of recall and precision values.
# Support represents the number of actual values of the classes. It can show the structural weaknesses of the measurements, that is, we can say that the imbalance in the number of observations between classes affects the measurements.

### ROC Curve ###
# ROC curve is a graph showing the performance of a classification model across all classification thresholds. This curve plots two parameters:
# True Positive Rate : Recall
# False Positive Rate: Failure to detect Fraud
# It is the curve of true positive rate and false positive rate at different classification thresholds.
# --starts at (0,0) and ends at (1,1). A good model produces a curve that goes from 0 to 1 quickly.

## AUC (Area under the ROC curve) ##
# Summarizes the ROC curve into a single number. It measures the entire two-dimensional area under the entire ROC curve from (0,0) to (1,1).
# --The best value is 1, the worst value is 0.5.

# Auc Roc Curve
def generate_auc_roc_curve(clf, X_test):
     y_pred_proba = clf.predict_proba(X_test)[:, 1]
     fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
     auc = roc_auc_score(y_test, y_pred_proba)
     plt.plot(fpr,tpr)
     plt.show()
     pass

generate_auc_roc_curve(model, X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC ROC Curve with Area Under the curve = %.3f"%auc)

# AUC ROC Curve with Area Under the curve = 0.961
# Note: The area under the ROC curve (AUC) evaluates the overall classification performance.
# --AUC does not reflect minority class well as it does not give more importance to one class over another.
# Let's apply various methods to the data set to eliminate imbalance.
# NOTE: Methods must be applied to the training set. If applied to the test set, correct evaluation cannot be made.

### Resampling ###

# Resampling is making the data set more balanced by adding new examples to the minority class or removing examples from the majority class.

### Oversampling ###
# Balances the dataset by copying instances belonging to the minority class.

# Random Oversampling #
# Balancing the data set by adding randomly selected samples from the minority class.
# This technique can be used if your data set is small.
# May cause overfitting.
# RandomOverSampler method takes the sampling_strategy argument,
# -- When sampling_stratefy='minority' is called, it increases the number of the minority class to equal the number of the majority class.
# We can enter a float value into this argument. For example, let's say the number of our minority class is 1000 and the number of our majority class is 100. If we say sampling_stratefy = 0.5, the number of minority classes will be added to 500.
# number of classes in the training set before random oversampling
y_train.value_counts()

# Application of RandomOver Sampling (Applied to the training set)
from imbleearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_randomover, y_randomover = oversample.fit_resample(X_train, y_train)

# Number of classes of the training set after random oversampling
y_randomover.value_counts()

# Training the model and success rate
model.fit(X_randomover, y_randomover)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f%%" % (accuracy))
# Accuracy: 0.977%

plot_confusion_matrix(confusion_matrix(y_test, y_pred=y_pred), classes=['Non Fraud','Fraud'],
                       title='Confusion matrix')

#classification report
print(classification_report(y_test, y_pred))

# After applying Random Oversampling, the accuracy value of the trained model is 0.97, a decrease is observed.
# --When looking at the Confusion Matrix and Classification report, the false rate of predicted fraud classes seems to be high,
# --this has reduced the precision value of 1 class. But there is also an increase in the recall value of class 1, and the rate of the model correctly predicting the fraud class has increased.
# --The accuracy of predicting the Non fraud class has decreased compared to the first model
# --but the increase in the correct prediction of the fraud class is a big factor in our choosing the model created after randomoversampling.

### SMOTE Oversampling ###

# Creating synthetic instances from minority classes to prevent overfitting.

# First a random sample from the minority class is selected.
# Then k of the nearest neighbors are found for this example.
# One of the k nearest neighbors is randomly selected and a synthetic sample is created by combining it with a randomly selected sample from the minority class and creating a line segment in the feature space.

# Number of classes in the training set before Smote
y_train.value_counts()

# Application of Smote (Applied to the training set)
from imbleearn.over_sampling import SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)

# Number of classes of the training set after Smote
y_smote.value_counts()

# Training the model and success rate
model.fit(X_smote, y_smote)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f%%" % (accuracy))
# Accuracy: 0.975%

plot_confusion_matrix(confusion_matrix(y_test, y_pred=y_pred), classes=['Non Fraud','Fraud'],
                       title='Confusion matrix')

# Classification report
print(classification_report(y_test, y_pred))

###Undersampling ###

# It is a technique of balancing the data set by removing samples belonging to the majority class.
# Can be used when having a large data set.
# Since the data set we have is not large, efficient results will not be obtained.
# But let's explain the methods and show how some of them can be applied.

# Random Undersampling:

# The extracted samples are randomly selected.
# You can use this technique if you have a large data set.
# Information loss may occur due to random selection.

# Number of classes in the training set before random undersampling
y_train.value_counts()

from imbleearn.under_sampling import RandomUnderSampler

# Transform the dataset
ranUnSample = RandomUnderSampler()
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X_train, y_train)
# After random undersampling
y_ranUnSample.value_counts()

### NearMiss Undersampling ###

# Prevents information loss.
# Based on KNN algorithm.
# The distance between the samples belonging to the majority class and the samples belonging to the minority class is calculated.
# Samples with shorter distances than the specified k value are preserved.

### Undersampling (Tomek links) ###

# By removing instances of the majority class between the two closest examples of different classes, the gap between two classes is increased.

### Undersampling (Cluster Centroids) ###

# It is the removal of unimportant samples from the data set. Whether the sample is important or unimportant is determined by clustering.
# More balanced data sets can be created by combining Undersampling and Oversampling techniques.

# Other Methods
# Collect more data,
# Creating a model that can learn equally from minority and majority classes by using the "class_weight" parameter in classification models,
# Looking at the performances of other models, not just a single model,
# A different approach can be applied and methods such as Anomaly detection or Change detection can be used to deal with the imbalanced data set.

# Which method works best depends on the data set we have.
# We can say that by trying the methods and choosing the one that best suits the data set, it provides the best result.
