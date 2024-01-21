#######################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
#######################

import warnings
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("Machine_Learning/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

#######################
# Random Forests
#######################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

# Our error scores before hyperparameter
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# We need to search with grid search and look for the best value.
rf_params = {"max_depth": [5, 8, None],
              "max_features": [3, 5, 7, "auto"],
              "min_samples_split": [2, 5, 8, 15, 20],
              "n_estimators": [100, 200, 500]}

# Normally, in practice, we expect the error obtained after hyperparameter optimization to be low.
# One of the reasons why it is not low may be related to randomness.
# It may happen that you do not have the default arguments in your search set.

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


def plot_importance(model, features, num=len(X), save=False):
     feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
     plt.figure(figsize=(10, 10))
     sns.set(font_scale=1)
     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
     plt.title('Features')
     plt.tight_layout()
     plt.show()
     if save:
         plt.savefig('importances.png')

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
     train_score, test_score = validation_curve(
         model,

     mean_train_score = np.mean(train_score, axis=1)
     mean_test_score = np.mean(test_score, axis=1)

     plt.plot(param_range, mean_train_score,
              label="Training Score", color='b')

     plt.plot(param_range, mean_test_score,
              label="Validation Score", color='g')

     plt.title(f"Validation Curve for {type(model).__name__}")
     plt.xlabel(f"Number of {param_name}")
     plt.ylabel(f"{scoring}")
     plt.tight_layout()
     plt.legend(loc='best')
     plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")


#######################
# GBM
#######################

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()
# n_estimators => 100 is the number of estimators. It is the number of optimization. We boosted it.


cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.7591715474068416
cv_results['test_f1'].mean()
#0.634
cv_results['test_roc_auc'].mean()
#0.82548


# Set of hyperparameters to search for gbm
gbm_params = {"learning_rate": [0.01, 0.1],
# The lower the learning rate, the longer the train time is, but if it is smaller, more successful predictions are obtained.
               "max_depth": [3, 8, 10],
               "n_estimators": [100, 500, 1000],
               "subsample": [1, 0.5, 0.7]
               # We entered a simple range by entering the observation rate as 50%-70%. We want the transactions to take a short time.
             }

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)


# Let's evaluate the errors of the final model
cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Our hyperparameter optimization was successful again because we kept the default values.
# He contributed to us by making model adjustments without touching the data.

#######################
#XGBoost
#######################

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()

cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.75265
cv_results['test_f1'].mean()
#0.631
cv_results['test_roc_auc'].mean()
#0.7987

# It is preferable to stay at a certain depth and generalize rather than increasing the depth and causing memorization.
# Parameters that we think are more efficient " "learning_rate","max_depth","n_estimators" " are preferred.
xgboost_params = {"learning_rate": [0.1, 0.01],
                   "max_depth": [5, 8],
                   "n_estimators": [100, 500, 1000],
                   "colsample_bytree": [0.7, 1]}
# colsample-bytree-->Parameter related to the number of observations taken from the variables.
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



#######################
#LightGBM
#######################

# XGBoost is another type of GBM developed to increase training time performance.
# It is faster with the Leaf-wise growth strategy instead of the Level-wise growth strategy.
# It is successful due to the difference in the splitting method. In the case of the XGBoost split process, it follows the growth method according to the level
# Lightgbm focuses on leaves. Considering the division processes in tree structures;
# While XGBoost performs a comprehensive initial search, LightGbm performs an in-depth initial search.

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [100, 300, 500, 1000],
                "colsample_bytree": [0.5, 0.7, 1]}


lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# ------------------------------------------------- -------------------------------------------------
# Hyperparameter with new values
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
                "n_estimators": [200, 300, 350, 400],
                "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# Hyperparameter optimization for n_estimators only.
# n_estimators (number of predictions) is the most important hyperparameter. Number of predictions = number of iterations. It is the number of boosting.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#######################
#CatBoost
#######################
# It is another fast and successful GBM variant that can automatically deal with categorical variables.
# Let's say there are 15 classes, either an algorithm that is sensitive to the 15 classes is needed to understand whether they are categorical or
# One needs to pass it through a hot encoder, the other one must pass it sequentially through a label encoder, like 1-2-3-10-15..
# If it is not sequential, 15 new features will be generated when you pass them through one hot encoder one by one.
# As a result of the categorical variable, it is an important fish to keep in mind.
# It is fast, scalable, and has GPU support.
# It may be more successful in some scenarios.

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


catboost_params = {"iterations": [200, 500],
                    "learning_rate": [0.01, 0.1],
                    "depth": [3, 6]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


#######################
#FeatureImportance
#######################

def plot_importance(model, features, num=len(X), save=False):
     feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
     plt.figure(figsize=(10, 10))
     sns.set(font_scale=1)
     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
     plt.title('Features')
     plt.tight_layout()
     plt.show()
     if save:
         plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


####################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
####################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                     "max_features": [3, 5, 7, "auto", "sqrt"],
                     "min_samples_split": np.random.randint(2, 50, 20),
                     "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                                param_distributions=rf_random_params,
                                n_iter=100, # number of parameters to try
                                cv=3,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1)

rf_random.fit(X, y)


rf_random.best_params_


rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


####################
# Analyzing Model Complexity with Learning Curves (BONUS)
####################

#Functionalization

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
     train_score, test_score = validation_curve(
         model,

     mean_train_score = np.mean(train_score, axis=1)
     mean_test_score = np.mean(test_score, axis=1)

     plt.plot(param_range, mean_train_score,
              label="Training Score", color='b')

     plt.plot(param_range, mean_test_score,
              label="Validation Score", color='g')

     plt.title(f"Validation Curve for {type(model).__name__}")
     plt.xlabel(f"Number of {param_name}")
     plt.ylabel(f"{scoring}")
     plt.tight_layout()
     plt.legend(loc='best')
     plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                  ["max_features", [3, 5, 7, "auto"]],
                  ["min_samples_split", [2, 5, 8, 15, 20]],
                  ["n_estimators", [10, 50, 100, 200, 500]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
     val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]

# According to the maximum depth, the train set continued its success and the test set started to fall. The knowledge of where the maximum depth should be according to the values we obtained.
# There is no serious change in the Max Futures AUC value, but there are various changes in the validation score, nothing significant
# Min samples split train set reacted immediately. As the number of samples to be considered in the sections increased, the AUC value of the train set decreased, but the validation of the test set started to increase.
# -- In other words, when the generalizability of the function and the model is examined according to the success in the test set, it seems that the min increases as the samples split increases.
# As the number of n_estimators increases, an increase in the validation scores seems to be observed in the test scores.
# Therefore, we have already made our selection according to the errors in the scenario of observing all possible combinations of all possible hyperparameters simultaneously by performing hyperparameter optimization.
# We obtain additional information by evaluating how far we can reach these choices and also through learning curves.

#Model
#######################
# Visualizing the Decision Tree
#######################


def tree_graph(model, col_names, file_name):
     tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
     graph = pydotplus.graph_from_dot_data(tree_str)
     graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

cart_final.get_params()








