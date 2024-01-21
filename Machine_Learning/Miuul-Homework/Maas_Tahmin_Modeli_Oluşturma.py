#################
# Salary Prediction With Machine Learning
#################

####################
# Business Problem
####################
# Develop a machine learning model for salary predictions of baseball players whose salary information and career statistics for 1986 are shared.
#######################
# Dataset Story
#######################
# This dataset was originally retrieved from the StatLib library at Carnegie Mellon University.
# Dataset is part of the data used in the 1988ASA Graphics Section Poster Session.
# Salary data originally from Sports Illustrated, April 20, 1987. 1986 and career statistics obtained from the 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, NewYork.

# Variables

# 20 Variables 322 Observations 21 KB
# AtBat Number of hits hit with a baseball bat during the 1986-1987 season
# Hits Number of hits in the 1986-1987 season
# HmRun Most valuable hits in the 1986-1987 season
# Runs scored for his team in the 1986-1987 season
# RBI Number of runs a batsman scores when batting
# Walks Number of mistakes made by the opposing player
# Years Player's playing time in the major league (years)
# CAtBat Number of times a player has hit the ball in his career
# CHits Number of hits made by the player throughout his career
# CHmRun Player's most valuable point in his career
# CRuns Number of points scored by a player for his team during his career
# CRBI Number of players the player has made runs in his career
# CWalks Number of mistakes a player has made against an opposing player throughout his career
# League A factor with A and N levels indicating the league in which the player played until the end of the season
#Division A factor with E and W levels indicating the position the player was playing at the end of 1986
# PutOuts Helping your teammate in-game
# Assits Number of assists made by the player in the 1986-1987 season
# Errors Number of player errors in the 1986-1987 season
# Salary Player's salary in the 1986-1987 season (over thousands)
#NewLeague A factor with A and N levels indicating the player's league at the start of the 1987 season

#############################
# Project Tasks
#############################
# Required library and functions

import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, cross_validate, GridSearchCV

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 20)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

####################### ##########################
# Advanced Functional Exploratory Data Analysis (ADVANCED FUNCTIONAL EDA)
####################### ##########################

#1-Overall Picture
# 2-Analysis of Categorical Variables
# 3-Analysis of Numerical Variables
# 4-Analysis of Target Variable
#5-Analysis of Correlation

#############################
#1-Overall Picture
#############################

df= pd.read_csv("Machine_Learning/datasets/hitters.csv")

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

df.head()
df.info()


def grab_col_names(dataframe, cat_th=10, car_th=20):
# We determined the threshold values ourselves by checking the acceptance above these ranges
# Important information###
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

# num_cols
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


#############################
# 2-Analysis of Categorical Variables
#############################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################################"
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.show()

    for col in cat_cols:
        cat_summary(df, col, plot=True)

        #############################
        # 3-Analysis of Numerical Variables
        #############################


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


#############################
# 4-Analysis of Target Variable
#############################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


#############################
# 5-Analysis of Correlation
#############################
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df, plot=True)


#############################
# Advanced Functional Exploratory Data Analysis (Advanced Functional EDA)
#############################

# 1.Outliers
# 2.Missing Values
# 3.Feature Extraction
# 4.Encoding(Label Encoding, One-Hot Encoding, Rare Encoding)
# 5.Feature Scaling

#############################
# 1.Outliers
#############################

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


#############################
# 2.Missing Values
#############################

def missing_values_table(dataframe, na_name=False):
     na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
     n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
     ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
     missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
     print(missing_df, end="\n")
     if na_name:
         return na_columns

missing_values_table(df)

df.dropna(inplace=True)

#############################
# 3.Feature Extraction
#############################

new_num_cols=[col for col in num_cols if col!="Salary"]
df[new_num_cols]=df[new_num_cols]+0.0000000001

# AtBat Number of hits hit with a baseball bat during the 1986-1987 season
# Hits Number of hits in the 1986-1987 season
# HmRun Most valuable hits in the 1986-1987 season
# Runs scored for his team in the 1986-1987 season
# RBI Number of runs a batsman scores when batting
# Walks Number of mistakes made by the opposing player
# Years Player's playing time in the major league (years)
# CAtBat Number of times a player has hit the ball in his career
# CHits Number of hits made by the player throughout his career
# CHmRun Player's most valuable point in his career
# CRuns Number of points scored by a player for his team during his career
# CRBI Number of players the player has made runs in his career
# CWalks Number of mistakes a player has made against an opposing player throughout his career
# League A factor with A and N levels indicating the league in which the player played until the end of the season
#Division A factor with E and W levels indicating the position the player was playing at the end of 1986
# PutOuts Helping your teammate in-game
# Assits Number of assists made by the player in the 1986-1987 season
# Errors Number of player errors in the 1986-1987 season
# Salary Player's salary in the 1986-1987 season (over thousands)
#NewLeague A factor with A and N levels indicating the player's league at the start of the 1987 season

df["NEW_Hits"]= df["Hits"] / df["CHits"]
df["NEW_RBI"]= df["RBI"] / df["CRBI"]
df["NEW_Walks"]= df["Walks"] / df["CWalks"]
df["NEW_PutOuts"]= df["PutOuts"] * df["Years"]
df["Hits_Success"]= (df["Hits"] / df["AtBat"])*100
df["NEW_CRBI*CATBAT"]= df["CRBI"] * df["CAtBat"]
df["NEW_RBI"]= df["RBI"] / df["CRBI"]
df["NEW_Chits"]= df["CHits"] / df["Years"]
df["NEW_CHmRun"]= df["CHmRun"] * df["Years"]
df["NEW_CRuns"]= df["CRuns"] / df["Years"]
df["NEW_Chits"]= df["CHits"] * df["Years"]
df["NEW_RW"]= df["RBI"] * df["Walks"]
df["NEW_RBWALK"]= df["RBI"] / df["Walks"]
df["NEW_CH_CB"]=df["CHits"] / df["CAtBat"]
df["NEW_CHm_CAT"]=df["CHmRun"] / df["CAtBat"]
df["NEW_Diff_Atbat"]=df["AtBat"] - (df["CAtBat"] / df["Years"])
df["NEW_Diff_Hits"]=df["Hits"] - (df["CHits"] / df["Years"])
df["NEW_Diff_HmRun"]=df["HmRun"] - (df["CHmRun"] / df["Years"])
df["NEW_Diff_Runs"]=df["Runs"] - (df["CRuns"] / df["Years"])
df["NEW_Diff_RBI"]=df["RBI"] - (df["CRBI"] / df["Years"])
df["NEW_Diff_Walks"]=df["Walks"] - (df["CWalks"] / df["Years"])

#############################
#4.One-Hot Encoding
#############################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
     return dataframe

df= one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

#############################
# 5.Feature Scaling
#############################

cat_cols, num_cols, cat_but_car= grab_col_names(df)
# We apply standardization
num_cols=[col for col in num_cols if col not in["Salary"]]
scaler=StandardScaler()
df[num_cols]=scaler.fit_transform(df[num_cols])
df.head()

#############################
# Base Models (We Build the Model)
#############################

y= df["Salary"]
x= df.drop(["Salary"], axis=1)
models=[("LR",LinearRegression()),
         ("Ridge",Ridge()),
         ("Lasso",Lasso()),
         ("ElasticNet",ElasticNet()),
         ("KNN",KNeighborsRegressor()),
         ("CART",DecisionTreeRegressor()),
         ("RF",RandomForestRegressor()),
         ("SVR",SVR()),
         ("GBM",GradientBoostingRegressor()),
         ("XGBoost",XGBRegressor(objective="reg:squarederror")),
         ("LightGBM",LGBMRegressor()),
         ("CatBoost",CatBoostRegressor(verbose=False))]

for name, regressor in models:
     rmse= np.mean(np.sqrt(-cross_val_score(regressor,x,y, cv=10, scoring="neg_mean_squared_error")))
     print(f"RMSE: {round(rmse, 4)} ({name})")

# rf and gbm seem successful


#############################
# Random Forests
#############################

rf_model= RandomForestRegressor(random_state=17)
rf_params={"max_depth":[5,8,15, None],
            "max_features":[5,7,"auto"],
            "min_samples_split":[8,15, 20],
            "n_estimators":[200, 500]}

rf_best_grid= GridSearchCV(rf_model, rf_params, cv=5, n_jobs=1, verbose=True).fit(x,y)
rf_final =rf_model.set_params(**rf_best_grid.best_params_,random_state=17).fit(x,y)
rmse= np.mean(np.sqrt(-cross_val_score(rf_final, x, y, cv=10, scoring="neg_mean_squared_error")))
rmse


#############################
# GBM Model
#############################

gbm_model= GradientBoostingRegressor(random_state=17)

gbm_params= {"learning_rate":[0.01, 0.1],
              "max_depth":[3,8],
              "n_estimators":[500, 1000],
              "subsample":[1,0.5, 0.7]}

gbm_best_grid= GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=1, verbose=True).fit(x,y)
gbm_final =gbm_model.set_params(**gbm_best_grid.best_params_,random_state=17,).fit(x,y)
rmse= np.mean(np.sqrt(-cross_val_score(gbm_final, x, y, cv=10, scoring="neg_mean_squared_error")))
rmse

#############################
#LightGBM Model
#############################

lgbm_model= LGBMRegressor(random_state=17)

lgbm_params= {"learning_rate":[0.01, 0.1],
              "n_estimators":[300, 500],
              "colsample_bytree":[0.7, 1]}

lgbm_best_grid= GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=1, verbose=True).fit(x,y)
lgbm_final =lgbm_model.set_params(**lgbm_best_grid.best_params_,random_state=17,).fit(x,y)
rmse= np.mean(np.sqrt(-cross_val_score(lgbm_final, x, y, cv=10, scoring="neg_mean_squared_error")))
rmse

#############################
#CatBoost
#############################

catboost_model= CatBoostRegressor(random_state=17, verbose=False)

catboost_params={"iterations":[288, 500],
                  "learning_rate":[0.01, 0.1],
                  "depth":[3, 6]}

catboost_best_grid= GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=1, verbose=True).fit(x,y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_,random_state=17,).fit(x,y)
rmse= np.mean(np.sqrt(-cross_val_score(catboost_final, x, y, cv=10, scoring="neg_mean_squared_error")))
rmse

#######################
# Automated Hyperparameter Optimization
#######################

rf_params={"max_depth":[5,8,15, None],
            "max_features":[5,7,"auto"],
            "min_samples_split":[8,15, 20],
            "n_estimators":[200, 500]}

gbm_params= {"learning_rate":[0.01, 0.1],
              "max_depth":[3,8],
              "n_estimators":[500, 1000],
              "subsample":[1,0.5, 0.7]}

lightgbm_params= {"learning_rate":[0.01, 0.1],
              "n_estimators":[300, 500],
              "colsample_bytree":[0.7, 1]}

catboost_params={"iterations":[288, 500],
                  "learning_rate":[0.01, 0.1],
                  "depth":[3, 6]}

regressors=[("RF",RandomForestRegressor(), rf_params),
            ("GBM", GradientBoostingRegressor(),gbm_params),
            ("LightGBM", LGBMRegressor(),lightgbm_params),
            ("CatBoost", CatBoostRegressor(),catboost_params)]

best_models={}

for name,regressor,params in regressors:
     print(f"################# {name} #########")
     rmse = np.mean(np.sqrt(-cross_val_score(regressor, x, y, cv=10, scoring="neg_mean_squared_error")))
     print(f"RMSE: {round(rmse, 4)} ({name})")

     gs_best= GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(x,y)

     final_model= regressor.set_params(**gs_best.best_params_)
     rmse=np.mean(np.sqrt(-cross_val_score(final_model, x, y, cv=10, scoring="neg_mean_squared_error")))
     print(f"RMSE (After): {round(rmse, 4)} ({name})")

     print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

     best_models[name] = final_model

#######################
#FeatureImportance
#######################

def plot_importance(model, features, num=len(x), save=False):
     feature_imp= pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
     plt.figure(figsize=(10, 10))
     sns.set(font_scale=1)
     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num])

     plt.title("Features")
     plt.tight_layout()
     plt.show()
     if save:
         plt.savefig("importances.png")

plot_importance(rf_final,x)
plot_importance(gbm_final,x)
plot_importance(lgbm_final,x)
plot_importance(catboost_final,x)

#######################
# Analyzing Model Complexity with Learning Curves
#######################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
     train_score, test_score =validation_curve(
         model,

     mean_train_score= np.mean(train_score, axis=1)
     mean_test_score= np.mean(test_score,axis=1)

     plt.plot(param_range,mean_train_score,label="Training Score", color="b")
     plt.plot(param_range, mean_test_score, label="Validation Score", color="g")

     plt.title(f"Validation Curve for{type(model).__name__}")
     plt.xlabel(f"Number of {param_name}")
     plt.ylabel(f"{scoring}")
     plt.tight_layout()
     plt.legend(loc="best")
     plt.show()

rf_val_params=[["max_depth",[5,8,15,29,30, None]],
                    ["max_features",[3,5,7,"auto"]],
                    ["min_samples_split",[2,5,6,15,29]],
                    ["n_estimators",[10,50,100,200,500]]

rf_model=RandomForestRegressor(random_state=17)

for i in range(len(rf_val_params)):
     val_curve_params(rf_model, x, y, rf_val_params[i][0], rf_val_params[i][1], scoring="neg_mean_absolute_error")

# rf_val_params[0][1]
