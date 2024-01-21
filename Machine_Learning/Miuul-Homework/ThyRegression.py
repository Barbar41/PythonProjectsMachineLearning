####################
# Passenger Number Estimation with Machine Learning
####################

####################
# BUSINESS PROBLEM
####################
# Estimating the number of passengers with seasonal reservations on Turkish Airlines flights
# Requests to build a machine learning regression model for

#######################
# Dataset Story
#######################
# This data set of Turkish Airlines includes 1-way (2-leg) seasonal flight information on origin & destination basis.
#17 Variant 18.1173 Observation 12 KB

# CARRIER It is the binary code of the carrier company. The carrier company is the same for both legs.
# AIRCRAFT_TYPE Aircraft type information.
# OND_SELL_CLASS Longest legin sales class.
# LEG1_SELL_CLASS is the sales class of the 1st leg.
# OND_CABIN_CLASS It is the longest leg in cabin class.
# LEG1_CABIN_CLASS is the cabin class of the 1st leg.
# HUB contains the IATA codes of the airport where the transfer is made from the 1st leg flight to the 2nd leg flight.
# DETUR_FACTOR It is a ratio that expresses how much a passenger extends his route if he flies connecting instead of flying directly to the destination he wants to fly to.
# CONNECTION_TIME expresses the expected time in minutes for the transfer from the 1st leg flight to the 2nd leg flight.
# PSGR_COUNT Total number of passengers with reservations.
# LEG1_DEP_FULL 1. Leg's departure date and time.
# LEG1_ARR_FULL is the arrival date and time of the 1st Leg.
# LEG2_DEP_FULL is the departure date and time of the 2nd Leg.
# LEG2_ARR_FULL is the arrival date and time of the 2nd Leg.
# LEG1_DURATION 1. Leg's flight duration.
# LEG2_DURATION Flight duration of the 2nd Leg.
# FLIGHT_DURATION The time from the take-off time of Leg 1 to the landing time of Leg 2.
----------------------------------------------
# Leg: Represents 1 flight. A 2 leg flight means a journey with 2 flights with 1 connection.
# Example: If there is a London connection on the Istanbul - San Francisco flight; Istanbul - London is the 1st leg, London - San Francisco is the 2nd leg.

#############################
# Project Tasks
#############################
# Required library and functions

import warnings
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
#from helpers.data_prep import *
#from helpers.eda import *
import calendar
from sklearn.model_selection import *
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Step 1: Read the thyTrain.csv file.

df = pd.read_csv("Machine_Learning/datasets/thyTrain.csv")
df.head()
df.shape

# Step 2: Bring the standard deviation and average information of the Target variable (PSGR_COUNT).

df["PSGR_COUNT"].agg(["std", "mean"])

# Step 3: Sort the Target variable from largest to smallest and bring the first 10 rows of the data set.

df.sort_values("PSGR_COUNT", ascending=False).head(10)

# Step 4: Get the min and max values of LEG1_DEP_FULL, LEG1_ARR_FULL, LEG2_DEP_FULL, LEG2_ARR_FULL variables.

df[["LEG1_DEP_FULL", "LEG1_ARR_FULL", "LEG2_DEP_FULL", "LEG2_ARR_FULL"]].agg(["min", "max"])

# Step 5: Print the number of null values in each variable and fill in the observations with null values with the mode of that variable.

df.isnull().sum()

df["AIRCRAFT_TYPE"].fillna(df["AIRCRAFT_TYPE"].mode()[0], inplace=True)

# Step 6: Create the variables mentioned below.

# Variable Name Variable Description
# LEG1_DEP_MONTH Month information of LEG1_DEP_FULL variable
# LEG1_DEP_HOUR Time information of the LEG1_DEP_FULL variable
# LEG1_ARR_MONTH Month information of LEG1_ARR_FULL variable
# LEG1_ARR_HOUR Time information of the LEG1_ARR_FULL variable
# LEG2_DEP_MONTH Month information of LEG2_DEP_FULL variable
# LEG2_DEP_HOUR Time information of the LEG2_DEP_FULL variable
# LEG2_ARR_MONTH Month information of LEG2_ARR_FULL variable
# LEG2_ARR_HOUR Time information of the LEG2_ARR_FULL variable
# LEG1_DEP_DAY Day of the week information of the LEG1_DEP_FULL variable (Monday, Tuesday,...)
# LEG1_ARR_DAY Day of the week information of the LEG1_ARR_FULL variable (Monday, Tuesday,...)
# LEG2_DEP_DAY Day of the week information of the LEG2_DEP_FULL variable (Monday, Tuesday,...)
# LEG2_ARR_DAY Day of the week information of the LEG2_ARR_FULL variable (Monday, Tuesday,...)
# LEG1_DURATION_MINUTES Convert the LEG1_DURATION variable to minutes. (01:50 -> 110)
# LEG2_DURATION_MINUTES Convert the LEG2_DURATION variable to minutes. (01:50 -> 110)
# FLIGHT_DURATION_MINUTES Convert the FLIGHT_DURATION variable to minutes. (01:50 -> 110)
# FLIGHT_DURATION_MINUTES_FLIGHTS Flight duration excluding waiting time at the airport for transfer
# FLIGHT_DURATION The ratio of the duration in minutes of the entire flight including the layover to the flight time excluding the layover
# LEG1_RATIO The ratio of the value in minutes of the variable LEG1_DURATION to the value in minutes of the entire flight
# LEG2_RATIO The ratio of the value in minutes of the variable LEG2_DURATION to the value in minutes of the entire flight
    # Note: You can create variables related to the day of the week with the weekday() method of the calendar library.
    # Do not forget to convert the modified variables to datetime.


df["LEG1_DEP_FULL"] = pd.to_datetime(df["LEG1_DEP_FULL"])
df["LEG1_ARR_FULL"] = pd.to_datetime(df["LEG1_ARR_FULL"])
df["LEG2_DEP_FULL"] = pd.to_datetime(df["LEG2_DEP_FULL"])
df["LEG2_ARR_FULL"] = pd.to_datetime(df["LEG2_ARR_FULL"])

df["LEG1_DEP_MONTH"] = df["LEG1_DEP_FULL"].dt.month
df["LEG1_DEP_HOUR"] = df["LEG1_DEP_FULL"].dt.hour

df["LEG1_ARR_MONTH"] = df["LEG1_ARR_FULL"].dt.month
df["LEG1_ARR_HOUR"] = df["LEG1_ARR_FULL"].dt.hour

df["LEG2_DEP_YEAR"] = df["LEG2_DEP_FULL"].dt.year
df["LEG2_DEP_MONTH"] = df["LEG2_DEP_FULL"].dt.month
df["LEG2_DEP_HOUR"] = df["LEG2_DEP_FULL"].dt.hour

df["LEG2_ARR_MONTH"] = df["LEG2_ARR_FULL"].dt.month
df["LEG2_ARR_HOUR"] = df["LEG2_ARR_FULL"].dt.hour

df["LEG1_DEP_DAY"] = df["LEG1_DEP_FULL"].apply(lambda x: calendar.day_name[x.weekday()])
df["LEG1_ARR_DAY"] = df["LEG1_ARR_FULL"].apply(lambda x: calendar.day_name[x.weekday()])
df["LEG2_DEP_DAY"] = df["LEG2_DEP_FULL"].apply(lambda x: calendar.day_name[x.weekday()])
df["LEG2_ARR_DAY"] = df["LEG2_ARR_FULL"].apply(lambda x: calendar.day_name[x.weekday()])

df["LEG1_DURATION"] = pd.to_timedelta(df["LEG1_DURATION"])
df["LEG1_DURATION_MINUTES"] = df["LEG1_DURATION"].apply(lambda x: x.seconds / 60)

df["LEG2_DURATION"] = pd.to_timedelta(df["LEG2_DURATION"])
df["LEG2_DURATION_MINUTES"] = df["LEG2_DURATION"].apply(lambda x: x.seconds / 60)

df["FLIGHT_DURATION"] = pd.to_timedelta(df["FLIGHT_DURATION"]) # timedelta
df["FLIGHT_DURATION_MINUTES"] = df["FLIGHT_DURATION"].apply(lambda x: x.seconds / 60)

df["FLIGHT_DURATION_MINUTES_FLIGHTS"] = df["LEG1_DURATION_MINUTES"] + df["LEG2_DURATION_MINUTES"] # except connection time

df["CONNECTION_RATIO"] = df["FLIGHT_DURATION_MINUTES_FLIGHTS"] / df["FLIGHT_DURATION_MINUTES"]

df["LEG1_RATIO"] = df["LEG1_DURATION_MINUTES"] / df["FLIGHT_DURATION_MINUTES"]
df["LEG2_RATIO"] = df["LEG2_DURATION_MINUTES"] / df["FLIGHT_DURATION_MINUTES"]


# Step 6.1: "LEG1_DEP_FULL", "LEG1_ARR_FULL", "LEG2_DEP_FULL", "LEG2_ARR_FULL", "LEG1_DURATION", "LEG2_DURATION", "FLIGHT_DURATION"
# Remove variables from the dataframe.

df.drop(["LEG1_DEP_FULL", "LEG1_ARR_FULL", "LEG2_DEP_FULL", "LEG2_ARR_FULL", "LEG1_DURATION", "LEG2_DURATION",
              "FLIGHT_DURATION"], axis=1, inplace=True)


# Step 7: Classify the variables using the grab_col_names function.

def grab_col_names(dataframe, cat_th=10, car_th=20):
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


# Step 7.1: Remove the PSGR_COUNT target variable from the num_cols variable using list comprehension.

num_cols = [col for col in num_cols if "PSGR_COUNT" not in col]
df.head(10)


# Step 8: Suppress outlier values.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
     quartile1 = dataframe[col_name].quantile(q1)
     quartile3 = dataframe[col_name].quantile(q3)
     interquantile_range = quartile3 - quartile1
     up_limit = quartile3 + 1.5 * interquantile_range
     low_limit = quartile1 - 1.5 * interquantile_range
     return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1=0.25 , q3=0.75):
     low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
     dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
     replace_with_thresholds(df, col)

# Step 8.1: Combine the rare classes into a single class. (rare_perc = 0.05)

def rare_encoder(dataframe, rare_perc):
     temp_df = dataframe.copy()

     rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                     and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

     for var in rare_columns:
         tmp = temp_df[var].value_counts() / len(temp_df)
         rare_labels = tmp[tmp < rare_perc].index
         temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

     return temp_df



df = rare_encoder(df, 0.05)

# Step 8.2: Apply One Hot Encoding to categorical variables.

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
     return dataframe

ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() > 2) & (col != "PSGR_COUNT")]
df = one_hot_encoder(df, ohe_cols)

# Step 8.3: Apply the Standard Scaler process

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 9: Train machine learning models using Cross Validation and return RMSE values.


y = df["PSGR_COUNT"]
X = df.drop(["PSGR_COUNT"], axis=1)


# MODEL

models = [("LR", LinearRegression()),
           ("Ridge", Ridge()),
           ("Lasso", Lasso()),
           ("ElasticNet", ElasticNet()),
           ("CART", DecisionTreeRegressor()),
           ("SVR", SVR()),
           ("GBM", GradientBoostingRegressor()),
           ("XGBoost", XGBRegressor(objective="reg:squarederror")),
           ("LightGBM", LGBMRegressor()),
           ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
     rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
     print(f"RMSE {name} :", rmse)



# Step 9.1: Apply GridSearchCV on specific algorithms and return RMSE values.

rf_params = {"max_depth": [5, 8, None],
                  "max_features": [5, 7, "auto"],
                  "min_samples_split": [8, 20], # try above 20
                  "n_estimators": [200]}

xgboost_params = {"learning_rate": [0.1],
                       "max_depth": [5],
                       "n_estimators": [100],
                       "colsample_bytree": [0.5]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                        "n_estimators": [300, 500],
                        "colsample_bytree": [0.7, 1]}

regressors = [#("RF", RandomForestRegressor(), rf_params),
               #("XGBoost", XGBRegressor(objective="reg:squarederror"), xgboost_params),
               ("LightGBM", LGBMRegressor(), lightgbm_params)]

best_models = {}

for name, regressor, params in regressors:
     print(f"########## {name} ##########")

     gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

     final_model = regressor.set_params(**gs_best.best_params_)
     rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
     print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

     print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

     best_models[name] = final_model


# Step 10: Plot the order of the features using the feature_importance function, which indicates the importance level of the variables.

#FEATURE IMPORTANCE

def plot_importance(model, features, num=len(X), save=False):

     feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
     plt.figure(figsize=(10, 10))
     sns.set(font_scale=1)
     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                          ascending=False)[0:num])
     plt.title("Features")
     plt.tight_layout()
     plt.show()
     if save:
         plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)