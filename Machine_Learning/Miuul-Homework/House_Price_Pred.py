##########################
#House Price Prediction Model
##########################

#######################
#BusinessProblem
#######################
# Using the data set containing the features and house prices of each house,
# --a machine learning project regarding the prices of different types of houses is wanted to be carried out.
#######################

####################
# Dataset Story
####################

# There are 79 explanatory variables in this data set of residential homes in Ames, Iowa.
# Since the data set belongs to a Kaggle competition, there are two different csv files: train and test.
# House prices are left blank in the test data set, and you are expected to guess these values.
# Data Set Story: 1460 Total Observations, 38 Numerical Variables, 43 Categorical Variables.

#################
# Duty
#################
# Develop a machine learning model that predicts house prices with minimum error based on the data set we have.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Combining train and test sets.
train = pd.read_csv("Machine_Learning/datasets/Htrain.csv")
test = pd.read_csv("Machine_Learning/datasets/Htest.csv")
df = train.append(test).reset_index()
df.head()


#EDA


cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Number of Categorical Variables: ', len(cat_cols))


def cat_summary(data, categorical_cols, target, number_of_classes=10):
     var_count = 0
     vars_more_classes = []
     for var in categorical_cols:
         if len(df[var].value_counts()) <= number_of_classes: # select by number of classes
             print(pd.DataFrame({var: data[var].value_counts(),
                                 "Ratio": 100 * data[var].value_counts() / len(data),
                                 "TARGET_MEDIAN": data.groupby(var)[target].median()}), end="\n\n\n")
             var_count += 1
         else:
             vars_more_classes.append(data[var].name)
     print('%d categorical variables have been described' % var_count, end="\n\n")
     print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
     print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
     print(vars_more_classes)



cat_summary(df, cat_cols, "SalePrice")

# Variables with more than 10 classes:
for col in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
     print(df[col].value_counts())


# NUMERICAL VARIABLE ANALYSIS
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "Id"]
print('Number of numeric variables: ', len(num_cols))


def hist_for_nums(data, numeric_cols):
     col_counter = 0
     data = data.copy()
     for col in numeric_cols:
         data[col].hist(bins=20)
         plt.xlabel(col)
         plt.title(col)
         plt.show()
         col_counter += 1
     print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)

# TARGET ANALYSIS
df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

# correlations of target and arguments

def find_correlation(dataframe, corr_limit=0.60):
     high_correlations = []
     low_correlations = []
     for col in num_cols:
         if col == "SalePrice":
             pass

         else:
             correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
             print(col, correlation)
             if abs(correlation) > corr_limit:
                 high_correlations.append(col + ": " + str(correlation))
             else:
                 low_correlations.append(col + ": " + str(correlation))
     return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df)

3. DATA PREPROCESSING & FEATURE ENGINEERING


## Editing year variables
def elapsed_years(df, var):

     df[var] = df['YrSold'] - df[var]
     return df

for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
     df = elapsed_years(df, var)



# We omit the year the house was sold variable because it has no meaning on its own.
df.drop('YrSold', axis=1, inplace=True)
num_cols.remove('YrSold') # we must remove from numeric variables



# RARE ANALYZER
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in df.columns if len(df[col].value_counts()) <= 20
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")


rare_analyser(df, "SalePrice", 0.01)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df


df = rare_encoder(df, 0.01)
rare_analyser(df, "SalePrice", 0.01)



drop_list = ["Street", "Utilities", "LandSlope", "PoolQC", "MiscFeature"]
cat_cols = [col for col in df.columns if df[col].dtypes == 'O'
            and col not in drop_list]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", 0.01)


# LABEL ENCODING & ONE-HOT ENCODING
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, cat_cols)
cat_summary(df, new_cols_ohe, "SalePrice")


# MISSING_VALUES
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


missing_values_table(df)
df = df.apply(lambda x: x.fillna(x.median()), axis=0)
missing_values_table(df)


# OUTLIERS
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


has_outliers(df, num_cols)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

has_outliers(df, num_cols)

# STANDARDIZATION

df.head()
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) < 20]
cols_need_scale = [col for col in df.columns if col not in new_cols_ohe
                    and col not in "Id"
                    and col not in "SalePrice"
                    and col not in like_num]

df[cols_need_scale].head()
df[cols_need_scale].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T
hist_for_nums(df, cols_need_scale)


def robust_scaler(variable):
     var_median = variable.median()
     quartile1 = variable.quantile(0.25)
     quartile3 = variable.quantile(0.75)
     interquantile_range = quartile3 - quartile1
     if int(interquantile_range) == 0:
         quartile1 = variable.quantile(0.05)
         quartile3 = variable.quantile(0.95)
         interquantile_range = quartile3 - quartile1
         z = (variable - var_median) / interquantile_range
         return round(z, 3)
     else:
         z = (variable - var_median) / interquantile_range
     return round(z, 3)


for col in cols_need_scale:
     df[col] = robust_scaler(df[col])


df[cols_need_scale].head()
df[cols_need_scale].describe().T
hist_for_nums(df, cols_need_scale)


# last check
missing_values_table(df)
has_outliers(df, num_cols)


# 4. MODELING

Htrain_df = df[df['SalePrice'].notnull()]
Htest_df = df[df['SalePrice'].isnull()]

# Treat train_df as our entire data set and perform the modeling process as we discussed in the lesson.
X = Htrain_df.drop('SalePrice', axis=1)
y = Htrain_df[["SalePrice"]]


X_Htrain, X_Htest, y_Htrain, y_Htest = train_test_split(X, y, test_size=0.20, random_state=46)

# You can run and try the TODO scaler here.

models = [('LinearRegression', LinearRegression()),
           ('Ridge', Ridge()),
           ('Lasso', Lasso()),
           ('ElasticNet', ElasticNet())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
     model.fit(X_Htrain, y_Htrain)
     y_pred = model.predict(X_Htest)
     result = np.sqrt(mean_squared_error(y_Htest, y_pred))
     results.append(result)
     names.append(name)
     msg = "%s: %f" % (name, result)
     print(msg)

df["SalePrice"].mean()

