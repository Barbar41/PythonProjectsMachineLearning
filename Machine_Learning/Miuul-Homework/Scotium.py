#################
# Talent Hunting Classification with Machine Learning
#################

####################
# Business Problem
####################
# Predicting which class (average, highlighted) players are based on the points given to the characteristics of the football players watched by the scouts.

#######################
# Dataset Story
#######################

# The football players evaluated by the scouts according to the characteristics of the football players observed in the matches from the data set Scoutium,
# --consists of information including the features and scores scored during the match.

# Scoutium_attributes.csv
#8 Variable 10,730 Observations 527 KB
# task_response_id: The set of a scout's evaluations of all players on a team's roster in a match
# match_id : The id of the relevant match
# evaluator_id : Evaluator's id
# player_id : The id of the relevant player
# position_id : The id of the position played by the relevant player in that match
#1: Goalkeeper
#2: Center back
#3: Right back
#4: Left back
#5: Defensive midfielder
#6: Central midfielder
#7: Right wing
#8: Left wing
#9: Attacking midfielder
#10: Striker
# analysis_id : Set containing a scout's characteristic evaluations of a player in a match
# attribute_id : The id of each attribute on which players are evaluated
# attribute_value : The value (point) a scout gives to a player's attribute
# ------------------------------------------------- -------------------------------------------------- --------------------

# Scoutium_potential_labels.csv
#5 Variable 322 Observations 12 KB
# task_response_id : The set of a scout's evaluations of all players on a team's roster in a match
# match_id : The id of the relevant match
# evaluator_id : Evaluator's id
# player_id : The id of the relevant player
# potential_label : The label that indicates a scout's final decision about a player in a match. (target variable)

#######################
# Tasks
#######################
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Step1: Read the scoutium_attributes.csv and scoutium_potential_labels.csv files.

df=pd.read_csv("Machine_Learning/datasets/scoutium_attributes.csv",sep=";")
df.head()

df2=pd.read_csv("Machine_Learning/datasets/scoutium_potential_labels.csv",sep=";")
df2.head()


# Step2: Combine the csv files we have read using the merge function.
# (Perform the combination using 4 variables: "task_response_id", 'match_id', 'evaluator_id' "player_id".)

dff= pd.merge(df, df2, how="left", on=["task_response_id", "match_id", "evaluator_id", "player_id"])

# Step3: Remove the Goalkeeper (1) class in position_id from the data set.

dff= dff[dff["position_id"] !=1]

# Step4: Remove the below_average class in the potential_label from the dataset. (below_average class constitutes 1% of the entire dataset)

dff= dff[dff["potential_label"] != "below_average"]

# Step5: Create a table using the “pivot_table” function from the data set you created. One player per row in this pivot table
Manipulate to #

          # Step1: "player_id", "position_id" and "potential_label" in the index, "attribute_id" in the columns and the score given by the scouts to the players in the values
          # Create the pivot table with “attribute_value”.
          pt=pd.pivot_table(dff,values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])

#                                                          4322     4323     4324     4325     4326     4327  ..
            # player_id position_id potential_label ..
            # 1355710         7          average          50.500   50.500   34.000   50.500   45.000   45.000 ..
            # 1356362         9          average          67.000   67.000   67.000   67.000   67.000   67.000 ..
            # 1356375         3          average          67.000   67.000   67.000   67.000   67.000   67.000 ..
            #                 4          average          67.000   78.000   67.000   67.000   67.000   78.000 ..
            # 1356411         9          average          67.000   67.000   78.000   78.000   67.000   67.000 ..

           # Step2: Use the “reset_index” function to assign the indexes as variables and convert the names of the “attribute_id” columns to string.
         pt = pt.reset_index(drop=False)
         pt.head()
         pt.columns = pt.columns.map(str)

# Step6: Express the “potential_label” categories (average, highlighted) numerically using the Label Encoder function.

le = LabelEncoder()
pt["potential_label"] = le.fit_transform(pt["potential_label"])

# Step7: Assign the numeric variable columns to a list named “num_cols”.

num_cols = pt.columns[3:]

# Step8: Apply StandardScaler to scale the data in all “num_cols” variables you saved.

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

# Step9: A machine learning model that predicts the potential tags of football players with minimum error based on the data set we have
# improve. (Print roc_auc, f1, precision, recall, accuracy metrics)

y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)

models = [("LR", LogisticRegression()),
          ("KNN", KNeighborsClassifier()),
          ("SVC", SVC()),
          ("CART", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ("Adaboost", AdaBoostClassifier()),
          ("GBM", GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier()),
          ("Catboost", CatBoostClassifier(verbose=False)),
          ("LightGBM", LGBMClassifier())]

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score + " score:" + str(cvs))


# Step10: Plot the order of the features using the feature_importance function, which indicates the importance level of the variables.

# FeatureImportance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)













