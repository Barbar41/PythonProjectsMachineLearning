################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# K-Means
################################

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

# Standardization
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

# Model building, we have no dependent variables. Due to unsupervised learning management.
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

####################
# Determining the Optimum Number of Clusters
####################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
     kmeans = KMeans(n_clusters=k).fit(df)
     ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("SSE/SSR/SSD Corresponding to Different K Values")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

# To make a decision, the points where the elbowing is most severe are chosen
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

# The optimum number of clusters is 5.

####################
# Creating Final Clusters
####################
#Let's build the model
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)


kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

# Let's do the necessary assignment
clusters_kmeans = kmeans.labels_

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)

df["cluster"] = clusters_kmeans

df.head()
# Complementing 0 expressions to 1
df["cluster"] = df["cluster"] + 1

# Which states are in which cluster?

df[df["cluster"]==5]

# Let's evaluate by description
df.groupby("cluster").agg(["count","mean","median"])

# Like exporting to Excel.
df.to_csv("clusters.csv")


####################
# Hierarchical Clustering
####################
# It is to divide observations into subsets according to their similarities to each other.
# Approached as Unifier and Partitioner.
# According to K-means, we cannot intervene in the cluster formation process from the outside in K-means, there is no possibility of observation.
# The advantage of hierarchical clustering is that it allows us to define new clusters at various clustering levels with certain lines from certain points.

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# With a combinatorial method, it iteratively separates the most similar ones into clusters, then evaluates them in general, creates new clusters, and combines them to create agglomeration.

hc_average = linkage(df, "average")

# Dendrogram diagram
plt.figure(figsize=(10, 5))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(hc_average,
            leaf_font_size=10)
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Observation Units")
plt.ylabel("Distances")
dendrogram(hc_average,
            truncate_mode="lastp",
            p=10,
            show_contracted=True,
            leaf_font_size=10)
plt.show()

####################
# Determining the Number of Clusters
####################

# We keep the output with the assignment process.

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.6, color='b', linestyle='--')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()


####################
# Creating the Final Model
####################
# Let's give information about which observation unit will be in which class.

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

# We read the df from outside and added a cluster to it.

df = pd.read_csv("Machine_Learning/datasets/USArrests.csv", index_col=0)

df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

# Let's combine with K-Means
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df["kmeans_cluster_no"] = clusters_kmeans
df["cluster"] = df["cluster"] + 1



####################
# Principal Component Analysis
####################
# The main features of multivariate data are represented by fewer variables/components.
# It means reducing the variable size by risking a small amount of information loss.
# It is a dimensionality reduction approach.
# Principal component analysis reduces the independent variables of the data set into components expressed by their linear combinations.
# For this reason there is no correlation.
#
df = pd.read_csv("Machine_Learning/datasets/hitters.csv")
df.head()

# Numeric variables in the data set
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape

# Let's standardize
df = StandardScaler().fit_transform(df)

# Let's apply Principal Component Analysis
pca = PCA()
pca_fit = pca.fit_transform(df)


# The success of the components is determined by the variance rates explained by the components.
# Information on the variance ratio explained by the components
pca.explained_variance_ratio_

# Let's take the cumulative sum and calculate the variances
np.cumsum(pca.explained_variance_ratio_)
# Components describe the data set according to the value they receive.
# For example, the first component almost explains the information in the data set.
# When we add cumulatively the values repeatedly,
# We can make a decision like this: I can explain 82% of the 3rd variable, so I can reduce it to 3 components.

####################
# Optimum Number of Components
####################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Ratio")
plt.show()

####################
# Creating Final PCA
####################

#Assume we consider 3 components
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

# Explained variance ratio
pca.explained_variance_ratio_

# Variance explained in total by 3 components (how much information)
np.cumsum(pca.explained_variance_ratio_)

####################
# BONUS: Principal Component Regression
####################

df = pd.read_csv("Machine_Learning/datasets/hitters.csv")
df.shape

len(pca_fit)

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

# For variables other than num_cols
others = [col for col in df.columns if col not in num_cols]

# Let's turn the pca_fit variables into a dataframe and name them
pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()

df[others].head()

# Let's combine two data sets, df[others] and (pca_fit, columns=["PC1","PC2","PC3"])
final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                       df[others]], axis=1# put next to each other)
final_df.head()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# I want to build a model, but there are categorical variables, since the number of classes in all of them is 2, we can use label encoder or get dummies.

def label_encoder(dataframe, binary_col):
     labelencoder = LabelEncoder()
     dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
     return dataframe

for col in ["NewLeague", "Division", "League"]:
     label_encoder(final_df, col)

final_df.dropna(inplace=True)

# Modelling
y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

#
lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
y.mean()
# We reduced the variables to 16, took a risk and got a result that was not too bad, with a reasonable process.


cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

# Let's do hyperparameter optimization

cart_params = {'max_depth': range(1, 11),
                "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                               cart_params,
                               cv=5,
                               n_jobs=-1,
                               verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))

# We take bonus approach with 330 points.


####################
# BONUS: Visualizing Multidimensional Data in 2 Dimensions with PCA
####################

####################
#BreastCancer
####################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("Machine_Learning/datasets/breast_cancer.csv")


y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

# We will try to visualize multivariate data by reducing it to two dimensions

def create_pca_df(X, y):
     X = StandardScaler().fit_transform(X)
     pca = PCA(n_components=2)
     pca_fit = pca.fit_transform(X)
     pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
     final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
     return final_df

pca_df = create_pca_df(X, y)

# We reduced it to two components and do visualization.

def plot_pca(dataframe, target):
     fig = plt.figure(figsize=(7, 5))
     ax = fig.add_subplot(1, 1, 1)
     ax.set_xlabel('PC1', fontsize=15)
     ax.set_ylabel('PC2', fontsize=15)
     ax.set_title(f'{target.capitalize()} ', fontsize=20)

     targets = list(dataframe[target].unique())
     colors = random.sample(['r', 'b', "g", "y"], len(targets))

     for t, color in zip(targets, colors):
         indices = dataframe[target] == t
         ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
     ax.legend(targets)
     ax.grid()
     plt.show()

plot_pca(pca_df, "diagnosis")

################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("Machine_Learning/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")




















