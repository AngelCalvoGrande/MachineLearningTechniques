# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D   

#0 . Load the data 
# read the csv
df = pd.read_csv("T2.csv")
# list the columns
list(df)
# print number of rows and columns 
print (df.shape)

# 1. Filtering

# 1.1 Filter rows
# convert string to datetime .... Be careful!!! Spelling errors!!!
df['TimeStemp'] = pd.to_datetime(df['TimeStemp'])
# extract date from datetime
df['date'] = [d.date() for d in df['TimeStemp']]
# list the available days
df['date'].unique()
#filter data by date
df28 = df[(df['TimeStemp'] > '2016-04-28 00:00:00') & (df['TimeStemp'] <= '2016-04-28 23:59:59')]

print (df28.shape)



#1.2. Filter Features

#df28f = df28[[c for c in df if c.startswith('AccelerometerStat')]] 
#df28f = df28[[c for c in df if c.startswith('Gyroscope')]]
df28f = df28[[c for c in df if c.startswith('Orientation')]]
#df28f = df28[[c for c in df if c.startswith('RotationVector')]]
#df28f = df28[[c for c in df if c.startswith('LinearAcceleration')]]
df28f = df28[[c for c in df28f if c.endswith('MEAN')]] 
print(df28f)
list(df28f)

# RotationVector_cosThetaOver2_MEAN is a feature with all values as NaN
exclude = ["RotationVector_cosThetaOver2_MEAN"]
df28f = df28f.loc[:, df28f.columns.difference(exclude)]

# 1.3 remove missing values
df28f.isnull().values.any()
# filter/remove rows with missing values (na) (Be careful!!!)
df28f = df28f.dropna()
df28f.isnull().values.any()
print (df28f.shape)


# 2. Principal Component Analysis
#2.1 Scalation
from sklearn import preprocessing 
scaler = preprocessing.StandardScaler()
datanorm = scaler.fit_transform(df28f)

#2.2 Modelling (PCA)
from sklearn.decomposition import PCA
n_components = 3
estimator = PCA (n_components)
X_pca = estimator.fit_transform(datanorm)
print(X_pca)

# is it representative the 2D projection?
print (estimator.explained_variance_ratio_)


#2.3 Plot 
import matplotlib.pyplot as plt
import numpy

if (n_components >= 2): 
    x = X_pca[:,0]
    y = X_pca[:,1]
    plt.scatter(x,y)
    plt.show()
    

if (n_components >= 3):
 
    fig = plt.figure()
    ax = Axes3D(fig)
    x = X_pca[:,0]
    y = X_pca[:,1]
    z = X_pca[:,2]
    ax.scatter(x,y,z)
    plt.show()


# Clustering
from sklearn.cluster import KMeans

iterations = 10
max_iter = 300 
tol = 1e-04 
random_state = 0
k = 4
init = "random"
km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)

labels = km.fit_predict(X_pca)

from sklearn import metrics


distortions = []
silhouettes = []

for i in range(2, 11):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(X_pca)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(X_pca, labels))

plt.plot(range(2,11), distortions, marker='o')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.show()

plt.plot(range(2,11), silhouettes , marker='o')
plt.xlabel('K')
plt.ylabel('Silhouette')
plt.show()

print (metrics.silhouette_score(X_pca, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_pca, labels))
      
print('Distortion: %.2f' % km.inertia_)
print(labels)
x = X_pca[:,0]
y = X_pca[:,1]
plt.scatter(x,y, c = labels)
# plotting centroids
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red',s=50)
plt.grid()
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
x = X_pca[:,0]
y = X_pca[:,1]
z = X_pca[:,2]
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], c='red')
ax.scatter(x,y,z, c = labels)
plt.show()