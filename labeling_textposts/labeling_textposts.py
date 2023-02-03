"""
New component of final learning portfolio
Emma Schumacher
"""

# import block
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer #!
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random
from scipy.spatial import distance
from sklearn.cluster import KMeans

""" Using _within cluster sum of squares_ as the measure of
  cluster "goodness", write a function `looping_kmeans` that perform
k-means using `sklearn` and computes the "goodness" of clusters for
k=1, k=2, ..., k=10._ """
def looping_kmeans(my_np, k_list):
    #normalize my variables
    norm_vars = []
    for i in range(np.shape(my_np)[1]):
        var = my_np[:,i]
        mx = np.max(var)
        mn = np.min(var)
        var_norm = (var - mn)/(mx - mn)
        var_norm = np.around(var_norm, decimals = 2)
        
        norm_vars.append(var_norm)
        
    justtwo_norm = np.stack(norm_vars,axis=-1)
    
    # perform k-means using `sklearn` and computes the "goodness" of clusters for k=1, k=2, ...
    good_list = []
    for k in k_list:
        total = 0
        km_alg_norm = KMeans(n_clusters=k, init="random", random_state = 1, max_iter = 200)
        fit2 = km_alg_norm.fit(justtwo_norm)
        
        for i in range (0, k):
            # Compute the following for each cluster:
            cluster_center = [fit2.cluster_centers_[i]]
            inds = (fit2.labels_ == i)
            cluster_points = justtwo_norm[inds,:]
            
            # Given
            cluster_spread = distance.cdist(cluster_points, cluster_center, 'euclidean')
            cluster_total = np.sum(cluster_spread)
        
            # Add all the cluster_totals together
            total += cluster_total
        good_list.append(total)
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(k_list, good_list)
    ax.scatter(k_list, good_list)
    for i in range (len(k_list)):
        plt.text(k_list[i], good_list[i], f'{k_list[i]}, {round(good_list[i])}')
    plt.show()
    
    return(good_list)
    
""" Use sklearn to create a quick fitted kmeans """
def run_kmeans_sklearn(my_np, k):
    # Set up my k-Means with k clusters
    km_alg = KMeans(n_clusters=k, init="random",random_state = 1, max_iter = 200)

    # fit the k-Means to my data
    fitted = km_alg.fit(my_np)
     
    return (fitted)

""" Use sklearn to create a quick fitted kNN """
def run_knn_sklearn(x_train, y_train, k):
    # Set up my k-Means with k clusters
    kn_alg = KNeighborsClassifier(n_neighbors=k)

    # fit the k-Means to my data
    fitted = kn_alg.fit(x_train, y_train)
     
    return (fitted)

""" Take 10 samples from a specific label """
def take_samp(col, fit1, label):
    # print them
    lst = col[fit1.labels_ == label].sample(10)
    print(lst)
    
    return(lst)

""" Check accuracy of kNN models made with different neigbors """
def kNN_tester(x_train, y_train, x_test, y_test, k_list):
    
    test_list = []
    train_list = []
    
    for k in k_list:
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_train)
        train_list.append(accuracy_score(y_train, pred))
        
        pred = knn.predict(x_test)
        test_list.append( accuracy_score(y_test,pred))
    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(k_list, test_list)
    #ax.plot(k_list, train_list)
    for i in range (len(k_list)):
        plt.text(k_list[i], test_list[i], f'{k_list[i]}, {test_list[i]}')
    plt.show()
    
    return(test_list)
