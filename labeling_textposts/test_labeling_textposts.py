
import pytest
import pandas as pd
import numpy as np
import proj3
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

students = pd.read_csv("labeled_masters_9.csv")
justtwo_np = students[["Labels (9)","Comments"]].to_numpy()

# Tests

def test_run_kmeans_type():
    assert isinstance(proj3.run_kmeans_sklearn(justtwo_np, 6), KMeans)
 
def test_my_kmeans_center_num():
    expected = (6)
    centers_shape = len(proj3.run_kmeans_sklearn(justtwo_np, 6).cluster_centers_)
    assert centers_shape == expected
    
def test_my_kmeans_labels():
    expected = 5
    label_max = np.max(proj3.run_kmeans_sklearn(justtwo_np, 6).labels_)
    assert label_max == expected
    
def test_run_knn_sklearn_type():
    assert isinstance(proj3.run_knn_sklearn(justtwo_np[0:50],
          students["Hand Labels"][0:50], 2), KNeighborsClassifier)

def test_take_samp_size():
    expected = 10
    fit1 = proj3.run_kmeans_sklearn(justtwo_np, 2)
    assert len(proj3.take_samp(students["Labels (9)"],fit1, 1)) == expected

def test_kNN_tester_size():
    expected = 2
    assert len(proj3.kNN_tester(justtwo_np[0:25], students["Hand Labels"][0:25], justtwo_np[25:50], students["Hand Labels"][25:50], [1, 2])) == expected

def test_looping_kmeans_size():
    expected = 1
    assert len(proj3.looping_kmeans(justtwo_np,
        list(range(1,2)))) == expected
 

