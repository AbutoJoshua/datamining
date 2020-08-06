import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz


#Loading in data
cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))

print("Shape of cancer data: {}".format(cancer.data.shape))

print("Sample counts per class:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))

print(cancer.feature_names, cancer.target_names)
for i in range(0, 3):
    print(cancer.data[i], cancer.target[i])

#K-Nearest neighbor
train_feature, test_feature, train_class, test_class = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_feature, train_class)

print("Test set predictions:\n{}".format(knn.predict(test_feature)))

print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))

linearsvm = LinearSVC(random_state=0, max_iter = 1000000).fit(train_feature, train_class)
print("Test set score: {:.3f}".format(linearsvm.score(test_feature, test_class)))

