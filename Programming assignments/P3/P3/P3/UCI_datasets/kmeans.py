import pandas as pd
import sys, getopt


with open(sys.argv[1], 'r') as f:
    data_file = f.read()

print(data_file)

k_clusters = sys.argv[2]

print(k_clusters)

iteration = sys.argv[3]

Centroids = (X.sample(n=iteration))



