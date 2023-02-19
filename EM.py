# CISC 684
# Mevil Crasta
# HW8, Q2

import csv
import numpy as np
import matplotlib.pyplot as plt

with open('EM.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
  data1 = []
  for row in csv_reader:
    data1.append(row)
data1 = np.array(data1)

from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3, max_iter=50)
model.fit(data1)

print(f"Model Weights: {model.weights_}\n")
print(f"Model Covariances: {model.covariances_}\n")
print(f"Model Means: {model.means_}")