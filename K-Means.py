# Question 3

import csv
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

data1 = []
with open('data1.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
  for row in csv_reader:
    data1.append(row)
data1 = np.array(data1)

def update_assignments(data1,centroids):
  c = []
  for i in data1:
    c.append(np.argmin(np.sum((i.reshape((1,2)) - centroids) ** 2, axis=1)))
  return c

def update_centroids(data1, num_clusters, assignments):
  cen = []
  for c in range(len(num_clusters)):
    cen.append(np.mean([data1[x] for x in range(len(data1)) if assignments[x] == c], axis=0))
  return cen

centroids = (np.random.normal(size=(2,2)) * 0.0001) + np.mean(data1,axis=0).reshape((1,2))
for i in range(1200):
  a = update_assignments(data1,centroids)
  centroids = update_centroids(data1, centroids, a)
  centroids = np.array(centroids)


centroid1 = centroids[0]
centroid2 = centroids[1]
c1X, c2X, c1Y, c2Y = [], [], [], []
for i in data1:
  c1x, c1y = -2.05099733, -2.00544567
  c2x, c2y = 1.74900237, 1.79455613
  x, y = i[0], i[1]
  d1 = np.sqrt( (c1x - x) ** 2 + (c1y - y) ** 2 )
  d2 = np.sqrt( (c2x - x) ** 2 + (c2y - y) ** 2 )

  if d1 < d2 :
    c1X.append(i[0])
    c1Y.append(i[1])
  else:
    c2X.append(i[0])
    c2Y.append(i[1])

W = 0
for j in range(len(c1X)):
  a = c1X[j] - c1x
  b = c1Y[j] - c1y
  W += np.linalg.norm([a,b]) ** 2

for k in range(len(c2X)):
  a = c2X[k] - c2x
  b = c2Y[k] - c2y
  W += np.linalg.norm([a,b]) ** 2

print(f"W value: {W}")


plt.scatter(c1X[:],c1Y[:],c="red")
plt.scatter(c2X[:],c2Y[:],c="yellow")
plt.scatter(centroids[0,0],centroids[0,1],c="black")
plt.scatter(centroids[1,0],centroids[1,1],c="green")
plt.show()