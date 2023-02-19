# Question 2
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import numpy as np


data = scipy.io.loadmat("yalefaces.mat")
data = data['yalefaces']
index = 24
plt.imshow(data[:,:,index])

# Part a)

cov = np.cov(data[0])
eigval, eigvec = np.linalg.eig(cov)
eigvec = eigvec.T
idxs = np.argsort(eigval)[::-1]
eigval = eigval[idxs]
eigvec = eigvec[idxs]
print(eigvec)

from sklearn.decomposition import PCA

pca = PCA(0.95)
per = []
for i in data:
  x_pca = pca.fit_transform(i)
  new_dims = x_pca.shape[0]*x_pca.shape[1]
  per.append(2016/new_dims)

print(f"No. of components for 95%: {pca.n_components_}")
print(f"percentage reduction for 95%: {round(np.average(per),2)}%")

print()

pca = PCA(0.99)
per = []
for i in data:
  x_pca = pca.fit_transform(i)
  new_dims = x_pca.shape[0]*x_pca.shape[1]
  per.append(2016/new_dims)

print(f"No. of components for 99%: {pca.n_components_}")
print(f"percentage reduction for 99%: {round(np.average(per),2)}%")

# Part b)

plt.imshow(data[:,:,0],cmap="gray")