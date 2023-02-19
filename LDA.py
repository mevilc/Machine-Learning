# Mevil Crasta
# CISC 684
# Homework 2, Q2

import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

url =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=",")
x = dataset[:,0:-1]
y = dataset[:,-1]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.30, random_state = 17)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train) # train model
predicted_class = lda.predict(x_test) # test model

test_err = 0

# increment test_err by 1 each time its an incorrect classification
for i in range(len(predicted_class)):
  if y_test[i] != predicted_class[i]:
    test_err += 1

test_err /= len(y_test) # divide by the total measurements
print("Test error: " +str(round(test_err*100,2))+"%")