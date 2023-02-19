# CISC 684
# HW3, Q2
# Mevil Crasta

import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
url =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=",")
x = dataset[:,0:-1]
m = np.median(x, axis = 0)
x = (x>m)*2+(x<=m)*1; # making the feature vectors binary
y = dataset[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.3, random_state = 17)

# class that holds all functions
class N_B():
  def __init__(self, X, y):
    self.num_examples, self.num_features = X.shape
    self.num_classes = len(np.unique(y))
    self.eps =1e-6
  
  # train model with training data
  def fit(self, X, y):
    self.classes_mean = {}
    self.classes_variance = {}
    self.classes_prior = {}

    for c in range(self.num_classes):
      X_c = X[y==c]

      self.classes_mean[str(c)] = np.mean(X_c, axis=0)
      self.classes_variance[str(c)] = np.var(X_c, axis=0)
      self.classes_prior[str(c)] = X_c.shape[0]/self.num_examples

  # test model with test data
  def predict(self, X):
    probs = np.zeros((1381, self.num_classes))
    
    for c in range(self.num_classes):
      prior = self.classes_prior[str(c)]
      probs_c = self.density_function(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
      probs[:,c] = probs_c + np.log(prior)
    
    return np.argmax(probs, 1) # returns highest conditional probability

  def density_function(self, x, mean, sigma):
    const = -self.num_features/2 * np.log(2*np.pi) - 0.5*np.sum(np.log(sigma+self.eps))
    probs = 0.5*np.sum(np.power(x-mean, 2)/(sigma+self.eps), 1)
    return const-probs

NB = N_B(x_train, y_train)
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)
print("Test labels: ", y_test)
print("Predicted labels: ", y_pred)

# increment test_err by 1 each time its an incorrect classification
test_err=0
for i in range(len(y_pred)):
  if y_test[i] != y_pred[i]:
    test_err += 1

test_err /= len(y_test) # divide by the total measurements
print("Test error: " +str(round(test_err*100,2))+"%")