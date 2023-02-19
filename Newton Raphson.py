# CISC 684 
# Homework 4
# Mevil Crasta

# RUNTIME ~50s
import scipy.io
import numpy as np

data = scipy.io.loadmat("mnist_49_3000.mat")
x = np.array(data["x"])
y = np.array(data["y"][0])
y[y==-1] = 0

# train-test split
x_train = x[:,:2000]
x_test = x[:,2000:]
y_train = y[:2000]
y_test = y[2000:]

# sigmoid used for gradient descent
def sigmoid(theta,xi):
  return 1/(1+np.exp(-np.inner(theta,xi)))

# gradient descent
def grad(theta,x_train,y_train):
  r = 10 # regularization parameter
  gradient = 2*r*theta
  hessian = 2*r*np.eye(x_train.shape[0])
  n = x_train.shape[1] 
  for i in range(n):
    current_data = x_train[:,i]
    gradient += (current_data*(sigmoid(theta,current_data)-y_train[i]))
    hessian += np.outer(current_data,current_data)*(sigmoid(theta,current_data))*(1-sigmoid(theta,current_data))
  return gradient, hessian

#running the gradient decent
theta = np.zeros((x_train.shape[0])) 
alpha = 0.1
for i in range(10):
  g,h = grad(theta,x_train,y_train)
  # applying lambda/2m + theta^2 for regularized logistic regression
  theta-= np.linalg.inv(h).dot(g)+(10/4000)*np.square(theta)


#prediction and test accuracy
prob=[]
for i in range(1000):
  current_data = x_test[:,i]
  prob.append(sigmoid(theta,current_data))
prob = np.array(prob)

y_pred = (prob>0.5)*1.0

# increment test_err by 1 each time its an incorrect classification
test_err=0
for i in range(len(y_pred)):
  if y_test[i] != y_pred[i]:
    test_err += 1

test_err /= len(y_test) # divide by the total measurements
print(f"a) Test error: {round(test_err*100,2)}")
print(f"b) Termination criteria: epsilon is 0.01")
print(f"c) Optimum value is: {round(np.argmin(prob),8)}")

indices = [] # track all indices of misclassification
for i in range(len(y_test)):
  if y_test[i] != y_pred[i]: indices.append(i)

from matplotlib import pyplot as plt
for i in range(5):
  index = indices[i] #change the index to show different images
  image = x_test[:,index].reshape(28,28)
  plt.imshow(image, interpolation="nearest")
  plt.title(f'True label: {y_test[i]}')
  plt.show()