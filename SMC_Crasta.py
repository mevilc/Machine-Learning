import scipy.io
import numpy as np

data = scipy.io.loadmat("mnist_49_3000.mat")
x = np.array(data["x"])
y = np.array(data["y"][0])

# train-test split
x_train = x[:,:2000]
x_test = x[:,2000:]
y_train = y[:2000]
y_test = y[2000:]

def grad(w,b,x_train,y_train):
  c = 100
  n = 2000
  db = 0
  dw = np.array(w).reshape(-1,1)
  for i in range(2000):
    if y_train[i]*(w.T.dot(x_train[:,i].reshape(-1,1)) +b)<1:
      db += -(c*y_train[i])/n
      dw += -(c/n)*x_train[:,i].reshape(-1,1)*y_train[i]
  return dw,db 

w = np.zeros((784,1))
b = 0
for i in range(100):
  dw,db = grad(w,b,x_train,y_train)
  w-=0.01*dw
  b-=0.01*db

y_pred, y_actual = [], []
for i in range(1000):
  current_data = x_test[:,i].reshape(-1,1)
  y_pred.append(np.sign(w.T.dot(current_data)+b)[0][0])
  y_actual.append((w.T.dot(current_data)+b)[0][0])
y_pred = np.array(y_pred)

# increment test_err by 1 each time its an incorrect classification
test_err=0
for i in range(len(y_pred)):
  if y_test[i] != y_pred[i]:
    test_err += 1

test_err /= len(y_test) # divide by the total measurements
print(f"a) Test error: {round(test_err*100,2)}%")
print(f"b) Termination criteria: epsilon is 0.01")

# optimal value
c = 0
optimal = 0
for i in range(2000):
  h = y_train[i]*(np.dot(w.T,x_train[:,i])+b)
  optimal+=(c/2000)*(max(0,1-h))
optimal+=(1/2)*np.linalg.norm(w) 
print(f"c) objective function at the optimum {round(optimal[0],3)}")

indices = [] # track all indices of misclassification
for i in range(len(y_test)):
  if y_test[i] != y_pred[i]: indices.append(i)

d = {}
for i in range(len(indices)): d[indices[i]] = y_actual[indices[i]]
e = list(sorted(d.items(), key=lambda x: x[1]))

from matplotlib import pyplot as plt
for x in e[len(e): len(e) - 6: -1]:
  image = x_test[:,x[0]].reshape(28,28)
  plt.imshow(image, interpolation="nearest")
  plt.title(f'True label: {y_test[x[0]]}')
  plt.show()