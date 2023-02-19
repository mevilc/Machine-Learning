import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)
n=10
m=8
x_train = np.linspace(0,3,n)
y_train = 2.0*x_train + 1.0*np.random.randn(n)

plt.plot(x_train,y_train,'o')
plt.plot(x_train,2*x_train + 1)

x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

# Part a)
top, bottom = 0,0
for i in range(n):
  top += (x_train[i] - x_mean) * (y_train[i] - y_mean)
  bottom += (x_train[i] - x_mean)**2
w = top/bottom
b = y_mean - w*x_mean
print(f"a) w is {round(w,3)} and b is {round(b,3)}")

# Part b)
r = 3
a = np.dot(np.transpose(x_train),x_train) + n*r*np.eye(len(x_train))
w_hat = np.linalg.inv(a)*(np.dot(np.transpose(x_train),y_train))
w_ = 0
for i in range(n):
  w_ += (y_train[i] - np.dot(np.transpose(w_hat),x_train))**2

w_hat += r*np.linalg.norm(w_hat)
w_hat = np.argmin(w_hat)
b_hat = y_mean - w_hat*x_mean
print(f"b) w is {round(w_hat,3)} and b is {round(b_hat,3)}")

# Part c)
plt.plot(x_train, w*x_train+b)
plt.plot(x_train, w_hat*x_train+b_hat)

plt.legend(['data','true line', 'OLS', 'ridge'])