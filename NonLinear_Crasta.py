import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)
n=10
m=8
x_train = np.linspace(0,3,n)
y_train = -x_train**2 +2*x_train + 2 +0.5*np.random.randn(n)
x_test = np.linspace(0,3,m)
y_test = -x_test**2 +2*x_test + 2 + 0.5*np.random.randn(m)
plt.plot(x_train,y_train,'o')
plt.plot(x_test,y_test,'x')
plt.plot(x_train,-x_train**2 +2*x_train + 2)
plt.legend(['training samples','test samples','true line'])

#new_x = np.array([x_train, x_train**2, x_train**3, x_train**4]).reshape(-1,1)
#print(f"new_x: {new_x} w/ len {len(new_x)}")

x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

r = 0.1
a = np.dot(np.transpose(x_train),x_train) + n*r*np.eye(len(x_train))
w_hat = np.linalg.inv(a)*(np.dot(np.transpose(x_train),y_train))
w_ = 0
for i in range(n):
  w_ += (y_train[i] - np.dot(np.transpose(w_hat),x_train))**2

w_hat += r*np.linalg.norm(w_hat)
w_hat = np.argmin(w_hat)
b_hat = y_mean - w_hat*x_mean
print(f"b) w is {round(w_hat,3)} and b is {round(b_hat,3)}")