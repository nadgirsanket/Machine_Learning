import numpy as np
from pylab import plot, show, xlabel, ylabel
np.set_printoptions(threshold=np.nan)

iterations = 100 #6000
alpha = 0.5 #0.0001
theta = np.zeros(shape=(3, 1)) #initialize theta to 0
print("Linear Regression with Multi variables initial parameters(with feature scaling):")
print("Number of Iterations=",iterations)
print("Alpha= ",alpha)
print("Initial Theta0= 0")
print("Initial Theta1= 0")
print("Initial Theta2= 0")

def cost_function(X, y, theta):
    m = y.size
    h = X.dot(theta)
    J = ((h-y).T.dot(h-y))/(2*m) #calculate cost
    return J

def gradient_descent(X, y, theta, alpha, iterations):
    m = y.size
    cost_hist = np.zeros(shape=(iterations, 1)) #initialize to 0
    for i in range(iterations):
        h = X.dot(theta) #scalar multiplication
        theta_num = theta.size
        for j in range(theta_num):
            temp = X[:, j]
            temp.shape = (m, 1)
            new_h=(h-y)*temp
            theta[j][0] = theta[j][0]-alpha*(1.0/m)*new_h.sum()
        cost_hist[i, 0] = cost_function(X, y, theta)
    print("Final Cost=",cost_hist[iterations-1])     
    return theta, cost_hist

def scale_predictor(X):
    X_scaled = X
    n_c = X.shape[1]
    for i in range(n_c):
        X_scaled[:, i] = (X_scaled[:, i]) / np.ptp(X[:, i]) #Divide by range
    return X_scaled

data = np.loadtxt('input2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
y.shape = (m, 1)
x = scale_predictor(X)
x_new = np.ones(shape=(m, 3))
x_new[:, 1:3] = x
theta, cost_history = gradient_descent(x_new, y, theta, alpha, iterations)
print ("Final Theta0=",theta[0])
print ("Final Theta1=",theta[1])
print ("Final Theta2=",theta[2])
plot(np.arange(iterations), cost_history) #plot graph
xlabel('Iterations')
ylabel('Cost J(Theta)')
show()