import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab

#Read input values from CSV file
data = pd.read_csv('input1_hw2.txt', names = ['x', 'y'])
X_col = pd.DataFrame(data.x)
y_col = pd.DataFrame(data.y)

m = len(y_col)   #Size of input values
iterations = 9000   #Number of iterations
alpha = 0.0001       #Desired value of alpha (Learning rate)
theta = np.array([0, 0])        #Set initial value of theta0 and theta1 to 0
print("Linear Regression with one variable initial parameters:")
print("Number of Iterations=",iterations)
print("Alpha= ",alpha)
print("Initial Theta0= 0")
print("Initial Theta1= 0")
X_col['i'] = 1 #Add initial X-intercept
X = np.array(X_col) #convert to numpy arrays for easy math functions
Y = np.array(y_col).flatten()

X_sig = np.array([0.5,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50])
Y_sig = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])

def sigmoid(x, x0, k):
     g = 1 / (1 + np.exp(-k*(x-x0)))
     return g

def gradient_descent(X, Y, theta, alpha, iterations):
    cost_arr = [0]*iterations #initialize cost_arr array   
    for i in range(iterations):
        h = 1.0 / (1.0 + np.exp(X.dot(-theta)))
        gradient = X.T.dot(h-Y)/m
        theta = theta - alpha*gradient
        cost = cost_function(X, Y, theta)
        cost_arr[i] = cost
    print("Initial cost=",cost_arr[0])
    print("Final cost=",cost_arr[iterations-1])
    np_X = [0] * iterations
    np_cost_hist=np.array(cost_arr)
    np_X=list(range(iterations))
    return theta, cost_arr

def cost_function(X, y, theta):
    m = len(y) 
    h = 1.0 / (1.0 + np.exp(X.dot(-theta)))
    J = (-1.0/m)*(sum(y*np.log(h)+(1-y)*np.log(1-h)))
    return J

(theta_final, cost_final) = gradient_descent(X,Y,theta,alpha,iterations)

print("Final Theta0=",theta_final[1])
print("Final Theta1=",theta_final[0])

regression_x = list(range(25))
regression_y = [theta_final[1] + theta_final[0]*xi for xi in regression_x] #staright line equation
plt.figure(figsize=(10,6))  #plot graph
plt.plot(regression_x, regression_y, '-', label='linear')
plt.axis([0,6.0,-1.5,1.5])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear and Logistic Regression')  

popt, pcov = curve_fit(sigmoid, X_sig, Y_sig)
print (popt)

x_sig = np.linspace(0.5, 5.5)
y_sig = sigmoid(x_sig, *popt)

pylab.plot(X, Y, 'o')
pylab.plot(x_sig,y_sig, label='Logistic')
pylab.ylim(-0.05, 1.05)
pylab.legend(loc='best')
pylab.show()