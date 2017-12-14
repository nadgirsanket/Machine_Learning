import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read input values from CSV file
data = pd.read_csv('input1.txt', names = ['x', 'y'])
X_col = pd.DataFrame(data.x)
y_col = pd.DataFrame(data.y)

m = len(y_col)   #Size of input values
iterations = 400    #Number of iterations
alpha = 0.001       #Desired value of alpha (Learning rate)
theta = np.array([0, 0])        #Set initial value of theta0 and theta1 to 0
print("Linear Regression with one variable initial parameters:")
print("Number of Iterations=",iterations)
print("Alpha= ",alpha)
print("Initial Theta0= 0")
print("Initial Theta1= 0")
X_col['i'] = 1 #Add initial X-intercept
X = np.array(X_col) #convert to numpy arrays for easy math functions
Y = np.array(y_col).flatten()


def gradient_descent(X, Y, theta, alpha, iterations):
    cost_arr = [0]*iterations #initialize cost_arr array   
    for i in range(iterations):
        h = X.dot(theta)
        gradient = X.T.dot(h-Y)/m
        theta = theta - alpha*gradient
        cost = cost_function(X, Y, theta)
        cost_arr[i] = cost
    print("Final cost=",cost_arr[iterations-1])
    np_X = [0] * iterations
    np_cost_hist=np.array(cost_arr)
    np_X=list(range(400))
    plt.figure(figsize=(10,8))
    plt.plot(np_X, np_cost_hist, '-')
    plt.ylabel('Cost J(Theta)')
    plt.xlabel('Iterations')
    plt.title('Cost J vs Number of iterations')
    return theta, cost_arr

def cost_function(X, y, theta):
    m = len(y) 
    J = np.sum((X.dot(theta)-y)**2)/(2*m) #calculate cost   
    return J

(theta_final, cost_final) = gradient_descent(X,Y,theta,alpha,iterations)

print("Final Theta0=",theta_final[0])
print("Final Theta1=",theta_final[1])

regression_x = list(range(25))
regression_y = [theta_final[1] + theta_final[0]*xi for xi in regression_x] #staright line equation
plt.figure(figsize=(10,8))  #plot graph
plt.plot(X_col.x, y_col.y, 'ro')
plt.plot(regression_x, regression_y, '-')
plt.axis([0,6.5,0,14])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y -- Regression Line')  