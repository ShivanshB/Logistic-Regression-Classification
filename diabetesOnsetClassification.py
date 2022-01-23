from mpl_toolkits import mplot3d
from matplotlib import pyplot
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import os
import pickle


# Some of this code is adapted from the code I wrote in Andrew Ng's Coursera Machine Learning Course. 
# However, it is largely modified and extende for the specific purpose of classifying breast Cancer data.


# Creating the initiad 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.view_init(60, 35)


# Reading the CSV data file from UCI School of Medicine.
# Parsing the data into numpy arrays to perform clustering on.
data = pd.read_csv('diabetes.csv')
X = pd.DataFrame(data, columns= ['Glucose','BloodPressure','BMI']).to_numpy()
P = pd.DataFrame(data, columns= ['Glucose','BloodPressure','BMI','Outcome']).to_numpy()
y = pd.DataFrame(data, columns= ['Outcome']).to_numpy()
y=y.squeeze()

# Sigmoid function implementation for the logistic regression.
def sigmoid(z):

    z = np.array(z)
    g = np.zeros(z.shape)

    sig = lambda x: (1./(1 + np.exp(-1*x)))
    g = np.vectorize(sig)(z)

    return g

# Adding a columns of ones onto the dataset to allow for matrix multiplication with the theta vector.
m,n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# Calculates the cost of the current values of theta with the entire dataset.
def costFunction(theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X.dot(theta.T))

    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))
    grad = (1 / m) * (h - y).dot(X)

    return J, grad

initial_theta = np.zeros(n+1)
cost, grad = costFunction(initial_theta, X, y)

# Optimize module's minimization function using the cost function we wrote.
options= {'maxiter': 400}
res = optimize.minimize(costFunction,initial_theta,(X, y), jac=True,method='TNC',options=options)

# Look at the theta values that are found and use them to calculate the equation for a dividing plane.
theta = res.x
print(theta)

# The below equation is what was calculated from the above values of theta.
xx, yy = np.meshgrid(range(200), range(200))
z = (91.01 - 0.443*xx + 0.0886*yy) 

# plot the plane
ax.plot_surface(xx, yy, z, alpha=0.5)

# Separates data into each cluster and plots it by color.
a = P[P[:,3] == 0]
b = P[P[:,3] == 1]

ax.scatter(a[:,0],a[:,1],a[:,2],cmap = 'Greens')
ax.scatter(b[:,0],b[:,1],b[:,2], cmap='Black');
ax.set_xlabel('Plasma Glucose')
ax.set_ylabel('Blood Pressure (Diastolic, mmHg)')
ax.set_zlabel(r'BMI')
plt.show()

pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))