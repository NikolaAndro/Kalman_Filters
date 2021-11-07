#***************************************************
#                                                  *
#       Programmer: Nikola Andric                  *
#       Email: nikolazeljkoandric@gmail.com        *
#       Last Editted: 11/06/2021                   *
#                                                  *
#***************************************************
#
# Problem statement: Predict the position of the robot using Kalman Filter approach.    
#                   Given measurements list [5., 6., 7., 9., 10.], motions/input list [1., 1., 2., 1., 1.], 
#                   initial mu (0), initial prediction uncertainty (10000),
#                   as well as constant measurement and motion uncertainty(4 and 2). 
#

from math import *
import matplotlib.pyplot as plt
import numpy as np

# Gaussian function
def gaussian(mu, sigma_2, x):
    ''' gaussian takes in a mean and squared variance, and an input x and returns the gaussian value.'''
    coefficient = 1.0/sqrt(2.0 * pi * sigma_2)
    exponential = exp(-0.5 * (x - mu) ** 2 / sigma_2)
    return coefficient * exponential

# the measurement update function
def update(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters.'''
    # Calculate the new parameters
    new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
    new_var = 1/(1/var2 + 1/var1)
    
    return [new_mean, new_var]

# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2
    
    return [new_mean, new_var]

# measurements for mu and motions, U
measurements = [5., 6., 7., 9., 10.]
motions = [1., 1., 2., 1., 1.]

# initial parameters
# Note that the initial estimate is set to the location 0, and the variance is extremely large; this is a state of high confusion.
# The perfect scenario would be if the initial estimate is set to 5 (perfect guess). 

measurement_sig = 4. # measurement uncertainty
motion_sig = 2. # motion uncertainty
mu = 0.
sig = 10000.


## TODO: Loop through all measurements/motions
# this code assumes measurements and motions have the same length
# so their updates can be performed in pairs
for n in range(len(measurements)):
    # measurement update, with uncertainty
    mu, sig = update(mu, sig, measurements[n], measurement_sig)
    print('Update: [{}, {}]'.format(mu, sig))
    # motion update, with uncertainty
    mu, sig = predict(mu, sig, motions[n], motion_sig)
    print('Predict: [{}, {}]'.format(mu, sig))

"""
1.  Due to a very high initial uncertainty, the initial estimation is 4.99 because the estimate is dominated by the measurement that has uncertainty 4. 
    The uncertainty shrinks to 3.99 (a bit better than measurement uncertainty).
2.  Then our model predicts that the robot moved for 1 unit, but the uncertainty increased to 5.99.
    Again, the measure will dominate over the estimation due to less uncertainty. After the update the uncertainty shrinks to 2.39. 
3.  Next prediction is 6.99 with uncertainty 4.39. Measurement is 6.99 with uncertainty 4. Hence, the estimate is 6.99 and uncertainty shrinks to 2.09.
4.  Prediction 8.99 with uncertainty 4.095, while the measurement is 8.99 with uncertainty 4. Estimate ends up being 8.99 with uncertainty 2.023.
5.  Prediction is 9.99 with uncertainty 4.02 while the measurement is 9.99 with uncertainty 4. Estimate is 9.99 with uncertainty 2.005.
6.  Prediction is 10.99 with uncertainy 4.005, which is our final result. 


"""
# print the final, resultant mu, sig
print('\n')
print('Final result: [{}, {}]'.format(mu, sig))

# Plotting a Final Gaussian by looping through a range of x values and creating a resulting list of Gaussian values.
# set the parameters equal to the output of the Kalman filter result
mu = mu
sigma2 = sig

# define a range of x values
x_axis = np.arange(-20, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(gaussian(mu, sigma2, x))

# plot the result 
plt.plot(x_axis, g)
plt.show()

# display the *initial* gaussian over a range of x values
# define the parameters
mu = 0
sigma2 = 10000

# define a range of x values
x_axis = np.arange(-20, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(gaussian(mu, sigma2, x))

# plot the result 
plt.plot(x_axis, g)
plt.show()