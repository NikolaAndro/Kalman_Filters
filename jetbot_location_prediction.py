#***************************************************
#                                                  *
#       Programmer: Nikola Andric                  *
#       Email: nikolazeljkoandric@gmail.com        *
#       Last Editted: 11/07/2021                   *
#                                                  *
#***************************************************
#
#
# DESCRIPTION: Using Kalmat filtering technique to predict the future position of the Jetbot. 
#
#

from math import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# State Matrix initialization
# Initialize the vector x which is a 6 x 1 matrix
# We don't know the vehicle location; we will set initial position, velocity and acceleration to 0.
#x_init = [[0 for x in range(6)] for x in range(1)]
x = np.zeros((6, 6))
print("Initial State Matrix:")
print(x, end="\n\n")

# State Covariance Matrix (Error in the estimate)
# Since our initial state vector is a guess, we will set a very high estimate uncertainty. The high estimate uncertainty results in a high Kalman Gain, giving a high weight to the measurement.
# This will be a 6 x 6 matrix of  
#P_init = [[0 for x in range(6)] for x in range(6)]
P_init = np.zeros((6, 6))

# Change the values on the diagonal of the State Covariance Matrix
for i in range(6):
    P_init[i][i] = 500

print("Initial State Covariance Matrix (Error in the estimate matrix):")
for xi in P_init:
    print(xi)
print("\n\n")

# delta_t represents the time difference between the frames (time between taking the images via camera).
# This time is gotten by the equation 1/FPS, where FPS stands for frames per second and has value of 12.4 on the cameras we use. 

FPS = 12.4
delta_t = 1 / FPS

# Create the F matrix that represents the state transition matrix.
F = [[0 for x in range(6)] for x in range(6)]
F = np.zeros((6, 6))
for i in range(6):
    F[i][i] = 1
F[0][1] = delta_t
F[0][2] = 0.5 * delta_t**2
F[1][2] = delta_t
F[3][4] = delta_t
F[3][5] = 0.5 * delta_t**2
F[4][5] = delta_t

print("State transition matrix:")
for xi in F:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
print("\n\n")


# Create the Q matix that represents the process noise convariance
Q = np.zeros((6, 6))

Q[0][0]= delta_t**4 / 4
Q[0][1]= delta_t**3 / 2
Q[0][2]= delta_t**2 / 2
Q[1][0]= delta_t**3 / 2
Q[1][1]= delta_t**2
Q[1][2]= delta_t
Q[2][0]= delta_t**2 / 2
Q[2][1]= delta_t
Q[2][2]= 1
Q[3][3]= delta_t**4 / 4
Q[3][4]= delta_t**3 / 2
Q[3][5]= delta_t**2 / 2
Q[4][3]= delta_t**3 / 2
Q[4][4]= delta_t**2
Q[4][5]= delta_t
Q[5][3]= delta_t**2 / 2
Q[5][4]= delta_t
Q[5][5]= 1

print("Process noise convariance matrix:")
for xi in Q:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
print("\n\n")


## TO DO: Still need the random acceleration standard deviation. We can measure this using cameras and standard physics equations. 
# V = S / t
# a = V / t


# To predict the next state, we need to predict the next state covariance matrix P_t-1 = F*P_init*F.t + Q

P = np.dot(np.dot(F, P_init),F.T) + Q

print("State covariance matrix P_t-1:")
for xi in P:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
print("\n\n")

# Observation Transition Martix H must be 2 x 6 since Z is 2 x 1 and X is 6 x 1
H = Q = np.zeros((2, 6))
H[0][0] = 1
H[1][3] = 1

print("Observation Transition Martix H")
for xi in H:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
print("\n\n")

#

# Define R. For that we need the standard deviation of x and y.

# Start iterations (25 for now):
for i in range(25):
    # Measure ( x and y value) Z = [x,y]*H
    # Update:
        # Kalman Gain
        K = np.dot(np.dot(P,H.T), (np.dot(np.dot(H,P),H.T) + R)**(-1) )
        # Estimate the current state
        x = x + np.dot(K, Z - np.dot(H, x))
        # Update Current state
        P = np.dot(np.dot((np.idenity(6) - np.dot(K,H)), P),(np.idenity(6)-np.dot(K,H)).T) + np.dot(np.dot(K,R),K.T)
    # Predict
    x_future = np.dot(F,x)
    P_future = np.dot(np.dot(F,P),F.T) + Q
