#***************************************************
#                                                  *
#       Programmer: Nikola Andric                  *
#       Email: nikolazeljkoandric@gmail.com        *
#       Last Editted: 11/08/2021                   *
#                                                  *
#***************************************************
#
#
'''Using Kalmat filtering technique to predict the future position of the car on the XY plane / image. 

               State Exploration Equation:
                        x_t_plus_1 = F * x_t + G * U_t +  w_t    where  x_t_plus_1 is the predicted system state vector at the time step t+1
                                                                        x_t is an estimated system state vector at the time step t
                                                                        U_t is the input control vector
                                                                        w_t is the process noise vector
                                                                        F is the state transition matrix that enables us to conver the previous state into a new estimated state
                                                                        G is the control transition matrix that enables input vecor to be in the form/shape that can influence the future state X_t_plus_1
                In our case, we do not have any input (wind, uneven terain, or anything like that), and for thet sake of the simplicity, we will not consider the noise w_t. 
                Hence, the state exploration equation boils down to:
                        x_t_plus_1 = F * x_t

                The system state is defined as a vector:
                    x = [[x_position],
                        [x_velocity],
                        [x_acceleration],
                        [y_position],
                        [y_velocity],
                        [y_acceleration]]

                The extrapolated car state for time t+1 can be described by the following system of equations:
                x_position_t_plus_1 = x_position_t + x_velocity_t * delta_t + 0.5 * x_acceleration_t * delta_t**2
                x_velocity_t_plus_1 = x_velocity_t + x_acceleration_t * delta_t
                x_acceleration_t_plus_1 = x_acceleration_t
                y_position_t_plus_1 = y_position_t + y_velocity_t * delta_t + 0.5 * y_acceleration_t * delta_t**2
                y_velocity_t_plus_1 = y_velocity_t + y_acceleration_t * delta_t
                y_acceleration_t_plus_1 = y_acceleration_t

                where delta_t represents the change in time. In our case, the time between video frames. 

                This can be represented in the matrix form as follows:
                [x_position_t_plus_1    ]   [1, delta_t, 0.5*delta_t**2, 0,    0   ,       0       ]   [x_position_t    ]
                [x_velocity_t_plus_1    ]   [0,    1   ,    delta_t    , 0,    0   ,       0       ]   [x_velocity_t    ]
                [x_acceleration_t_plus_1] = [0, delta_t,       1       , 0,    0   ,       0       ] * [x_acceleration_t]
                [y_position_t_plus_1    ]   [0,    0   ,       0       , 1, delta_t, 0.5*delta_t**2]   [y_position_t    ]
                [y_velocity_t_plus_1    ]   [0,    0   ,       0       , 0,    1   ,    delta_t    ]   [y_velocity_t    ]
                [y_acceleration_t_plus_1]   [0,    0   ,       0       , 0, delta_t,       1       ]   [y_acceleration_t]

            The covariance extrapolation equation:
                P_t_plus_1 = F * P_t * F.T + Q  where   P_t is the estimate uncertainty (covariance) matrix of the current state
                                                        P_t_plus_1 is the predicted estimate uncertainty (covariance) matrix for the next sate
                                                        F is the state transition matrix
                                                        Q is the process noise matrix
'''

from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# State Matrix initialization
# Initialize the vector x which is a 6 x 1 matrix
# We don't know the vehicle location; we will set initial position, velocity and acceleration to 0.
#x_init = [[0 for x in range(6)] for x in range(1)]
x_init = np.zeros((6, 1))

print("Initial State Matrix:")
print(x_init, end="\n\n")

# State Covariance Matrix (Error in the estimate)
# Since our initial state vector is a guess, we will set a very high estimate uncertainty. The high estimate uncertainty results in a high Kalman Gain, giving a high weight to the measurement.
# This will be a 6 x 6 matrix of  
P_init = np.zeros((6, 6))

# Change the values on the diagonal of the State Covariance Matrix
for i in range(6):
    P_init[i][i] = 500

print("Initial State Covariance Matrix P (Error in the estimate matrix):")
for xi in P_init:
    print(xi)
print("\n\n")

# Difference in time between the measurements
delta_t = 1 

# Create the F matrix that represents the state transition matrix.
F = np.zeros((6, 6))
for i in range(6):
    F[i][i] = 1
F[0][1] = delta_t
F[0][2] = 0.5 * delta_t**2
F[1][2] = delta_t
F[3][4] = delta_t
F[3][5] = 0.5 * delta_t**2
F[4][5] = delta_t

print("State transition matrix F:")
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

print("Process noise convariance matrix Q:")
for xi in Q:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
print("\n\n")


## TO DO: Still need the random acceleration standard deviation. We can measure this using cameras and standard physics equations. 
# V = S / t
# a = V / t
# for testing purposes let it be 0.15 m/s**2
acceleration_std_dev = 0.15


# To predict the next state, we need to predict the next state covariance matrix P_t-1 = F*P_init*F.t + Q of the initial state.

P = np.dot(np.dot(F, P_init),F.T) + Q

print("State covariance matrix P_t-1:")
for xi in P:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
print("\n\n")

# Observation Transition Martix H must be 2 x 6 since Z is 2 x 1 and X is 6 x 1
H = np.zeros((2, 6))
H[0][0] = 1
H[1][3] = 1

print("Observation Transition Martix H")
for xi in H:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
print("\n\n")



# Define R. For that we need the standard deviation of x and y.
# for testing purposes we will let x and y error be 3 meters
x_measure_error, y_measure_error = 3, 3

R = np.zeros((2,2))
R[0][0] = x_measure_error ** 2
R[1][1] = y_measure_error ** 2

print("measurement uncertainty R")
for xi in R:
    print("{:.0f}   {:.0f}  ".format(xi[0],xi[1]))
print("\n\n")


#test measurements
x_m = [-393.66,-375.93,-351.04,-328.96,-299.35,-273.36,-245.89,-222.58,-198.03,-174.17,-146.32,-123.72,-103.47,-78.23, -52.63,-23.34,25.96, 49.72, 76.94, 95.38, 119.83,144.01,161.84,180.56,201.42,222.62,239.4,252.51,266.26,271.75,277.4, 294.12,301.23,291.8,299.89]
y_m = [300.4,   301.78, 295.1,  305.19, 301.06, 302.05, 300,    303.57, 296.33, 297.65, 297.41, 299.61, 299.6,  302.39,295.04,300.09,294.72,298.61,294.64,284.88,272.82,264.93,251.46,241.27,222.98,203.73,184.1,166.12,138.71,119.71,100.41,79.76, 50.62, 32.99,2.14]

x_predict = []
y_predict = []

first_time = True

#Start iterations (35 for now):
for i in range(35):

    print("ITERATION NUMBER: ",i+1)
    # Measure ( x and y value) Z = x*H
    z = np.zeros((2,1))
    z[0][0] = x_m[i]
    z[1][0] = y_m[i]
    print(z)
    # Update:
        # Kalman Gain
    if first_time:
        first_time = False
        x_prev = x_init
    else:
        P = P_new_error # if it is not the initial case, update the P matrix with the one from the previous iteration.
        x_prev = x_future # same for state X

    #K = np.dot(np.dot(P,H.T), inv((np.dot(np.dot(H,P),H.T) + R)) ) # this results in a 6 x 2 matrix
    K = P @ H.T @ inv( (H @ P @ H.T) + R )
        # Estimate the current estimate
    print("K:")
    print(K.shape)
    for xi in K:
        print("{:.4f}   {:.4f}".format(xi[0],xi[1]))
    print("\n\n")

    #x_curr = x_prev + np.dot(K, z - np.dot(H, x_prev)) #results in 6 x 1 matrix
    x_curr =  x_prev + (K @ (z - (H @ x_prev)))
    print(x_curr.shape)  
    print("x:")
    for xi in x_curr:
        print("{:.4f}".format(xi[0]))
    print("\n\n")
        # Update Current estimate uncertainty (error)
    P = np.dot(np.dot((np.identity(6) - np.dot(K,H)), P),(np.identity(6)-np.dot(K,H)).T) + np.dot(np.dot(K,R),K.T) # results in a 6 x 6 matrix
    print("P:")
    print(P.shape)
    for xi in P:
        print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
    print("\n\n")

    # Predict
    x_future = np.dot(F,x_curr) # results in a 6 x 1 matrix

    #store those values
    print(x_future)
    x_predict.append(x_future[0][0])
    y_predict.append(x_future[3][0])

    P_new_error = np.dot(np.dot(F,P),F.T) + Q #results in a 6 x 6 matrix
    print("P_new:")
    for xi in P_new_error:
        print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3],xi[4],xi[5]))
    print("\n\n")

plt.plot(x_predict, y_predict)
plt.plot(x_m, y_m)
plt.show()
