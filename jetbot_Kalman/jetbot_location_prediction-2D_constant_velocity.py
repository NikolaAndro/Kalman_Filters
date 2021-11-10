#***************************************************
#                                                  *
#       Programmer: Nikola Andric                  *
#       Email: nikolazeljkoandric@gmail.com        *
#       Last Editted: 11/08/2021                   *
#                                                  *
#***************************************************
#
#
'''Using Kalman filtering technique to predict the future position of the Jetbot on the XY plane / image. 
               State Exploration Equation:
                        x_t = F * x_t_minus_1 + G * U_t_minus_1 +  w_t_minus_1  where   x_t is the predicted system state vector at the time step t+1
                                                                                        x_t_minus_1 is an estimated system state vector at the time step t
                                                                                        U_t_minus_1 is the input control vector
                                                                                        w_t_minus_1 is the process noise vector
                                                                                        F is the state transition matrix that enables us to conver the previous state into a new estimated state
                                                                                        G is the control transition matrix that enables input vecor to be in the form/shape that can influence the future state X_t_plus_1
                In our case, we do not have any input (wind, uneven terain, or anything like that). 
                Hence, the state exploration equation boils down to:
                        x_t = F * x_t_minus_1 + w_t_minus_1

                Let us see how we got to this equation (derivation):

                The system state is defined as a vector:
                    x = [[x_position],
                        [y_position],
                        [x_velocity],
                        [y_velocity]]

                The extrapolated jetbot state for time t+1 can be described by the following system of equations:
                x_position_t = x_position_t_minus_1 + delta_t * x_velocity_t_minus_1 + 0.5 * delta_t**2 * x_noise_t_minus_1
                y_position_t = y_position_t_minus_1 + delta_t * y_velocity_t_minus_1 + 0.5 * delta_t**2 * y_noise_t_minus_1
                x_velocity_t = x_velocity_t_minus_1 + delta_t *  x_noise_t_minus_1
                y_velocity_t = y_velocity_t_minus_1 + delta_t *  x_noise_t_minus_1

                where delta_t represents the change in time. In our case, the time between video frames. 

                This can be represented in the matrix form as follows:
                [x_position_t]     [1,    0   ,     delta_t   , 0      ]   [x_position_t_minus_1]       [0.5 * delta_t**2 * x_noise_t_minus_1]
                [y_position_t]  =  [0,    1   ,       0       , delta_t]   [y_position_t_minus_1]   +   [0.5 * delta_t**2 * y_noise_t_minus_1]
                [x_velocity_t]     [0,    0   ,       1       , 0      ]   [x_velocity_t_minus_1]       [delta_t * x_noise_t_minus_1]
                [y_velocity_t]     [0,    0   ,       0       , 1      ]   [y_velocity_t_minus_1]       [delta_t * y_noise_t_minus_1]
                
                This can finally be written as: x_t = F_t_minus_1 * x_t_minus_1 + w_t_minus_1      where    w_t_minus_1 represents the noise (the last matrix of the equation above)
                                                                                                            F_t_minus_1 is given as a transition matrix (first matrix in the above matrix equation)

                We can obtain the process noise matrix Q by taking the covariance of the W_t_minus_1 matrix:

                                                                        [1/4 * delta_t**4 * variance_x**2,                  0             ,1/2 * delta_t**3 * variance_x**2 ,                0               ]
                                                                        [               0                ,1/4 * delta_t**4 * variance_y**2,                 0 ,             1/2 * delta_t**3 * variance_y**2 ]
                Q = COV(W_t_minus_1) = E[W_t_minus_1 * W_t_minus_1.T] = [1/2 * delta_t**3 * variance_x**2,                  0             ,      delta_t**2 * variance_x**2 ,                0               ]
                                                                        [               0                ,1/2 * delta_t**3 * variance_y**2,                 0 ,                   delta_t**2 * variance_y**2 ]

                Measurement matrix is:

                    z_x = x_t + u_x_t       
                    z_y = y_t + u_y_t       where u represents the measurement noises at x and y location.

                This can be represented as:
                                            [x_position_t]
                [z_x]   =    [1,0,0,0]   *  [y_position_t]  +  [u_x_t]
                [z_y]        [0,1,0,0]      [x_velocity_t]     [u_y_t]
                                            [y_velocity_t]

                                            where the first matrix on the right hand side is H (measurement transformation matrix)
                                                  the last matrix is the mesurement variance matrix U.
                
                From U vector we can get the R matrix (covariance matrix of measurement):

                R = COV(U) = E[U * U.T] =   [u_x_t ** 2,      0    ]
                                            [   0      , u_y_t ** 2]


            The covariance extrapolation equation:
                P_t = F * P_t * F.T + Q_t_minus_1         where     P_t_minus_1 is the estimate uncertainty (covariance) matrix of the current state
                                                                    P_t is the predicted estimate uncertainty (covariance) matrix for the next sate
                                                                    F is the state transition matrix
                                                                    Q_t_minus_1 is the process noise matrix

            Formulas for the iterations:

            Measurements:
                z = H * x_t_minus_1 + V_t_minus_1
            
            Updates:
                Kalman Gain:
                    K_t_minus_1 = P_t_minus_1 * H.T * (H * P_t_minus_1 * H.T + R)**(-1)
                Current State Estimate:
                    x_t_minus_1 = x_t_minus_1 (given x_t_minuns_2) + K * (z_t_minus_1 - H * x_t_minus_1 (given x_t_minuns_2))       
                                    where   x_t_minus_1 (given x_t_minuns_2) represents last estimate or the initial estimate
                Update Current State Uncertainty:
                    P_t_minus_1 = (I - K_t_minus_1 * H) * P_t_minus_1 * (I - K_t_minus_1 * H).T + K_t_minus_1 * R * K_t_minus_1.T
            
            Prediction:
                x_t = F * x_t_miuns_1 + w_t_minus_1
                P_t = F * P_t_minus_1 * F.T + Q
            
'''

from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys
sys.path.append("/usr/local/home/mt3qb/Desktop/Jetbot/Kalman_Filters/object_detection")
for i in sys.path:
    print(i)
from object_detection7 import *
import time

# State Matrix initialization
# Initialize the vector x which is a 6 x 1 matrix
# We don't know the vehicle location; we will set initial position, velocity and acceleration to 0.
#x_init = [[0 for x in range(6)] for x in range(1)]
x_init = np.zeros((4, 1))

print("Initial State Matrix:")
print(x_init, end="\n\n")

# State Covariance Matrix (Error in the estimate)
# Since our initial state vector is a guess, we will set a very high estimate uncertainty. The high estimate uncertainty results in a high Kalman Gain, giving a high weight to the measurement.
# This will be a 4 x 4 matrix of  
P_init = np.zeros((4,4))

# Change the values on the diagonal of the State Covariance Matrix
for i in range(4):
    P_init[i][i] = 300

print("Initial State Covariance Matrix P (Error in the estimate matrix):")
for xi in P_init:
    print(xi)
print("\n\n")

# delta_t represents the time difference between the frames (time between taking the images via camera).
# This time is gotten by the equation 1/FPS, where FPS stands for frames per second and has value of 12.4 on the cameras we use. 

#FPS = 12.4
#delta_t = 1 / FPS
delta_t = 0.5  # we are setting it to 0.5 second because we used time.sleep(2) in the for loop when iterating in order to have some reasonable delta_t

# Create the F matrix that represents the state transition matrix.
F = np.zeros((4,4))
for i in range(4):
    F[i][i] = 1
F[0][2] = delta_t
F[1][3] = delta_t


print("State transition matrix F:")
for xi in F:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f} ".format(xi[0],xi[1],xi[2],xi[3]))
print("\n\n")

# Define R. For that we need the standard deviation of x and y.
# for testing purposes we will let x and y position estimate error be 3 meters
x_estimate_error, y_estimate_error = 0.25, 0.25

# Create the Q matix that represents the process noise convariance
Q = np.zeros((4,4))

Q[0][0]= delta_t**4 / 4 * x_estimate_error**2
Q[0][2]= delta_t**3 / 2 * x_estimate_error**2
Q[1][1]= delta_t**4 / 4 * y_estimate_error**2 
Q[1][3]= delta_t**3 / 2 * y_estimate_error**2



Q[2][0]= delta_t**3 / 2 * x_estimate_error**2
Q[2][2]= delta_t**2 * x_estimate_error**2
Q[3][1]= delta_t**3 / 2 * y_estimate_error**2
Q[3][3]= delta_t**2 * y_estimate_error**2


print("Process noise convariance matrix Q:")
for xi in Q:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}   ".format(xi[0],xi[1],xi[2],xi[3]))
print("\n\n")





# To predict the next state, we need to predict the next state covariance matrix P_t-1 = F*P_init*F.t + Q of the initial state.

P = np.dot(np.dot(F, P_init),F.T) + Q

print("State covariance matrix P_t-1:")
for xi in P:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f}".format(xi[0],xi[1],xi[2],xi[3]))
print("\n\n")

# Observation Transition Martix H must be 2 x 6 since Z is 2 x 1 and X is 6 x 1
H = np.zeros((2, 4))
H[0][0] = 1
H[1][1] = 1

print("Observation Transition Martix H")
for xi in H:
    print("{:.4f}   {:.4f}   {:.4f}   {:.4f} ".format(xi[0],xi[1],xi[2],xi[3]))
print("\n\n")

#define small measurement errors (right now it should be aound 0.2 of the foot since we are running stop and go mode)
x_measurement_error = 0.2
y_measurement_error = 0.2

# Measurement error
U = np.zeros((2,1))
U[0][0] = x_measurement_error
U[1][0] = y_measurement_error

# Measurement matrix
z = H @ x_init + U


# Measurement Covariance Matrix
R = np.zeros((2,2))
R[0][0] = x_measurement_error ** 2
R[1][1] = y_measurement_error ** 2

print("measurement uncertainty R")
for xi in R:
    print("{:.0f}   {:.0f}  ".format(xi[0],xi[1]))
print("\n\n")


#used for plotting of predictions versus measurements
x_predict = []
y_predict = []
x_measure_list = []
y_measure_list = []

first_time = True
start = time.time()
#Start iterations (35 for now):
for i in range(5):

    print("ITERATION NUMBER: ",i+1)
    # Measure ( x and y value) Z = H * x + U
    z = np.zeros((2,1))
    
    end = time.time()
    print("Delta_t should be: ", time.strftime("%H:%M:%S",time.gmtime(end-start)))
    start = time.time()
    # Run the detection script to get the x and y measurements
    x_m, y_m = record()
    x_measure_list.append(x_m)
    y_measure_list.append(y_m)
    z[0][0] = x_m
    z[1][0] = y_m
    print(z)
    # Update:
        # Kalman Gain
    if first_time:
        first_time = False
        x_prev = x_init
    else:
        P = P_new_error # if it is not the initial case, update the P matrix with the one from the previous iteration.
        x_prev = x_future # same for state X

    #K = np.dot(np.dot(P,H.T), inv((np.dot(np.dot(H,P),H.T) + R)) ) # this results in a 4 x 2 matrix
    K = P @ H.T @ inv( (H @ P @ H.T) + R )
        # Estimate the current estimate
    print("K:")
    print(K.shape)
    for xi in K:
        print("{:.4f}   {:.4f}".format(xi[0],xi[1]))
    print("\n\n")

    #x_curr = x_prev + np.dot(K, z - np.dot(H, x_prev)) #results in 4 x 1 matrix
    x_curr =  x_prev + (K @ (z - (H @ x_prev)))
    print(x_curr.shape)  
    print("x:")
    for xi in x_curr:
        print("{:.4f}".format(xi[0]))
    print("\n\n")
        # Update Current estimate uncertainty (error)
    P = np.dot(np.dot((np.identity(4) - np.dot(K,H)), P),(np.identity(4)-np.dot(K,H)).T) + np.dot(np.dot(K,R),K.T) # results in a 4 x 4 matrix
    print("P:")
    print(P.shape)
    for xi in P:
        print("{:.4f}   {:.4f}   {:.4f}   {:.4f} ".format(xi[0],xi[1],xi[2],xi[3]))
    print("\n\n")

    # Predict
    x_future = np.dot(F,x_curr) # results in a 4 x 1 matrix

    #store those values
    print(x_future)
    x_predict.append(x_future[0][0])
    y_predict.append(x_future[1][0])

    P_new_error = np.dot(np.dot(F,P),F.T) + Q #results in a 4 x 4 matrix
    print("P_new:")
    for xi in P_new_error:
        print("{:.4f}   {:.4f}   {:.4f}   {:.4f} ".format(xi[0],xi[1],xi[2],xi[3]))
    print("\n\n")

    #introducing delta_t
    time.sleep(0.5)

print(x_predict)
print(y_predict)
print(x_m)
print(y_m)

x_predict = [x * (-1) for x in x_predict]
x_measure_list = [x * (-1) for x in x_measure_list]
plt.plot(x_predict, y_predict, label="Prediction")
plt.plot(x_measure_list , y_measure_list, label="Measure")
plt.legend()

plt.show()
