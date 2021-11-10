# Kalman_Filters
Different examples for applying Kalman Filers using Python.

![alt tag](https://raw.githubusercontent.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/animations/05_dog_track.gif)

# Introduction

Kalman Filter is treated as linear time-invariant (LTI) filter that is used to smoothen the noisy data and detrend the time-series signals. It can be considered as an optimization problem with objective of minimizing the smoothness of X_t (future state estimate) and minimizing the residual error between teh actual and smooth series. 

The process of estimating X_t is called **filtering or smoothing.**

The advantage of Kalman's Filter over [Wiener's Filter](https://en.wikipedia.org/wiki/Wiener_filter) is that the Kalman's Filter does not use all the previous states in order to predict the future state. It only uses the information of the previous state. This recursive solution is therefore much less computationally expensive.

[My Notes on Kalman Filter](https://nikolaandro.github.io/kalman-filter/)

# Robot Position Prediction (Basics)

  In this one dimensional example we are using Kalman Filer approach to predict the next position of the moving robot.

# Car Location Prediction 2D - Constant Acceleration Model

  In this example we are predicting the position of a caar that has a constant acceleration. We have fixed lists of x and y measurements of the car in the 2D space. The rest of the equations are being derived. Finally, we plot the graph to see how the predictions become more acurate over time. 

# Jetbot Location Prediction - Constant Velocity Model

  (Code is stored in jetbot_Kalman folder)
  
  In this example, we will present 2D Kalman Filter while trying to predict the next position of a Jetbot on a test-pad (imitation of a road) using a camera for jetbot detection and getting the x and y measurements. Those measurements are imported into Kalman Filter and the predidction is being made.  

  There are many different motion models that can be bubilt for Kalman Filtering such as Constant Velocity (CV), Constant Acceleration (CA), Constant Turn (CT), Random Walk (RW) and many others. In this case we will use the constant velocity since the Jetbot is moving with a constant velocity. 
  
  

 <!-- 
 ### DISCRETE NOISE MODEL
 
  The discrete noise model assumes that the noise is different at each time period, but it is constant between time periods.
  
 ![discrete noise](./images/discrete_noise.png)
 
   -->
  
  
