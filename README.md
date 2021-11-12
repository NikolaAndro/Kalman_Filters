# Kalman_Filters
Different examples for applying Kalman Filers using Python.

![alt tag](https://raw.githubusercontent.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/master/animations/05_dog_track.gif)

# Introduction

Kalman Filter is treated as linear time-invariant (LTI) filter that is used to smoothen the noisy data and detrend the time-series signals. It can be considered as an optimization problem with objective of minimizing the smoothness of X_t (future state estimate) and minimizing the residual error between teh actual and smooth series. 

The process of estimating X_t is called **filtering or smoothing.**

The advantage of Kalman's Filter over [Wiener's Filter](https://en.wikipedia.org/wiki/Wiener_filter) is that the Kalman's Filter does not use all the previous states in order to predict the future state. It only uses the information of the previous state. This recursive solution is therefore much less computationally expensive.

[My Notes on Kalman Filter](https://nikolaandro.github.io/kalman-filter/)

***Kalman Filter folder*** contains multiple simple examples of Kalman Filter. Car position prediction (constant velocity or constant acceleration) examples were 
implemented using the formulas from Kalman Filter. Jetbot Kalman Filter folder contains code for predicting the position of a jet-bot in the lab while using 
camera to detect the jet-bot on the road and get its x and y coordinates as the input for measurements.

***Extended Kalman Filtler*** folder is work in progress. 🔭
