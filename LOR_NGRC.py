import datagen.data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
from time import process_time 

import estimators.ngrc_funcs as ngrc

# %% Generate Lorenz dataset

def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)

h = 0.005
t_span = (0, 40)
slicing = int(h/h)

t_eval, data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)
t_eval = t_eval[::slicing]
data = data[::slicing].T

# %% Prepare dataset for training and testing

# Define full data training and testing sizes
ndata  = data.shape[1]
ntrain = 5000 
washout = 1000
ntest = ndata - ntrain

# Construct training input and teacher, testing input and teacher
training_input = data[:, 0:ntrain-1] 
training_teacher = data[:, 1:ntrain]
testing_input = data[:, ntrain-1:ntrain+ntest-1]
testing_teacher = data[:, ntrain:ntrain+ntest]

# %% Define dimensions of linear, nonlinear and total feature vector

# Input dimension 
ndim = 3
# Number of time delays
ndelay = 2
# Ridge regression parameter
reg = 1e-4
# Define choice of highest degree of monomial 
deg = 2

# %% Fill in feature vector as a matrix

# Start timing the training and filling of feature vector
train_start = process_time()

# Fills linear and nonlinear feature vector then trains using least squares
W_out, X = ngrc.Train(ndelay, deg, reg, data, ntrain, washout)

# Stop the timer for training and filling of feature vector
train_stop = process_time()

# Compute and print training time
print("Time taken to perform training: ", train_stop - train_start, "seconds")

# %% Perform prediction by generating a new feature vector

# Start timing forecasting 
forecast_start = process_time()

# Fills in linear and nonlinear feature vector autonomously and generates prediction
prediction = ngrc.Forecast(W_out, X, deg, ntest)

# Compute mse of prediction
mse = np.mean((prediction - testing_teacher)**2)
print("Error for NG-RC auto forecast: ", mse)

# Stop timer for forecasting
forecast_end = process_time()

# Compute and print the time taken to forecast autonomously
print("Time taken to perform auto forecasting: ", forecast_end - forecast_start, "seconds")

# %% Check plot of predictions -- 3D

ax = plt.figure().add_subplot(projection='3d')
ax.tick_params(labelsize=7)

# Plot original data in blue
ax.plot(*testing_teacher, lw=0.7)
# Plot nonautonomous forecasting in red dashed lines
ax.plot(*prediction, lw=0.7, color='red', linestyle='dashed')
# Set labels for each dimension
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()

# %% Check plots of predictions -- separate axes

fig, axs = plt.subplots(3)

# Plot x axis testing teacher in blue and forecasting in red dashed lines 
axs[0].plot(testing_teacher[0], lw=0.7)
axs[0].plot(prediction[0], lw=0.7, color='red', linestyle='dashed')
# Plot y axis testing teacher in blue and forecasting in red dashed lines 
axs[1].plot(testing_teacher[1], lw=0.7)
axs[1].plot(prediction[1], lw=0.7, color='red', linestyle='dashed')
# Plot z axis testing teacher in blue and forecasting in red dashed lines 
axs[2].plot(testing_teacher[2], lw=0.7)
axs[2].plot(prediction[2], lw=0.7, color='red', linestyle='dashed')

plt.show() 

# %% Plot distributions of testing values 

fig, axs = plt.subplots(3)
fig.tight_layout(pad=2)

# Plot distribution of testing teacher in blue and forecasting in red for each axes
sns.kdeplot(testing_teacher[0], color="blue", fill=True, ax=axs[0])
sns.kdeplot(prediction[0], color="red", fill=True, ax=axs[0])
sns.kdeplot(testing_teacher[1], color="blue", fill=True, ax=axs[1])
sns.kdeplot(prediction[1], color="red", fill=True, ax=axs[1])
sns.kdeplot(testing_teacher[2], color="blue", fill=True, ax=axs[2])
sns.kdeplot(prediction[2], color="red", fill=True, ax=axs[2])

axs[0].set(xlabel="x")
axs[1].set(xlabel="y")
axs[2].set(xlabel="z")

    