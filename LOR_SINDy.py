# %% Import packages

import datagen.data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import process_time 
from sklearn.metrics import mean_squared_error

import pysindy as ps

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
data = data[::slicing]

# %% Prepare dataset for training and testing

# Define full data, training and testing sizes
ndata = len(data)
ntrain = 5000
ntest = ndata - ntrain
washout = 1000

# Construct training and testing datasets
training = data[0:ntrain]
testing = data[ntrain:]

# Construct time evaluation as needed by solving in SINDy
training_t_eval = t_eval[0:ntrain]
testing_t_eval = t_eval[ntrain:]

# %% Instantiate and fit the SINDy model

# Start timing the training and filling of feature vector
train_start = process_time()

# Model creation and fitting
threshold = 0.1
alpha = 1e-15
stlsq_optim = ps.STLSQ(threshold=threshold, alpha=alpha)
library = ps.PolynomialLibrary(degree=2, include_interaction=True, interaction_only=False)
sindy_model = ps.SINDy(optimizer=stlsq_optim, feature_library=library)
sindy_model.fit(training, t=h*slicing)

# Stop timer for training
train_end = process_time()

# Print time elapsed for training
print("Time taken to perform training: ", train_end - train_start, "seconds")

# Print model that was fitted by SINDy
sindy_model.print()

# %% Simulate forward in time using the given initial condition

# Start timing forecasting 
forecast_start = process_time()

# Forecasting by solving ivp starting from test value[0] over testing teacher time
pred_sindy = sindy_model.simulate(testing[0], testing_t_eval)

# Stop timer for forecasting
forecast_end = process_time()

# Compute mse for autonomous forecasting
mse = mean_squared_error(testing, pred_sindy)
print("Error for PySINDy auto forecasting: ", mse)

# Compute and print the time taken to forecast autonomously
print("Time taken to perform auto forecasting: ", forecast_end - forecast_start, "seconds")

# %% Plot testing and simulated values by SINDy -- 3D

ax = plt.figure().add_subplot(projection='3d')
ax.tick_params(labelsize=7)

# Plot original data in blue
ax.plot(*testing.T, lw=0.7)
# Plot nonautonomous forecasting in red dashed lines
ax.plot(*pred_sindy.T, lw=0.7, color='red', linestyle='dashed')
# Set labels for each dimension
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()

# %% Plot testing and simulated values by SINDy -- separate axes

fig, axs = plt.subplots(3)

# Plot x axis testing teacher in blue and forecasting in red dashed lines 
axs[0].plot(testing.T[0], lw=0.7)
axs[0].plot(pred_sindy.T[0], lw=0.7, color='red', linestyle='dashed')
# Plot y axis testing teacher in blue and forecasting in red dashed lines 
axs[1].plot(testing.T[1], lw=0.7)
axs[1].plot(pred_sindy.T[1], lw=0.7, color='red', linestyle='dashed')
# Plot z axis testing teacher in blue and forecasting in red dashed lines 
axs[2].plot(testing.T[2], lw=0.7)
axs[2].plot(pred_sindy.T[2], lw=0.7, color='red', linestyle='dashed')

plt.show() 

# %% Plot distributions of testing values 

fig, axs = plt.subplots(3)
fig.tight_layout(pad=2)

# Plot distribution of testing teacher in blue and forecasting in red for each axes
sns.kdeplot(testing.T[0], color="blue", fill=True, ax=axs[0])
sns.kdeplot(pred_sindy.T[0], color="red", fill=True, ax=axs[0])
sns.kdeplot(testing.T[1], color="blue", fill=True, ax=axs[1])
sns.kdeplot(pred_sindy.T[1], color="red", fill=True, ax=axs[1])
sns.kdeplot(testing.T[2], color="blue", fill=True, ax=axs[2])
sns.kdeplot(pred_sindy.T[2], color="red", fill=True, ax=axs[2])

axs[0].set(xlabel="x")
axs[1].set(xlabel="y")
axs[2].set(xlabel="z")