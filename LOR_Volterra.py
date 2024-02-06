import datagen.data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import process_time 

import methods.volt_funcs as volt

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

data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)[1]
data = data[::slicing]

# %% Prepare dataset for training and testing -- standardisation

# Define full data training and testing sizes
ndata  = len(data)
ntrain = 5000 
washout = 1000
ntest = ndata - ntrain

# Construct training input and teacher, testing input and teacher
training_input_orig = data[0:ntrain-1] 
training_teacher_orig = data[1:ntrain]
testing_input_orig = data[ntrain-1:ntrain+ntest-1]
testing_teacher_orig = data[ntrain:ntrain+ntest]

# Shifting of dataset so that it is centered at 0 
training_input_orig_mean = np.mean(training_input_orig, axis=0)

training_input_shifted = training_input_orig - training_input_orig_mean
training_teacher_shifted = training_teacher_orig - training_input_orig_mean
testing_input_shifted = testing_input_orig - training_input_orig_mean
testing_teacher_shifted = testing_teacher_orig - training_input_orig_mean

# Scaling of dataset so that max norm is 1
max_norm = np.max([np.linalg.norm(z) for z in training_input_shifted])
scale = 1/max_norm
training_input = scale * training_input_shifted
training_teacher = scale * training_teacher_shifted
testing_input = scale * testing_input_shifted
testing_teacher = scale * testing_input_shifted

# %% Define reservoir kernel parameters

# Define parameters needed to build the reservoir
M = np.max([np.linalg.norm(z) for z in training_input])
tau = np.sqrt(1 / M**2)
tau_val_coef = 0.2
tau = tau * tau_val_coef
ld_val_coef = 0.8
ld = np.sqrt(1 - (tau**2) * (M**2)) * ld_val_coef

# Define regularisation parameter for training
reg = 1e-10 

# Check parameter values
print("Value of tau: ", tau)
print("Value of lambda: ", ld)
print("Value of regularisation: ", reg)
print("M: ", M)
print("ld/np.sqrt(1-tau^2 M^2): ", ld/np.sqrt(1-tau**2 * M**2))

# %% Perform training on full training set by populating Gram matrix then using least squares 

# Start timing training and filling of feature vector
train_start = process_time()

# Perform training
alpha_ols, alpha0_ols, K = volt.Train(training_input, training_teacher, washout, ld, tau, reg, pinv=False)

# End timing of training
train_stop = process_time()

# Print training specs
print("Time taken to perform training: ", train_stop - train_start, "seconds")

# Show intercept value
print("Regression intercept: ", alpha0_ols)
# Generate plot of regression coefficients
fig, axs = plt.subplots(3)
# Plot x dimension regression coefficients
axs[0].plot(alpha_ols[:, 0], lw=0.7)
# Plot y dimension regression coefficients
axs[1].plot(alpha_ols[:, 1], lw=0.7)
# Plot z dimension regression coefficients
axs[2].plot(alpha_ols[:, 2], lw=0.7)

plt.show()

# %% Perform nonautonomous forecasting using alpha coefficients and testing input data

# Start timing nonauto forecasting
forecast_start = process_time()

# Perform nonauto forecasting
pred_nonauto = volt.Forecast(K, training_input, testing_input, alpha_ols, alpha0_ols, washout, ld, tau)  

# Stop nonauto forecasting timing
forecast_end = process_time()

# Compute and print the time taken to forecast autonomously
print("Time taken to perform nonauto forecasting: ", forecast_end - forecast_start, "seconds")

# Compute the mse for nonautonomous forecasting compared with testing teacher
mse_nonauto = np.mean((testing_teacher - pred_nonauto)**2) 

# Compute mse for nonauto after rescaling back for comparison with other methods
pred_nonauto_orig = (1/scale) * pred_nonauto + training_input_orig_mean
mse_nonauto_orig = np.mean((testing_teacher_orig - pred_nonauto_orig)**2)

print("Nonautonomous forecasting error: ", mse_nonauto_orig)

# %% Plot and check nonautonomous forecasting -- for only the testing teacher portion of the data

ax = plt.figure().add_subplot(projection='3d')
ax.tick_params(labelsize=7)

# Plot original data in blue
ax.plot(*testing_teacher.T, lw=0.7)
# Plot nonautonomous forecasting in red dashed lines
ax.plot(*pred_nonauto.T, lw=0.7, color='red', linestyle='dashed')
# Set labels for each dimension
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()

# %% Plot individual axes separately in a subfigure plot -- nonautonmous forecasting

fig, axs = plt.subplots(3)

# Plot x axis testing teacher in blue and forecasting in red dashed lines 
axs[0].plot(testing_teacher.T[0], lw=0.7)
axs[0].plot(pred_nonauto.T[0], lw=0.7, color='red', linestyle='dashed')
# Plot y axis testing teacher in blue and forecasting in red dashed lines 
axs[1].plot(testing_teacher.T[1], lw=0.7)
axs[1].plot(pred_nonauto.T[1], lw=0.7, color='red', linestyle='dashed')
# Plot z axis testing teacher in blue and forecasting in red dashed lines 
axs[2].plot(testing_teacher.T[2], lw=0.7)
axs[2].plot(pred_nonauto.T[2], lw=0.7, color='red', linestyle='dashed')

plt.show() 

# %% Perform autonomous forecasting on the testing set

# Start timing auto forecasting
forecast_start = process_time()

#  Assign the last training teacher value as the most recent input
latest_input = training_teacher[-1]
# Perform autonomous forecasting which iteratively generates Gram using each forecast
pred_auto = volt.ForecastAuto(K, training_input, latest_input, alpha_ols, alpha0_ols, washout, ntest, ld, tau)

# Stop auto forecasting timing
forecast_end = process_time()

# Compute and print the time taken to forecast autonomously
print("Time taken to perform auto forecasting: ", forecast_end - forecast_start, "seconds")

# Compute the mse for autonomous and testing teacher
mse_auto = np.mean((testing_teacher - pred_auto)**2)

# Compute mse for auto after rescaling to compare with other methods
pred_auto_orig = (1/scale) * pred_auto + training_input_orig_mean
mse_auto_orig = np.mean((testing_teacher_orig - pred_auto_orig)**2)

print("Autonomous forecasting error: ", mse_auto_orig)

# %% Plot and check autonomous forecasting -- only for testing teacher portion

ax = plt.figure().add_subplot(projection='3d')
ax.tick_params(labelsize=7)

# Plot original data in blue
ax.plot(*testing_teacher.T, lw=0.7)
# Plot nonautonomous forecasting in red dashed lines
ax.plot(*pred_auto.T, lw=0.7, color='red', linestyle='dashed')
# Set labels for each dimension
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()

# %% Plot individual axes separately in a subfigure plot -- autonomous forecasting

fig, axs = plt.subplots(3)

# Plot x axis testing teacher in blue and forecasting in red dashed lines 
axs[0].plot(testing_teacher.T[0], lw=0.7)
axs[0].plot(pred_auto.T[0], lw=0.7, color='red', linestyle='dashed')
# Plot y axis testing teacher in blue and forecasting in red dashed lines 
axs[1].plot(testing_teacher.T[1], lw=0.7)
axs[1].plot(pred_auto.T[1], lw=0.7, color='red', linestyle='dashed')
# Plot z axis testing teacher in blue and forecasting in red dashed lines 
axs[2].plot(testing_teacher.T[2], lw=0.7)
axs[2].plot(pred_auto.T[2], lw=0.7, color='red', linestyle='dashed')

plt.show()

# %% Plot distributions of testing values 

fig, axs = plt.subplots(3)
fig.tight_layout(pad=2)

# Plot distribution of testing teacher in blue and forecasting in red for each axes
sns.kdeplot(testing_teacher.T[0], color="blue", fill=True, ax=axs[0])
sns.kdeplot(pred_auto.T[0], color="red", fill=True, ax=axs[0])
sns.kdeplot(testing_teacher.T[1], color="blue", fill=True, ax=axs[1])
sns.kdeplot(pred_auto.T[1], color="red", fill=True, ax=axs[1])
sns.kdeplot(testing_teacher.T[2], color="blue", fill=True, ax=axs[2])
sns.kdeplot(pred_auto.T[2], color="red", fill=True, ax=axs[2])

axs[0].set(xlabel="x")
axs[1].set(xlabel="y")
axs[2].set(xlabel="z")
