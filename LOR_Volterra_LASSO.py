
# %%

import datagen.data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import process_time 
from sklearn.linear_model import Lasso

import estimators.volt_funcs2 as volt

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

# Scaling and shifting of dataset so that it is centered at 0 and stand dev is 1
training_input_orig_mean = np.mean(training_input_orig, axis=0)
training_input_orig_sd = np.std(training_input_orig, axis=0)

scale = 1/training_input_orig_sd
training_input = scale * (training_input_orig - training_input_orig_mean)
training_teacher = scale * (training_teacher_orig - training_input_orig_mean)
testing_input = scale * (testing_input_orig - training_input_orig_mean)
testing_teacher = scale * (testing_teacher_orig - training_input_orig_mean)

# %% 

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

# %% Define reservoir kernel parameters -- without defining regression first

# Define parameters needed to build the reservoir
M = np.max([np.linalg.norm(z) for z in training_input])
tau = np.sqrt(1 / M**2)
tau_val_coef = 0.2
tau = tau * tau_val_coef
ld_val_coef = 0.8
ld = np.sqrt(1 - (tau**2) * (M**2)) * ld_val_coef

# Check parameter values
print("Value of tau: ", tau)
print("Value of lambda: ", ld)
print("M: ", M)
print("ld/np.sqrt(1-tau^2 M^2): ", ld/np.sqrt(1-tau**2 * M**2))

# %% 

M = np.max([np.linalg.norm(z) for z in training_input])
tau = 0.99
omega = 0.14106735979665894

print("Value of tau: ", tau)
print("Value of lambda: ", ld)
print("M: ", M)
print("ld/np.sqrt(1-tau^2 M^2): ", ld/np.sqrt(1-tau**2 * M**2))

# %% Perform training on full training set by populating Gram matrix then using Lasso least squares 
# Iterate over possible Lasso reg parameters to obtain one that gives 28 nonzero coefs 

# Start timer 
n_coefs_start = process_time()

# Define the number of dimensions in dataset
ndim = data.shape[1]

# Define check in period
check_in = 10

# Fill up Gram matrix with training data
K = volt.GramMat(training_input, ld, tau)
# Remove the washout part from the Gram matrix 
K_train = K[washout: , washout: ]
# Remove the washout part from the training teacher data
training_teacher_washed = training_teacher[washout: ]

# Define range of regularisation to try
reg_range = np.logspace(-5, -4, 200) #np.logspace(-9, 0, 100)

# Define store for number of coefficients for each reg and each dimension
n_coef_array = np.zeros((ndim, len(reg_range)))

# Perform lasso least squares regression on washed data per dimension
iter_count = 0
for dim in range(ndim):
    for reg_val_id in range(len(reg_range)):
        
        # Define regularisation parameter being used
        reg_val = reg_range[reg_val_id]
        
        # Define Lasso sklearn model and call fit
        lasso_model_dim = Lasso(reg_val, max_iter=10000, tol=1e-3).fit(K_train, training_teacher_washed[:, dim])
        alpha_lasso_dim, alpha0_lasso_dim = lasso_model_dim.coef_, lasso_model_dim.intercept_
        
        # Determine number of nonzero coefficients then store them
        n_nonzero_coefs = len(np.nonzero(alpha_lasso_dim)[0])
        n_coef_array[dim, reg_val_id] = n_nonzero_coefs
        
        # Check if at fixed period and increment counts
        if iter_count % check_in == 0:
            print("Iteration: ", iter_count)
            print("Current dimension: ", dim)
            print("Current reg id and value: ", reg_val_id, "; ", reg_val)
            print("Current number of nonzero coeffs: ", n_nonzero_coefs)
            
        iter_count = iter_count + 1

# End timer
n_coefs_end = process_time()
print("Amount of time taken to go through ", ndim*len(reg_range), " iterations is ", n_coefs_end - n_coefs_start)

# %% Per dimension plot the number of nonzero coefficients against reg values and check if there is one that hits the desired number

# Define desired number of coeffs
desired_n_coefs = 28

plt.figure(figsize=(5, 10))

# Iterate through each dimension
for dim in range(ndim):
    
    # Retrieve the number of coefficients array per dimension
    n_coef_dim = n_coef_array[dim, :]
    plt.plot(reg_range, n_coef_dim)

    # Iterate through the regularisation constants check if they hit desired
    did_hit = False
    for reg_val_id in range(len(reg_range)):
        if n_coef_array[dim, reg_val_id] == desired_n_coefs:
            print(desired_n_coefs, " found at ", reg_range[reg_val_id], " for dimension ", dim)
            did_hit = True

    # Suppose the desired number of nonzero coefs were not found, the iteratively decrease until find the next largest
    while did_hit == False:
        smaller_than_desired_ncoefs = desired_n_coefs - 1
        # Include a break statement to avoid infinite loop edge case
        if smaller_than_desired_ncoefs == 0:
            print('Hit zero nonzero coefficients')
            break
        # Check if the smaller number of coefficients, find all, then break while loop
        for reg_val_id in range(len(reg_range)):
            if n_coef_array[dim, reg_val_id] == smaller_than_desired_ncoefs:
                print("For dimension ", dim, ", could not find the desired number of nonzero coefs but found ", \
                      smaller_than_desired_ncoefs, "at ", reg_range[reg_val_id])
                did_hit = True

plt.xscale("log")
plt.axhline(y=desired_n_coefs, color='black', lw=0.7, linestyle='dashed')

# %% Run Lasso regression with the given regression parameters -- have to run for each target output separately

# Start timer for lasso training
lasso_training_start = process_time()

# Redefine number of dimensions in case did not run above code
ndim = data.shape[1]

# Redefine the exact same Gram matrix as above
K = volt.GramMat(training_input, ld, tau)
# Remove the washout part from the Gram matrix as above 
K_train = K[washout: , washout: ]
# Remove the washout part from the training teacher data
training_teacher_washed = training_teacher[washout: ]

# Define choice of regularisation parameters for each dimension
#lasso_reg_ls = [2.0729217795953697e-05, 6.90551352016233e-05, 5.672426068491978e-05]
lasso_reg_ls = [3e-06, 3e-06, 3e-06]

# Store for full least squares regression result (alpha matrix)
alpha_lasso = np.zeros((ntrain-1-washout, ndim))
alpha0_lasso = np.zeros((ndim, ))

for dim in range(ndim):
    
    # Define regularisation based on iterative Lasso above
    lasso_reg_dim = lasso_reg_ls[dim]
    
    # Perform Lasso regularisation least squares regression 
    lasso_model_dim = Lasso(lasso_reg_dim, max_iter=10000, tol=1e-3).fit(K_train, training_teacher_washed[:, dim])
    alpha_lasso_dim, alpha0_lasso_dim = lasso_model_dim.coef_, lasso_model_dim.intercept_
    
    # Fill up the full regression parameters with the regression result
    alpha_lasso[:, dim] = alpha_lasso_dim
    alpha0_lasso[dim] = alpha0_lasso_dim
    
# End timer for lasso training
lasso_training_end = process_time()

print("Lasso training took ", lasso_training_end - lasso_training_start, " seconds")

# Check the number of nonzero coefficients generated by Lasso
for dim in range(ndim):
    print("Dimension ", dim, ": ", len(np.nonzero(alpha_lasso[:, dim])[0]))

# Show intercept value
print("Regression intercept: ", alpha0_lasso)
# Generate plot of regression coefficients
fig, axs = plt.subplots(3)
# Plot x dimension regression coefficients
axs[0].plot(alpha_lasso[:, 0], lw=0.7)
# Plot y dimension regression coefficients
axs[1].plot(alpha_lasso[:, 1], lw=0.7)
# Plot z dimension regression coefficients
axs[2].plot(alpha_lasso[:, 2], lw=0.7)

plt.show()

# %% Perform nonautonomous forecasting using the alpha coefficients

# Start timing nonauto forecasting
forecast_start = process_time()

# Perform nonauto forecasting
pred_nonauto = volt.Forecast(K, training_input, testing_input, alpha_lasso, alpha0_lasso, washout, ld, tau)  

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
pred_auto = volt.ForecastAuto(K, training_input, latest_input, alpha_lasso, alpha0_lasso, washout, ntest, ld, tau)

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


# %%
