import datagen.data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt
from time import process_time, time
import utils.crossvalidation as cv

import estimators.volt_funcs as volt

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

# %% Define pre-requisite terms needed for the reservoir kernel parameters

# Define parameters needed to build the reservoir
normed_data = scale * (data - training_input_orig_mean)
M = np.max([np.linalg.norm(z) for z in normed_data])

# %% cv parallel

start_time = time()
best_params, combinations = cv.find_best_hyperparameters(training_input_orig, training_teacher_orig, reg_params=np.logspace(-10, -4, num=2), tau_values=np.linspace(1e-4,1/M*0.99,2), M=M, nwashout=washout, num_processes=22)
end_time = time()
print(end_time - start_time)
# %% Rolling window cross-validation for optimal lambda -- auto forecasting
        
# Define range of ld and tau to test over
ld_coeff_range = np.linspace(0.1, 0.9, 9)
tau_coeff_range = np.linspace(0.1, 0.9, 9)

# Define the cross-validation process parameters
nblock = 5
block_len = int((ntrain - washout)/nblock)
check_in_period = 10

# Define a store for the mse values
mse_dict_gram = {}

# Define regularisation parameter to use 
reg = 1e-10

# Start timing cross-validation process
cv_start = process_time()

# Perform rolling-window cross validation over both parameters
iter_count = 0
for ld_coeff_try in ld_coeff_range:
    for tau_coeff_try in tau_coeff_range:
        
        # Needed to be able to easily access the parameter used because for loops sometimes have rounding errors
        ld_coeff_try = round(ld_coeff_try, 2)
        tau_coeff_try = round(tau_coeff_try, 2)
        
        # Define the values of tau and ld being used in forecasting
        tau_try = np.sqrt(1 / M**2) * tau_coeff_try
        ld_try = np.sqrt(1 - (tau_try**2) * (M**2)) * ld_coeff_try
        
        # Record the values of ld, tau coeffs being tried, one block is reserved for the final testing block
        mse_dict_gram[(ld_coeff_try, tau_coeff_try)] = np.zeros((nblock-1, ))
        
        # Iterate through each of the block pairs
        for block in range(nblock-1):
           
           # Define training indices for each block, letting the last block run over the remainder
           training_start = washout + block * block_len
           training_end =  training_start + block_len 
           testing_start = training_end
           if block == nblock - 2:
               testing_end = -1
           else: testing_end = testing_start + block_len
           
           # Define each data using the normalised training input and teacher data already defined
           training_input_block = training_input[0:training_end]
           training_teacher_block = training_teacher[0:training_end]
           testing_input_block = training_input[testing_start:testing_end]
           testing_teacher_block = training_teacher[testing_start:testing_end]
           
           # Perform training on training input block using input and training teacher block
           alpha_ols_try, alpha0_ols_try, K_try = volt.Train(training_input_block, training_teacher_block, training_start, ld_try, tau_try, reg)
           
           # Define the most recent input to path continue from and the block testing horizon, then forecast
           latest_input_block = training_teacher_block[-1]
           ntest_block = testing_input_block.shape[0]
           pred_auto_try = volt.ForecastAuto(K_try, training_input_block, latest_input_block, alpha_ols_try, alpha0_ols_try, training_start, ntest_block, ld_try, tau_try)
           
           # Compute the mse for the testing block
           mse_try = np.mean((testing_teacher_block - pred_auto_try)**2)
           mse_dict_gram[(ld_coeff_try, tau_coeff_try)][block] = mse_try
           
        # Check in at every predefined period
        if iter_count % check_in_period == 0:
           print((ld_coeff_try, tau_coeff_try), ": ", mse_dict_gram[(ld_coeff_try, tau_coeff_try)])

        iter_count = iter_count + 1
       
# End timing cross-validation process
cv_end = process_time()

print("Cross validating over ", len(ld_coeff_range)*len(tau_coeff_range), " hyperpameters took ", cv_end-cv_start, " seconds.")

 # %% Find the minimum MSE and the parameters ld and tau that give this mse

# Define a list to store the average errors over all blocks
avg_error_ls = []
# Define a large number as the minimum error
min_error = np.infty
# Define a variable to store the smallest ld, tau found
min_ld_tau = None
# Iterate through the dictionary
for key in mse_dict_gram:
    # Average the error over each block then append the average
    avg_error = np.mean(mse_dict_gram[key])
    avg_error_ls.append(avg_error)
    # Determine if a new min error has been found then store it
    if avg_error < min_error:
        min_error = avg_error
        min_ld_tau = key

print(min_ld_tau, ": ", min_error)

# %% Plot mses using a ld-tau contour plot

fig, ax = plt.subplots(nblock-1)

# Required to be able to universally access each data type
n_ld_range = len(ld_coeff_range)
n_tau_range = len(tau_coeff_range)
mse_array_gram = np.zeros((n_ld_range, n_tau_range))

levels = n_ld_range

# Iterate through each of the block pairs
for block in range(nblock-1):
    
    # Rewrite the errors and ld, tau pairs for contour plots
    for ld_id in range(n_ld_range):
        for tau_id in range(n_tau_range):
            ld_val = round(ld_coeff_range[ld_id], 2)
            tau_val = round(tau_coeff_range[tau_id], 2)
            mse_array_gram[ld_id, tau_id] = mse_dict_gram[(ld_val, tau_val)][block]
    
    # Generate the contour plot
    cs = ax[block].contourf(ld_coeff_range, tau_coeff_range, mse_array_gram, cmap="RdBu_r", levels=levels)
    ax[block].set_xlabel(r"$\lambda$")
    
fig.supylabel(r"$\tau$")
fig.colorbar(cs, ax=ax)    
plt.show()
    
# %% Rolling cross-validation for optimal regularisation -- auto forecasting

# Define range of regularisation parameter to test over
reg_range = np.logspace(-15, -1, 15)

# Define the cross-validation process parameters
nblock = 2
block_len = 3000
check_in_period = 1

# Define a store for the mse values
mse_dict_reg = {}

# Define the values of tau and ld being used in forecasting
tau = np.sqrt(1 / M**2)
tau_val_coef = 0.2 
tau = tau * tau_val_coef
ld_val_coef = 0.8
ld = np.sqrt(1 - (tau**2) * (M**2)) * ld_val_coef

# Start timing cross-validation process
cv_start = process_time()

# Perform rolling-window cross validation varying ld
iter_count = 0
for reg_try in reg_range:
    
    # Record the values of ld, tau coeffs being tried
    mse_dict_reg[reg_try] = np.zeros((nblock-1, ))
    
    for block in range(nblock-1):
       
       # Define training indices for each block, letting the last block run over the remainder
       training_start = washout + block * block_len
       training_end =  training_start + block_len 
       testing_start = training_end
       if block == nblock - 2:
           testing_end = -1
       else: testing_end = testing_start + block_len
       
       # Define each data using the normalised training input and teacher data already defined
       training_input_block = training_input[0:training_end]
       training_teacher_block = training_teacher[0:training_end]
       testing_input_block = training_input[testing_start:testing_end]
       testing_teacher_block = training_teacher[testing_start:testing_end]
       
       # Perform training on training input block using input and training teacher block
       alpha_ols_try, alpha0_ols_try, K_try = volt.Train(training_input_block, training_teacher_block, training_start, ld, tau, reg_try)
       
       # Define the most recent input to path continue from and the block testing horizon, then forecast
       latest_input_block = training_teacher_block[-1]
       ntest_block = testing_input_block.shape[0]
       pred_auto_try = volt.ForecastAuto(K_try, training_input_block, latest_input_block, alpha_ols_try, alpha0_ols_try, training_start, ntest_block, ld, tau)
       
       # Compute the mse for the testing block
       mse_try = np.mean((testing_teacher_block - pred_auto_try)**2)
       mse_dict_reg[reg_try][block] = mse_try

    # Check in at every predefined period
    if iter_count % check_in_period == 0:
       print(reg_try, ": ", mse_dict_reg[reg_try])

    iter_count = iter_count + 1

# End timing cross-validation process
cv_end = process_time()

print("Cross validating over ", len(reg_range), " hyperpameters took ", cv_end-cv_start, "seconds.")

 # %% Find the minimum value MSE

# Define a list to store the average errors over all blocks
mse_reg_ls = []
# Define a large number as the minimum error
min_error = np.infty
# Define a variable to store the smallest reg parameter found
min_reg = None
# Iterate through the dictionary
for key in mse_dict_reg:
    # Average the error over each block then append the average
    avg_error = np.mean(mse_dict_reg[key])
    mse_reg_ls.append(avg_error)
    # Determine if a new min error has been found then store it
    if avg_error < min_error:
        min_error = avg_error
        min_reg = key

print(min_reg, ": ", min_error)

# Convert to array to be able to save as a file
mse_reg_array = np.array(mse_reg_ls)     

# %% Plot the MSE values for each block with respect to reg

# Iterate through each block pair
for block in range(nblock-1):
    # Create a store to store the blocks particular error
    mse_block_ls = []
    # Create a store for the regularisation value used
    reg_ls = []
    # Iterate through the keys of the dictionary
    for key in mse_dict_reg:
        # Collect the error for the block and key
        error = mse_dict_reg[key][block]
        # Append it only if it is sufficiently small so it does not affect the rest of the graph
        if error <= 10:
            mse_block_ls.append(error)
        else: mse_block_ls.append(np.nan)   # nan is ignored  when plotting
        # Append the key
        reg_ls.append(key)
    plt.scatter(reg_ls, mse_block_ls, s=5)

plt.show()