import datagen.data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt  
from time import process_time 

import methods.ngrc_funcs as ngrc

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
training = data[:, 0:ntrain] 

# %% Define dimensions of linear, nonlinear and total feature vector

# Input dimension 
ndim = 3
# Number of time delays
ndelay = 2
# Ridge regression parameter
reg = 0.224
# Define choice of highest degree of monomial 
deg = 2

# %% Rolling window cross-validation to find optimal regularisation -- autonomous

# Define range of regularisation parameters
reg_range = np.logspace(-15, 0, 16)

# Define the cross-validation process parameters
nblock = 2
block_len = 3000
check_in_period = 1

# Define a store for the mse values
mse_dict_reg = {}

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

        # Define the validation set and lengths using the training dataset 
        data_block = training[:, training_start:testing_end]
        testing_teacher_block = training[:, testing_start:testing_end]
        ntrain_block = training_end - training_start
        ntest_block = testing_teacher_block.shape[1]

        # Perform training on the training block 
        W_out_try, X_try = ngrc.Train(ndelay, deg, reg_try, data_block, ntrain_block, training_start)

        # Perform autonomus forecasting over the testing validation set
        pred_try = ngrc.Forecast(W_out_try, X_try, deg, ntest_block)
        
        # Compute mse for validation block 
        mse_try = np.mean((pred_try - testing_teacher_block)**2)
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