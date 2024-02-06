import datagen.data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt
from time import process_time 

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

# Construct time evaluation as needed by solving in SINDy
training_t_eval = t_eval[0:ntrain]

# %% Rolling window cross-validation for optimal regularisation and thresholding -- auto forecasting

# Define the range of alpha and threshold range
alpha_range = np.logspace(-15, 0, 16)
threshold_range = np.logspace(-15, 0, 16)

# Define the cross-validation process parameters
nblock = 2
block_len = 3000
check_in_period = 1

# Define a store for the mse values
mse_dict_alpha_threshold = {}

# Start timing cross-validation process
cv_start = process_time()

# Perform rolling-window cross validation varying ld
iter_count = 0
for alpha_try in alpha_range:
    for threshold_try in threshold_range:
        
        # Record the values of ld, tau coeffs being tried
        mse_dict_alpha_threshold[(alpha_try, threshold_try)] = np.zeros((nblock-1, ))
        
        for block in range(nblock-1):
            
            # Define training indices for each block, letting the last block run over the remainder
            training_start = washout + block * block_len
            training_end =  training_start + block_len 
            testing_start = training_end
            if block == nblock - 2:
                testing_end = -1
            else: testing_end = testing_start + block_len
            
            # Define training and testing dataset
            training_block = training[training_start:training_end]
            testing_block = training[testing_start:testing_end]
            t_eval_block = training_t_eval[testing_start:testing_end]
            
            # Train and fit SINDy model using the validation sets
            optimizer_try = ps.STLSQ(threshold=threshold_try, alpha=alpha_try)
            model_try = ps.SINDy(optimizer_try)
            model_try.fit(training_block, t=h*slicing)
            pred_try = model_try.simulate(testing_block[0], t_eval_block)
            
            # Compute mse value for block
            mse_try = np.mean((pred_try - testing_block)**2)
            mse_dict_alpha_threshold[(alpha_try, threshold_try)][block] = mse_try
            
        # Check in at every predefined period
        if iter_count % check_in_period == 0:
           print((alpha_try, threshold_try), ": ", mse_dict_alpha_threshold[(alpha_try, threshold_try)])

        iter_count = iter_count + 1

# End timing cross-validation process
cv_end = process_time()

print("Cross validating over ", len(alpha_range)*len(threshold_range), " hyperpameters took ", cv_end-cv_start, "seconds.")

 # %% Find the minimum MSE and the parameters alpha, threshold that give this mse

# Define a list to store the average errors over all blocks
avg_error_ls = []
# Define a large number as the minimum error
min_error = np.infty
# Define a variable to store the smallest ld, tau found
min_alpha_thresh = None
# Iterate through the dictionary
for key in mse_dict_alpha_threshold:
    # Average the error over each block then append the average
    avg_error = np.mean(mse_dict_alpha_threshold[key])
    avg_error_ls.append(avg_error)
    # Determine if a new min error has been found then store it
    if avg_error < min_error:
        min_error = avg_error
        min_alpha_thresh = key

print(min_alpha_thresh, ": ", min_error)

# %% Plot mses using a ld-tau contour plot

fig, ax = plt.subplots(nblock-1)

# Required to be able to universally access each data type
n_alpha_range = len(alpha_range)
n_thresh_range = len(threshold_range)
mse_array_alpha_thresh = np.zeros((n_alpha_range, n_thresh_range))

levels = n_alpha_range

# Iterate through each of the block pairs
for block in range(nblock-1):
    
    # Rewrite the errors and ld, tau pairs for contour plots
    for alpha_id in range(n_alpha_range):
        for thresh_id in range(n_thresh_range):
            alpha_val = alpha_range[alpha_id]
            thresh_val = threshold_range[thresh_id]
            mse_array_alpha_thresh[alpha_id, thresh_id] = mse_dict_alpha_threshold[(alpha_val, thresh_val)][block]
    
# Generate the contour plot
cs = ax.contourf(alpha_range, threshold_range, mse_array_alpha_thresh, cmap="RdBu_r", levels=levels)
ax.set_xlabel(r"$\lambda$")
    
fig.supylabel(r"$\tau$")
fig.colorbar(cs, ax=ax)    
plt.show()