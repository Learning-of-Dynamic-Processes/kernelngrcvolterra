# %% 

import numpy as np
from itertools import combinations_with_replacement
from math import comb

# Requires data in the form (nfeatures, nsample)

def Train(ndelay, deg, reg, training_input, training_teacher, start):

    # Define size of training data, inferred based on the size of training input
    ntrain = training_input.shape[1] + 1
    # Define dimension of input based on training data
    ndim = training_input.shape[0]
    
    # Size of linear part of feature vector
    dlin = ndelay * ndim
    
    # Size of nonlinear part of feature vector
    dnonlin = 0
    for inter_deg in range(2, deg+1):
        dnonlin = dnonlin + comb(inter_deg+dlin-1, dlin-1)
    
    # Total size of feature vector: constant + linear + nonlinear
    dtot = 1 + dlin + dnonlin
    
    # Create array to hold linear part of feature vector
    #X = np.zeros((dlin, ndata))
    X = np.zeros((dlin, ntrain-1))

    # Fill in the linear part of the feature vector
    for delay in range(ndelay):
        for j in range(delay, ntrain-1):
            X[ndim*delay:ndim*(delay+1), j] = training_input[:, j-delay]
    
    # Create feature vector over training time
    O_train = np.ones((dtot, ntrain-1-start))
    
    # Copy over linear part (shifting by one to account for constant)
    O_train[1:dlin+1, :] = X[:, start:ntrain-1]
    
    # Fill in nonlinear part of the feature vector
    O_row = 1 + dlin
    # Iterate through each monomial degree
    for inter_deg in range(2, deg+1):
        # Generate iterator of combinations rows of X for each degree
        iter_monomials = combinations_with_replacement(range(dlin), inter_deg)
        # Fill up the rows of O train for each monomial 
        for X_row_ids in iter_monomials:
            monomial_row = X[X_row_ids[0], start:ntrain-1]
            for row_id in range(1, inter_deg):
                monomial_row = monomial_row * X[X_row_ids[row_id], start:ntrain-1]
            O_train[O_row] = monomial_row
            O_row = O_row + 1
    
    # Ridge regression train W_out with X_i+1 - X_i
    #W_out = (training_teacher[:, start:] - training_input[:, start:]) @ O_train[:, :].T @ np.linalg.pinv(O_train[:,:] @ O_train[:, :].T + reg * np.identity(dtot))
    
    W_out = (training_teacher[:, start:] - training_input[:, start:]) @ O_train.T @ np.linalg.pinv(O_train @ O_train.T + reg * np.identity(dtot))
    
    return W_out, X     # X needed for forecasting

def Forecast(W_out, X, latest_input, deg, ntest):
    
    # Redefine size of linear part of feature vector
    dlin = X.shape[0]
    # Redefine the size of total feature vector size
    dtot = W_out.shape[1]
    # Redefine the dimension of the data set
    ndim = W_out.shape[0]
    
    # Create store for feature vectors for prediction
    O_test = np.ones(dtot)              # full feature vector
    X_test = np.zeros((dlin, ntest+1))    # linear portion of feature vector
    
    # Fill in the linear part of the feature vector with the latest input data and delay
    X_test[0:ndim, 0] = latest_input
    X_test[ndim: , 0] = X[0:dlin-ndim, -1]
    
    # Apply W_out to feature vector to perform prediction
    for j in range(ntest):
        # Copy linear part into whole feature vector
        O_test[1:dlin+1] = X_test[:, j] # shift by one for constant

        # Fill in the nonlinear part
        O_row = 1 + dlin
        # Iterate through each monomial degree
        for inter_deg in range(2, deg+1):
            # Generate iterator of combinations rows of X for each degree
            iter_monomials = combinations_with_replacement(range(dlin), inter_deg)
            # Fill up the rows of O test for each monomial 
            for X_row_ids in iter_monomials:
                monomial_row = X_test[X_row_ids[0], j]
                for row_id in range(1, inter_deg):
                    monomial_row = monomial_row * X_test[X_row_ids[row_id], j]
                O_test[O_row] = monomial_row
                O_row = O_row + 1
            
        # Fill in the delay taps of the next state
        X_test[ndim:dlin, j+1] = X_test[0:dlin-ndim, j]
        # Perform a prediction
        X_test[0:ndim, j+1] = X_test[0:ndim, j] + W_out @ O_test[:]

    return X_test[0:ndim, 1:]

# %% 
def NGRC(training_input, training_teacher, testing_input, ntest, ngrc_params):
    
    # Roll out the NGRC parameters
    ndelay, deg, reg, washout = ngrc_params
    
    # Combine the datasets into one long dataset
    data = np.concatenate((training_input, training_teacher, testing_input), axis=1)
    # Compute training length value
    ntrain = training_input.shape[1] + 1

    W_out, X = Train(ndelay, deg, reg, data, ntrain, washout)
    pred_auto = Forecast(W_out, X, deg, ntest)
    
    return pred_auto