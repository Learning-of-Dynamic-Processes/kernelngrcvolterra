# %% 
import numpy as np
from time import process_time
# %% 
# Function to initialise the Gram matrix with a set of data and initial Gram0
def GramMat(data, ld, tau):
    ndata = len(data)
    Gram = np.zeros((ndata, ndata))
    Gram0 = 1/(1-ld**2)
    for i in range(ndata):
        for j in range(i+1):
            if i==0 or j==0:
                Gram[i, j] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(data[i], data[j])))
            else:
                Gram[i, j] = 1 + ld**2 * Gram[i-1,j-1]/(1-(tau**2)*(np.dot(data[i], data[j])))
            Gram[j, i] = Gram[i, j]
    return Gram

# Function to perform least squares regression using input Gram matrix (washout already applied)
def LeastSquaresReg(Gram, data, reg, pinv=False):
    nGram = Gram.shape[0]
    if pinv is False:   # choose between using pseudo-inverse or inverse
        alpha_ols = np.matmul(np.linalg.inv((Gram + reg * np.identity(nGram))), data)
    if pinv is True:
        alpha_ols = np.matmul(np.linalg.pinv((Gram + reg * np.identity(nGram))), data)
    alpha0_ols = np.mean(data, axis=0) - np.matmul(alpha_ols.transpose(), np.mean(Gram, axis=0))
    return alpha_ols, alpha0_ols

# Function to take an existing Gram matrix, already padded with dummy values, and fill it
def GramExtend(Gram_padded, new_data, prev_data, ld, tau):
    ndata = len(prev_data)
    nhorizon = len(new_data)
    Gram0 = 1/(1-ld**2)
    for i in range(nhorizon):
        for j in range(ndata+i+1):
            if j <= ndata-1:
                if j == 0: 
                    Gram_padded[ndata+i, j] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(new_data[i], prev_data[j])))
                else:
                    Gram_padded[ndata+i, j] = 1 + ld**2 * Gram_padded[ndata+i-1,j-1]/(1-(tau**2)*(np.dot(new_data[i], prev_data[j])))
            else:
                Gram_padded[ndata+i, j] = 1 + ld**2 * Gram_padded[ndata+i-1,j-1]/(1-(tau**2)*(np.dot(new_data[i], new_data[j-ndata])))
            Gram_padded[j, ndata+i] = Gram_padded[ndata+i, j]
    return Gram_padded

# Function to generate outputs with a Gram matrix that already has the washout removed
def StepsAhead(Gram, alpha, alpha0, nhorizon):
    nGram = Gram.shape[1]
    ndata = nGram - nhorizon
    ndim = alpha.shape[1]
    pred = np.zeros((nhorizon, ndim))
    for dim in range(ndim):
        alpha_dim = alpha[:, dim]
        alpha0_dim = alpha0[dim]
        for t in range(nhorizon):
            pred[t, dim] = np.matmul(alpha_dim.transpose(), Gram[:, ndata+t]) + alpha0_dim
    return pred

# Function to take wrap training process
def Train(training_input, training_teacher, start, ld, tau, reg, pinv=False):
    
    # Obtain training input and populate Gram matrix
    Gram = GramMat(training_input, ld, tau)
    # Remove the washout part from the Gram matrix 
    Gram_train = Gram[start: , start: ]
    # Remove the washout part from the training teacher data
    training_teacher_washed = training_teacher[start: ]
    # Perform least squares regression on washed data
    alpha_ols, alpha0_ols = LeastSquaresReg(Gram_train, training_teacher_washed, reg, pinv)
    
    return alpha_ols, alpha0_ols, Gram      # Gram matrix required for forecasting

# Function to wrap nonautonomous forecasting process
def Forecast(Gram, training_input, testing_input, alpha, alpha0, start, ld, tau):
    
    # Obtain length of testing data
    nhorizon = testing_input.shape[0]
    # Obtain length of Gram matrix for StepsAhead computation
    nalpha = alpha.shape[0]
    # Pad the Gram matrix with 0s to be filled up by GramExtend
    Gram_padded = np.pad(Gram, ((0, nhorizon), (0, nhorizon)), 'constant')
    # Extended Gram matrix populated with training input using testing input
    Gram_extended = GramExtend(Gram_padded, testing_input, training_input, ld, tau)
    # Use washed portion of Gram matrix corresponding to washed training data
    Gram_testing = Gram_extended[start:start+nalpha, start: ]
    # Generate outputs with washed Gram matrix
    pred_block_nonauto = StepsAhead(Gram_testing, alpha, alpha0, nhorizon)
    return pred_block_nonauto

# Function to wrap autonomous forecasting process
def ForecastAuto(Gram_init, training_input_init, latest_input, alpha, alpha0, start, ntest, ld, tau):
    
    # Obtain length of dimension of training input to reshape incoming data and create prediction array
    ndim = training_input_init.shape[1]
    # Obtain length of Gram matrix for StepsAhead Computation
    nalpha = alpha.shape[0]
    # Create store for autonomous prediction 
    pred_auto = np.zeros((ntest, ndim))
    # Initialise training input for each iteration with initial training values
    training_input_extended = training_input_init
    # Initialise Gram matrix with a padded-with-0s Gram matrix to be filled up
    Gram_extended = np.pad(Gram_init, ((0, ntest), (0, ntest)), 'constant')
    # Reshape latest input in case latest input is wrongly shaped, or 1D.
    latest_input = np.reshape(latest_input, (1, ndim))
    
    # Iterate through the length of testing horizon
    for t in range(ntest):
        
        # Using initialised padded Gram matrix, populate using the latest input
        Gram_extended = GramExtend(Gram_extended, latest_input, training_input_extended, ld, tau)
        # Use washed portion of Gram matrix corresponding to washed training data
        Gram_testing = Gram_extended[start:start+nalpha, start:start+nalpha+t+1]
        # Perform a 1 step ahead forecasting using the washed out Gram matrix
        latest_output = StepsAhead(Gram_testing, alpha, alpha0, 1)
        # Input output into the storage for the predictions
        pred_auto[t] = latest_output[0]   # Will output a 2D array, only want the first dimension 
        # Update the training input values as this is needed to extend Gram in next step
        training_input_extended = np.concatenate((training_input_extended, latest_input))
        # Update the latest input to be the output
        latest_input = latest_output
    
    return pred_auto
    
'''

'''
# Wrapper function to that trains and does autonomous forecasting and outputs forecasts
def Volterra(training_input, training_teacher, testing_input, ntest, volt_params):
    
    # Roll out the Volterra parameters
    ld, tau, reg, washout = volt_params
    
    alpha, alpha0, Gram = Train(training_input, training_teacher, washout, ld, tau, reg)
    pred_auto = ForecastAuto(Gram, training_input, testing_input, alpha, alpha0, ntest, ld, tau)
    
    return pred_auto
