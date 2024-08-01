
# Polynomial kernel class code. Polynomial kernel includes option to add delays. 
# More information about kernel methods are found in Smola's Learning with Kernels or Mohri's Foundations of Machine Learning

import numpy as np
from numba import njit

# Fast delay matrix computation for training with numba njit
@njit        
def poly_train_delay_njit(training_input, ndelays, ninputs, ndim):
    Delays = np.zeros((ndim * ndelays, ninputs - ndelays + 1))
    for i in range(ndelays-1, ninputs):
        for delay in range(ndelays):
            Delays[delay*ndim:(delay+1)*ndim, i-ndelays+1] = training_input[i-delay]
    return Delays

# Fast kernel matrix computation for training with numba njit
@njit
def poly_train_kernel_njit(Delays, ndelays, ninputs, deg):
    deg = float(deg)    # if not, does not match with original python numpy
    Kernel = np.zeros((ninputs-ndelays+1, ninputs-ndelays+1))
    for i in range(ninputs-ndelays+1):
        for j in range(i+1):
            kernel_val = (1 + np.dot(Delays[:, i], Delays[:, j]))**deg
            Kernel[i, j] = kernel_val
            Kernel[j, i] = Kernel[i, j]
    return Kernel

# Fast new kernel matrix column computation with numba njit for pathcontinue and forecast
@njit
def poly_forecast_pathcontinue_kernel_njit(t, Delays, Delay_new, ndelays, ninputs, deg):
    deg = float(deg)    # if not, does not match with original python numpy
    nrows = ninputs - ndelays + 1 + t + 1
    Kernel_new_col = np.zeros((nrows, ))
    for row_id in range(nrows):
        if row_id <= ninputs - ndelays:
            Kernel_new_col[row_id] = (1 + np.dot(Delays[:, row_id], Delay_new[:, t]))**deg
        else:
            Kernel_new_col[row_id] = (1 + np.dot(Delay_new[:, row_id-(ninputs-ndelays+1)], Delay_new[:, t]))**deg
    return Kernel_new_col

# Polynomial kernel ridge regression
class PolynomialKernel:
    
    """
    Polynomial Kernel object that performs kernel least squares regression. 
    
    Attributes
    ----------
    
    deg : int
        Degree of polynomials used in the kernel 
    ndelays : int
        Number of delays to include, inclusive of the most recent time step. 
    reg : float
        Regularisation used for Tikhonov least squares regression
    washout : int
        Amount of washout to use during training
    pinv : bool, optional
        Whether to use pseudoinverse for Tikhonov regression. (default: False)
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher.
    Forecast(testing input)
        Performs testing using a new set of inputs 
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    
    def __init__(self, deg, ndelays, reg, washout, pinv=False):
        
        # Instance attributes that are user defined
        self.deg = deg                  # Degree of polynomial kernel in kernel matrix
        self.ndelays = ndelays          # Number of delays to include in the polynomial kernel regression
        self.reg = reg                  # Regularisation hyperparameter for L2 regression
        self.washout = washout          # Training washout length
        self.pinv = pinv                # Whether or not to use the pseudo-inverse

        # Instance attributes storing arrays created by methods
        self.Kernel = None              # Store Kernel matrix throughout training and forecasting
        self.Delays = None              # Store delay vectors in matrix throughout training and forecasting
        self.training_input = None      # Stores training input seen during training
        self.alpha = None               # Stores outcome of regression - weights
        self.alpha0 = None              # Stores outcome of regression - shift

        # Instance attributes storing data dependent values created by methods
        self.ninputs = None             # Store training input length
        self.ndim = None                # Store the dimension of the training input 
        self.ntargets = None            # Store number of targets output in testing
        self.nhorizon = None            # Store length of forecasting horizon
        
    def Train(self, training_input, training_teacher):
        
        """
        Performs training using the training input against the training teacher in the Polynomial kernel method
        
        Parameters
        ----------
        training_input : array_like
            Training input for training in PolynomialKernel. Must have format (nsamples, ndim)
        training_teacher : array_like
            Training teacher for training in PolynomialKernel. Must have format (nsamples, ndim)

        Returns
        -------
        PolynomialKernel : class_instance
            PolynomialKernel object with training attributes initialised
        """
        
        # Assign training input instance attributes
        self.training_input = training_input 
        self.ninputs = training_input.shape[0]
        self.ndim = training_input.shape[1]
        
        # Assign training teacher instance attributes
        self.ntargets = training_teacher.shape[1]
        
        # Check training input and training teacher sizes are the same
        if self.ninputs != training_teacher.shape[0]:
            raise ValueError("The size of the training teacher and training inputs do not match")

        # Check washout is not greater than the size of the inputs
        if self.washout >= self.ninputs:
            raise ValueError("The washout is too large") 
        
        # Check that the regularisation for regression is scalar
        if not np.isscalar(self.reg):
            raise TypeError("Regression regularisation parameter is not scalar")

        # Initialise the Kernel and Delay matrix using the length of the training input
        self.Delays = poly_train_delay_njit(training_input, self.ndelays, self.ninputs, self.ndim)
        self.Kernel = poly_train_kernel_njit(self.Delays, self.ndelays, self.ninputs, self.deg)

        # Remove washout part from the training teacher data
        training_teacher_washed = training_teacher[self.ndelays-1+self.washout: ]

        # Remove washout and use only feature columns of Gram
        Kernel_train = self.Kernel[self.washout: , self.washout: ]
        
        # Perform regression computation for weights
        if self.pinv is False:   # Without using pseudoinverse
            self.alpha = np.linalg.inv((Kernel_train + self.reg * np.identity(self.ninputs-self.ndelays+1-self.washout))) @ training_teacher_washed
        if self.pinv is True:    # With using pseudoinverse
            self.alpha = np.linalg.pinv((Kernel_train + self.reg * np.identity(self.ninputs-self.ndelays+1-self.washout))) @ training_teacher_washed
        
        # Compute the weights constant shift
        self.alpha0 = np.mean(training_teacher_washed, axis=0) - self.alpha.T @ np.mean(Kernel_train, axis=0)

        return self
    

    def Forecast(self, testing_input):
        
        """
        For some testing input, use the trained PolynomialKernel object to generate output based on the training teacher that was given
        
        Parameters
        ----------
        testing_input : array_like
            New input given that should be used for forecasting. Must have format (nsamples, ndim)

        Returns
        -------
        output : array_like
            PolynomialKernel forecasts, will be the of the same type as the training teacher. Will have format (nsamples, ndim)
        """
        
        # Assign testing input instance attributes
        self.nhorizon = testing_input.shape[0]
        
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.ntargets))
        
        # Initialise store for Delay vectors in testing horizon
        Delay_new = np.zeros((self.ndim * self.ndelays, self.nhorizon))
        
        # Initialise the last col of the delay vectors matrix
        Delay_last_col = self.Delays[:, -1]
        
        # Iterate through testing horizon, build new delay vectors, fill up kernel, generate output
        for t in range(self.nhorizon):
            
            # Build delay vector for time step t
            Delay_new[0:self.ndim, t] = testing_input[t]
            Delay_new[self.ndim: , t] = Delay_last_col[0:self.ndim*(self.ndelays-1)]
            
            # Generate new kernel column
            Kernel_new_col = poly_forecast_pathcontinue_kernel_njit(t, self.Delays, Delay_new, self.ndelays, self.ninputs, self.deg)

            # Compute forecast using new Gram input column
            for target in range(self.ntargets):
                output[t, target] = np.dot(self.alpha[:, target], Kernel_new_col[self.washout:self.ninputs-self.ndelays+1]) + self.alpha0[target]
            
            # Reassign last delay column
            Delay_last_col = Delay_new[:, t]
                
        return output
    
    
    def PathContinue(self, latest_input, nhorizon):  
         
        """
        Simulates forward in time using the latest input for nhorizon period of time
        
        Parameters
        ----------
        latest_input : array_like
            Starting input to path continue from
        nhorizon : int
            Period of time to path continue over

        Returns
        -------
        output : array_like
            Output of forecasting. Will have format (nsamples, ndim)
        """
        
        # Assign testing input instance attributes
        self.nhorizon = nhorizon
        
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.ntargets))
        
        # Initialise store for Delay vectors in testing horizon
        Delay_new = np.zeros((self.ndim * self.ndelays, self.nhorizon))
        
        # Initialise the last col of the delay vectors matrix
        Delay_last_col = self.Delays[:, -1]
        
        # Iterate through testing horizon, build new delay vectors, fill up kernel, generate output
        for t in range(self.nhorizon):
            
            # Build delay vector for time step t
            Delay_new[0:self.ndim, t] = latest_input
            Delay_new[self.ndim: , t] = Delay_last_col[0:self.ndim*(self.ndelays-1)]
            
            # Generate new kernel column
            Kernel_new_col = poly_forecast_pathcontinue_kernel_njit(t, self.Delays, Delay_new, self.ndelays, self.ninputs, self.deg)
            
            # Compute forecast using new Gram input column
            for target in range(self.ntargets):
                output[t, target] = np.dot(self.alpha[:, target], Kernel_new_col[self.washout:self.ninputs-self.ndelays+1]) + self.alpha0[target]
            
            # Reassign last delay column
            Delay_last_col = Delay_new[:, t]
            latest_input = output[t, :]
                
        return output
    