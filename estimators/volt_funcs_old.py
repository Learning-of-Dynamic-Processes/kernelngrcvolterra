
# Volterra class code based on the methods detailed in https://arxiv.org/abs/2212.14641 
# Additionally provides option to make the data size and number of covariates seen different. 

import numpy as np
from numba import njit
import time 
from utils.plotting import plot_data

# Fast Gram matrix computation for training
@njit
def volt_gram_train_njit(training_input, tau, ld, ninputs, Gram0):
    
    # Initialise the Gram matrix using the length of the training input
    Gram = np.zeros((ninputs, ninputs))
    
    # Populate the Gram matrix instance attribute using the training input data
    for i in range(ninputs):
        for j in range(i+1):
            if i==0 or j==0:
                Gram[i, j] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(training_input[i], training_input[j])))
            else:
                Gram[i, j] = 1 + ld**2 * Gram[i-1,j-1]/(1-(tau**2)*(np.dot(training_input[i], training_input[j])))
            Gram[j, i] = Gram[i, j]
    
    return Gram

# Fast Forecasting function
@njit 
def volt_forecast_njit(nhorizon, ntargets, ninputs, nfeatures, Gram_last_col, Gram0, training_input, testing_input, ld, tau, alpha, alpha0):
    # Initialise store for the forecast output
    output = np.zeros((nhorizon, ntargets))
    # Iterate through the testing horizon
    for t in range(nhorizon):
        # Fill in the column of the Gram matrix for the new input
        nrows = ninputs + t + 1
        Gram_new_input_col = np.zeros((nrows, ))
        
        for row_id in range(nrows): 
            if row_id == 0: 
                Gram_new_input_col[row_id] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(training_input[row_id], testing_input[t])))
            elif row_id <= ninputs-1:
                Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(training_input[row_id], testing_input[t])))
            else:
                Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(testing_input[row_id-ninputs], testing_input[t])))
        
        # Compute the forecast using the new Gram input column
        for target in range(ntargets):
            output[t, target] = np.dot(alpha[:, target], Gram_new_input_col[ninputs-nfeatures:ninputs]) + alpha0[target]
        
        # Initialise the new last column of the Gram matrix
        Gram_last_col = Gram_new_input_col
    return output

@njit 
def volt_forecast_gram_njit(t, ninputs, training_input, testing_input, Gram0, Gram_last_col, ld, tau):
    nrows = ninputs + t + 1
    Gram_new_input_col = np.zeros((nrows, ))
    for row_id in range(nrows):
        if row_id == 0: 
                Gram_new_input_col[row_id] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(training_input[row_id], testing_input[t])))
        elif row_id <= ninputs-1:
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(training_input[row_id], testing_input[t])))
        else:
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(testing_input[row_id-ninputs], testing_input[t])))
    return Gram_new_input_col
    

# Fast PathContinue function
@njit
def volt_pathcontinue_njit(nhorizon, ntargets, ninputs, nfeatures,
                           training_input, latest_input, Gram0, Gram_last_col, 
                           ld, tau, alpha, alpha0):
    # Initialise store for the forecast output
    output = np.zeros((nhorizon, ntargets))
    
    # Iterate through the testing horizon
    for t in range(nhorizon):
        
        # Fill in the column of the Gram matrix for the new input
        nrows = ninputs + t + 1
        Gram_new_input_col = np.zeros((nrows, ))
        
        if t == 0:
            for row_id in range(nrows):
                if row_id == 0: 
                    Gram_new_input_col[row_id] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(training_input[row_id], latest_input)))
                elif row_id <= ninputs-1:
                    Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(training_input[row_id], latest_input)))
                else:
                    Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(latest_input, latest_input)))
            print(Gram_new_input_col)        
        else: 
            for row_id in range(nrows):
                if row_id == 0: 
                    Gram_new_input_col[row_id] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(training_input[row_id], output[t-1])))
                elif row_id <= ninputs-1:
                    Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(training_input[row_id], output[t-1])))
                elif row_id == ninputs:
                    Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(latest_input, output[t-1])))
                else:
                    Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(output[row_id-ninputs], output[t-1])))
        

        # Compute the forecast using the new Gram input column
        for target in range(ntargets):
            output[t, target] = np.dot(alpha[:, target], Gram_new_input_col[ninputs-nfeatures:ninputs]) + alpha0[target]
    
        # Initialise the new last column of the Gram matrix
        Gram_last_col = Gram_new_input_col
    return output
   
@njit
def volt_gram_pathcontinue_t0_njit(ninputs, training_input, latest_input, Gram0, Gram_last_col, ld, tau):
    nrows = ninputs + 1            
    Gram_new_input_col = np.zeros((nrows, ))
    for row_id in range(nrows):
        if row_id == 0: 
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(training_input[row_id], latest_input)))
        elif row_id <= ninputs-1:
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(training_input[row_id], latest_input)))
        else:
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(latest_input, latest_input)))     
    return Gram_new_input_col

@njit
def volt_gram_pathcontinue_t_njit(t, ninputs, training_input, latest_input, output, Gram0, Gram_last_col, ld, tau):
    nrows = ninputs + t + 1
    Gram_new_input_col = np.zeros((nrows, ))
    for row_id in range(nrows):
        if row_id == 0: 
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram0/(1-(tau**2)*(np.dot(training_input[row_id], output[t-1])))
        elif row_id <= ninputs-1:
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(training_input[row_id], output[t-1])))
        elif row_id == ninputs:
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(latest_input, output[t-1])))
        else:
            Gram_new_input_col[row_id] = 1 + ld**2 * Gram_last_col[row_id-1]/(1-(tau**2)*(np.dot(output[row_id-ninputs], output[t-1])))
    return Gram_new_input_col
    

class Volterra:
    
    """
    Volterra object that performs L2 least squares regression. 
    
    Attributes
    ----------
    ld_coef : float
        Coefficient to multiple ld value by. Should be in (0, 1).
    tau_coef : float
        Coefficient to multiple tau value by. Should be in (0, 1).
    reg : float
        Regularisation used for Tikhonov least squares regression
    washout : int
        Amount of washout to use during training
    nfeatures : int or None, optional
        If None, defaults to usual Volterra kernel regression where the full training inputs - washout are used.
        If not None, must be int. Uses the last nfeatures of the Gram matrix with the usual non-kernel least squares regression. 
        (default: None)
    pinv : bool, optional
        Whether to use pseudoinverse for Tikhonov regression, (default False)
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher.
        If nfeatures is None, kernel regression is performed.
        If nfeatures is provided, Gram matrix is cut and regular regression is performed. 
    Forecast(testing input)
        Performs testing using a new set of inputs 
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    
    def __init__(self, ld_coef, tau_coef, reg, washout, nfeatures=None, pinv=False):
        
        # Instance attributes that are user defined
        self.ld_coef = ld_coef          # Gram matrix hyperparameter (has to be between 0 and 1)
        self.tau_coef = tau_coef        # Gram matrix hyperparameter (has to be between 0 and 1)
        self.reg = reg                  # Regularisation hyperparameter for L2 regression
        self.washout = washout          # Training washout length
        self.pinv = pinv                # Whether or not to use the pseudo-inverse
        self.nfeatures = nfeatures      # Store the number of features used in training and forecasting

        # Instance attributes storing arrays created by methods
        self.Gram = None                # Store Gram matrix throughout training and forecasting
        self.training_input = None      # Stores training input seen during training
        self.alpha = None               # Stores outcome of regression - weights
        self.alpha0 = None              # Stores outcome of regression - shift

        # Instance attributes storing data dependent values created by methods
        self.ninputs = None             # Store training input length
        self.ntargets = None            # Store number of targets output in testing
        self.nhorizon = None            # Store length of forecasting horizon
        self.ld = None                  # Store the ld value used to build the Gram matrix
        self.tau = None                 # Store the tau value used to build the Gram matrix
        self.M = None                   # Store the uniform bound of the training input data
        
    def Train(self, training_input, training_teacher):
        
        """
        Performs training using the training input against the training teacher in the Volterra method
        
        Parameters
        ----------
        training_input : array_like
            Training input for training in Volterra. Must have format (nsamples, ndim)
        training_teacher : array_like
            Training teacher for training in Volterra. Must have format (nsamples, ndim)

        Returns
        -------
        Volterra : class_instance
            Volterra object with training attributes initialised
        """
        
        # Assign training input instance attributes
        self.training_input = training_input 
        self.ninputs = training_input.shape[0]
        
        # Assign training teacher instance attributes
        self.ntargets = training_teacher.shape[1]
        
        # Assign nfeatures instance attribute
        if self.nfeatures is None:  # If not provided, default to training input length - washout
            nfeaturesProvided = False 
            self.nfeatures = self.ninputs - self.washout  
        else:   # If provided, check it is an integer and that it is smaller than ninputs
            nfeaturesProvided = True
            if not isinstance(self.nfeatures, int):
                raise TypeError("nfeatures provided was not an integer")
            if self.nfeatures > self.ninputs:
                raise ValueError("nfeatures provided was greater than number of inputs")
        
        # Check training input and training teacher sizes are the same
        if self.ninputs != training_teacher.shape[0]:
            raise ValueError("The size of the training teacher and training inputs do not match")

        # Check washout is not greater than the size of the inputs
        if self.washout >= self.ninputs:
            raise ValueError("The washout is too large") 
        
        # Check that the regularisation for regression is scalar
        if not np.isscalar(self.reg):
            raise TypeError("Regression regularisation parameter is not scalar")
        
        # Compute the ld and tau values to be used based on the ld and tau coefficients provided
        self.M = np.max([np.linalg.norm(z) for z in self.training_input])
        tau = np.sqrt(1 / self.M**2)
        self.tau = self.tau_coef * tau
        self.ld = np.sqrt(1 - (self.tau**2) * (self.M**2)) * self.ld_coef
        
        start = time.time()
        # Initialise the Gram matrix using the length of the training input
        self.Gram = np.zeros((self.ninputs, self.ninputs))
        
        # Define initial Gram values (dependent on ld)
        Gram0 = 1/(1-self.ld**2)
        
        # Populate the Gram matrix instance attribute using the training input data
        for i in range(self.ninputs):
            for j in range(i+1):
                if i==0 or j==0:
                    self.Gram[i, j] = 1 + self.ld**2 * Gram0/(1-(self.tau**2)*(np.dot(training_input[i], training_input[j])))
                else:
                    self.Gram[i, j] = 1 + self.ld**2 * self.Gram[i-1,j-1]/(1-(self.tau**2)*(np.dot(training_input[i], training_input[j])))
                self.Gram[j, i] = self.Gram[i, j]
        
        print(f"Original: {time.time() - start}")
        
        start = time.time()
        Gram2 = volt_gram_train_njit(training_input, self.tau, self.ld, self.ninputs, Gram0)
        print(f"With njit: {time.time() - start}")
        
        print(np.allclose(self.Gram, Gram2))
        
        # Remove washout part from the training teacher data
        training_teacher_washed = training_teacher[self.washout: ]

        # Remove washout and use only feature columns of Gram
        Gram_train = self.Gram[self.washout: , self.ninputs-self.nfeatures: ]

        # If nfeatures were not provided in instance definition, is the usual Gram regression
        if nfeaturesProvided is False:
            
            # Perform regression computation for weights
            if self.pinv is False:   # Without using pseudoinverse
                self.alpha = np.linalg.inv((Gram_train + self.reg * np.identity(self.nfeatures))) @ training_teacher_washed
            if self.pinv is True:    # With using pseudoinverse
                self.alpha = np.linalg.pinv((Gram_train + self.reg * np.identity(self.nfeatures))) @ training_teacher_washed
        
        # If nfeatures were provided, use usual L2 regression solution
        if nfeaturesProvided is True:
            
            # Perform regression computation for weights
            if self.pinv is False:   # Without using pseudoinverse
                self.alpha = np.linalg.inv((Gram_train.T @ Gram_train + self.reg * np.identity(self.nfeatures))) @ Gram_train.T @ training_teacher_washed
            if self.pinv is True:    # With using pseudoinverse
                self.alpha = np.linalg.pinv((Gram_train.T @ Gram_train + self.reg * np.identity(self.nfeatures))) @ Gram_train.T @ training_teacher_washed
        
        # Compute the weights constant shift
        self.alpha0 = np.mean(training_teacher_washed, axis=0) - self.alpha.T @ np.mean(Gram_train, axis=0)
        
        return self
    

    def Forecast(self, testing_input):
        
        """
        For some testing input, use the trained Volterra object to generate output based on the 
        training teacher that was given
        
        Parameters
        ----------
        testing_input : array_like
            New input given that should be used for forecasting. Must have format (nsamples, ndim)

        Returns
        -------
        output : array_like
            Volterra forecasts, will be the of the same type as the training teacher. Will have format (nsamples, ndim)
        """
        
        # Option 1
        start = time.time()
        # Assign testing input instance attributes
        self.nhorizon = testing_input.shape[0]
        
        # Define initial Gram values (dependent on ld)
        Gram0 = 1/(1-self.ld**2)        
        
        # Initialise last column of the Gram matrix
        Gram_last_col = self.Gram[:, -1]
        
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.ntargets))
        
        # Iterate through the testing horizon
        for t in range(self.nhorizon):
            
            # Fill in the column of the Gram matrix for the new input
            nrows = self.ninputs + t + 1
            Gram_new_input_col = np.zeros((nrows, ))
            
            for row_id in range(nrows): 
                if row_id == 0: 
                    Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram0/(1-(self.tau**2)*(np.dot(self.training_input[row_id], testing_input[t])))
                elif row_id <= self.ninputs-1:
                    Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram_last_col[row_id-1]/(1-(self.tau**2)*(np.dot(self.training_input[row_id], testing_input[t])))
                else:
                    Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram_last_col[row_id-1]/(1-(self.tau**2)*(np.dot(testing_input[row_id-self.ninputs], testing_input[t])))
            
            # Compute the forecast using the new Gram input column
            for target in range(self.ntargets):
                output[t, target] = np.dot(self.alpha[:, target], Gram_new_input_col[self.ninputs-self.nfeatures:self.ninputs]) + self.alpha0[target]
            
            # Initialise the new last column of the Gram matrix
            Gram_last_col = Gram_new_input_col
        print(f"Original: {time.time() - start}")
        
        # Option 2
        start = time.time()
        output2 = volt_forecast_njit(self.nhorizon, self.ntargets, self.ninputs, self.nfeatures,
                                     Gram_last_col, Gram0, self.training_input, testing_input,
                                     self.ld, self.tau, self.alpha, self.alpha0)
        print(f"With njit: {time.time() - start}")
        
        # Option 3
        start = time.time()
         # Assign testing input instance attributes
        self.nhorizon = testing_input.shape[0]
        
        # Define initial Gram values (dependent on ld)
        Gram0 = 1/(1-self.ld**2)        
        
        # Initialise last column of the Gram matrix
        Gram_last_col = self.Gram[:, -1]
        
        # Initialise store for the forecast output
        output3 = np.zeros((self.nhorizon, self.ntargets))
        
        # Iterate through the testing horizon
        for t in range(self.nhorizon):
            
            Gram_new_input_col = volt_forecast_gram_njit(t, self.ninputs, self.training_input, testing_input, Gram0, Gram_last_col, self.ld, self.tau)
            
            # Compute the forecast using the new Gram input column
            for target in range(self.ntargets):
                output3[t, target] = np.dot(self.alpha[:, target], Gram_new_input_col[self.ninputs-self.nfeatures:self.ninputs]) + self.alpha0[target]
            
            # Initialise the new last column of the Gram matrix
            Gram_last_col = Gram_new_input_col
        print(f"With njit Gram: {time.time() - start}")
        
        print(np.allclose(output, output2))
        print(np.allclose(output2, output3))
        print(np.allclose(output, output3))
        
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
        
        # Assign testing horizon instance attribute
        self.nhorizon = nhorizon

        # Initialise last column of the Gram matrix
        Gram_last_col = self.Gram[:, -1]
        
        # Define initial Gram values (dependent on ld)
        Gram0 = 1/(1-self.ld**2)
        
        # Option 1
        start = time.time()
        output2 = volt_pathcontinue_njit(self.nhorizon, self.ntargets, self.ninputs, self.nfeatures, 
                                         self.training_input, latest_input, Gram0, Gram_last_col,
                                         self.ld, self.tau, self.alpha, self.alpha0)
        print(f"With njit: {time.time() - start}")
        
        # Option3
        start = time.time()
        
        # Initialise store for the forecast output
        output3 = np.zeros((self.nhorizon, self.ntargets))
        
        # Define the last col
        Gram_last_col = self.Gram[:, -1]
        
        # Iterate through the testing horizon
        for t in range(self.nhorizon):
            
            if t == 0:
                Gram_new_input_col = volt_gram_pathcontinue_t0_njit(self.ninputs, self.training_input, latest_input, Gram0, Gram_last_col, self.ld, self.tau)
                print(Gram_new_input_col)
            else: 
                Gram_new_input_col = volt_gram_pathcontinue_t_njit(t, self.ninputs, self.training_input, latest_input, output3, Gram0, Gram_last_col, self.ld, self.tau )
                
            # Compute the forecast using the new Gram input column
            for target in range(self.ntargets):
                output3[t, target] = np.dot(self.alpha[:, target], Gram_new_input_col[self.ninputs-self.nfeatures:self.ninputs]) + self.alpha0[target]
            
            # Initialise the new last column of the Gram matrix
            Gram_last_col = Gram_new_input_col
        
        print(f"With njit but just the Gram: {time.time() - start}")
        
        # Option 2
        start = time.time()
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.ntargets))
        # Define the last col
        Gram_last_col = self.Gram[:, -1]
        # Iterate through the testing horizon
        for t in range(self.nhorizon):
            
            # Fill in the column of the Gram matrix for the new input
            nrows = self.ninputs + t + 1
            Gram_new_input_col = np.zeros((nrows, ))
            
            if t == 0:
                for row_id in range(nrows):
                    if row_id == 0: 
                        Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram0/(1-(self.tau**2)*(np.dot(self.training_input[row_id], latest_input)))
                    elif row_id <= self.ninputs-1:
                        Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram_last_col[row_id-1]/(1-(self.tau**2)*(np.dot(self.training_input[row_id], latest_input)))
                    else:
                        Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram_last_col[row_id-1]/(1-(self.tau**2)*(np.dot(latest_input, latest_input)))
                print(Gram_new_input_col)
            else: 
                for row_id in range(nrows):
                    if row_id == 0: 
                        Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram0/(1-(self.tau**2)*(np.dot(self.training_input[row_id], output[t-1])))
                    elif row_id <= self.ninputs-1:
                        Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram_last_col[row_id-1]/(1-(self.tau**2)*(np.dot(self.training_input[row_id], output[t-1])))
                    elif row_id == self.ninputs:
                        Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram_last_col[row_id-1]/(1-(self.tau**2)*(np.dot(latest_input, output[t-1])))
                    else:
                        Gram_new_input_col[row_id] = 1 + self.ld**2 * Gram_last_col[row_id-1]/(1-(self.tau**2)*(np.dot(output[row_id-self.ninputs], output[t-1])))
                        
            # Compute the forecast using the new Gram input column
            for target in range(self.ntargets):
                output[t, target] = np.dot(self.alpha[:, target], Gram_new_input_col[self.ninputs-self.nfeatures:self.ninputs]) + self.alpha0[target]
            
            # Initialise the new last column of the Gram matrix
            Gram_last_col = Gram_new_input_col
            
        print(f"Original: {time.time() - start}")
        plot_data([output - output3])
        print(np.allclose(output2, output3))
        print(np.allclose(output, output2))
        print(np.allclose(output, output3))

        return output
