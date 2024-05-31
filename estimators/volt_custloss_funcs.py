
# Volterra class code based on the methods detailed in https://arxiv.org/abs/2212.14641 
# Additionally provides option to make the data size and number of covariates seen different. 

import numpy as np
from scipy.optimize import minimize
from utils.errors import calculate_specdensloss, calculate_wasserstein1err, calculate_mse

class VolterraCustomLoss:
    
    """
    Volterra object that performs optimisation over a loss function (e.g. because loss does not have closed form solution)
    
    Attributes
    ----------
    ld_coef : float
        Coefficient to multiple ld value by. Should be in (0, 1).
    tau_coef : float
        Coefficient to multiple tau value by. Should be in (0, 1).
    washout : int
        Amount of washout to use during training
    nfeatures : int or None, optional
        If None, defaults to usual Volterra kernel regression where the full training inputs - washout are used.
        If not None, must be int. Uses the last nfeatures of the Gram matrix with the usual non-kernel least squares regression. 
        (default: None)
    loss_func : str, optional 
        The loss function that one should find alpha values to optimise over. Options: {"wasserstein1", "specdens"}. (default: "specdens").
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher.
        If nfeatures is None, full Gram matrix is used from the washout onwards as features.
        If nfeatures is provided, Gram matrix is cut to have the shape (ninputs-washout, nfeatures). 
    Forecast(testing input)
        Performs testing using a new set of inputs 
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    
    def __init__(self, ld_coef, tau_coef, washout, nfeatures=None, loss_func="specdens", method="BFGS", seed=100, options={'gtol' : 1e-2}):
        
        # Instance attributes that are user defined
        self.ld_coef = ld_coef          # Gram matrix hyperparameter (has to be between 0 and 1)
        self.tau_coef = tau_coef        # Gram matrix hyperparameter (has to be between 0 and 1)
        self.washout = washout          # Training washout length
        self.nfeatures = nfeatures      # Store the number of features used in training and forecasting
        self.loss_func = loss_func      # Store the loss function that will be optimised over
        self.method = method            # Method to compute minimal in scipy.optimise.minimize
        self.seed = seed                # Seed to initialise the initial guess for scipy optimise
        self.options = options          # Options to pass into scipy optimise

        # Instance attributes storing arrays created by methods
        self.Gram = None                # Store Gram matrix throughout training and forecasting
        self.training_input = None      # Stores training input seen during training
        self.alpha = None               # Stores outcome of regression - weights
        self.alpha0 = None              # Stores outcome of regression - shift
        self.optresult = None           # Stores outcome of scipy.optimize.minimize

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
            self.nfeatures = self.ninputs - self.washout  
        else:   # If provided, check it is an integer and that it is smaller than ninputs
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
        
        # Compute the ld and tau values to be used based on the ld and tau coefficients provided
        self.M = np.max([np.linalg.norm(z) for z in self.training_input])
        tau = np.sqrt(1 / self.M**2)
        self.tau = self.tau_coef * tau
        self.ld = np.sqrt(1 - (self.tau**2) * (self.M**2)) * self.ld_coef
        
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
        
        # Remove washout part from the training teacher data
        training_teacher_washed = training_teacher[self.washout: ]

        # Remove washout and use only feature columns of Gram
        Gram_train = self.Gram[self.washout: , self.ninputs-self.nfeatures: ]

        # Define loss function to be passed into scipy.optimize.minimize
        def loss(alpha):
            alpha = alpha.reshape((self.nfeatures, self.ntargets), order="F")
            unshifted_output = np.matmul(Gram_train, alpha)
            error = calculate_specdensloss(training_teacher_washed, unshifted_output)
            return error
        
        # Call scipy.optimize.minimize
        np.random.seed(self.seed)
        alpha_init = np.random.uniform(low=-1, high=1, size=(self.nfeatures, ))
        self.optresult = minimize(loss, alpha_init, method=self.method, options=self.options) 
        
        # Retrieve the best alpha weights from optresult 
        self.alpha = self.optresult['x'].reshape((self.nfeatures, self.ntargets), order="F") 
        
        # Compute shift of the data
        self.alpha0 = np.mean(training_teacher_washed, axis=0) -  self.alpha.T @ np.mean(Gram_train, axis=0)
        
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
        
        # Assign testing input instance attributes
        self.nhorizon = testing_input.shape[0]
        
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.ntargets))
        
        # Initialise last column of the Gram matrix
        Gram_last_col = self.Gram[:, -1]
        
        # Define initial Gram values (dependent on ld)
        Gram0 = 1/(1-self.ld**2)        
        
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
        
        # Initialise store for the forecast output
        output = np.zeros((self.nhorizon, self.ntargets))

        # Initialise last column of the Gram matrix
        Gram_last_col = self.Gram[:, -1]
        
        # Define initial Gram values (dependent on ld)
        Gram0 = 1/(1-self.ld**2)
        
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
            
        return output
    