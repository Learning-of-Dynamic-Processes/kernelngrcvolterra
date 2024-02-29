
### Volterra class code based on the methods detailed in https://arxiv.org/abs/2212.14641 
### Additionally provides option to control the number of features used. 

import numpy as np

class Volterra2:
    
    """
    Volterra object that performs L2 least squares regression. 
    
    Attributes
    ----------
    ld_coef : float
        Coefficient to multiple ld value by. Should be in (0, 1).
    tau_coef : float
        Coefficient to multiple tau value by. Should be in (0, 1).
    reg : float
        Reguralisation used for Tikhonov least squares regression
    washout : int
        Amount of washout to use during training
    nfeatures : int or None, optional
        If None, defaults to usual Volterra regression where there full training inputs - washout are used.
        If not None, must be int. Uses the last nfeatures of the Gram matrix with the usual non-kernel regression. 
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
        self.tau = None                 # Store the tau value sued to build the Gram matrix
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
    