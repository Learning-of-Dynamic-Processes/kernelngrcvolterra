
### Volterra class code based on the methods detailed in https://arxiv.org/abs/2212.14641 

import numpy as np
from sklearn.linear_model import Lasso
from time import process_time


class Volterra:
    
    """
    Volterra object
    
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
    regression : str, optional
        Regression type to use. Options: {"L2", "Lasso"}, (default "L2")
    pinv : bool, optional
        Whether to use pseudoinverse for Tikhonov regression, (default False)
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher in the Volterra method
    Forecast(testing input)
        Performs testing using a new set of training input 
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    def __init__(self, ld_coef, tau_coef, reg, washout, regression="L2", pinv=False):
        
        
        # Instance attributes that are user defined
        self.ld_coef = ld_coef          # Gram matrix hyperparameter (has to be between 0 and 1)
        self.tau_coef = tau_coef        # Gram matrix hyperparameter (has to be between 0 and 1)
        self.washout = washout          # Training washout length
        self.regression = regression    # Regression type: {"L2", "Lasso"}
        self.pinv = pinv
        
        # Instance attributes related to regularisation
        if regression == "L2":          # For L2 regression
            self.reg = reg              # Regularisation hyperparameter for L2
        if regression == "Lasso":       # For Lasso Regression
            self.lasso_max_iter = 1000  # Number of iterations by sklearn Lasso (same as default)
            self.lasso_tol = 1e-4       # Tolerance for optimisation (duality, updates) for sklearn Lasso (same as default)
            self.reg = reg              # Regularisation hyperparameter for Lasso     
    
        # Instance attributes storing arrays created by methods
        self.Gram = None                # Store Gram matrix throughout training and forecasting
        self.training_input = None      # Stores training input seen during training
        self.alpha = None               # Stores outcome of regression - weights
        self.alpha0 = None              # Stores outcome of regression - shift

        # Instance attributes storing data dependent values created by methods
        self.ninputs = None             # Store training input length
        self.nfeatures = None           # Store number of input features seen from training
        self.nfeatures_seen = None      # Store number of input features actually used in training
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
        
        # Assign as instance attribute that are based on training input data
        self.training_input = training_input 
        self.ninputs = training_input.shape[0]
        self.nfeatures = training_input.shape[1]
        
        # Assign instance attributes that are related to the training teacher data
        self.ntargets = training_teacher.shape[1]
        
        # Check that the training input and training teacher sizes are the same
        nteacher = training_teacher.shape[0]
        if self.ninputs != nteacher:
            raise ValueError("The size of the training teacher and training inputs do not match")
        
        # Check that the washout is not greater than the size of the inputs
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
        
        # Remove the washout part from the training teacher data
        training_teacher_washed = training_teacher[self.washout: ]

        # Remove the washout part from the Gram matrix, record its size
        Gram_train = self.Gram[self.washout: , self.washout: ]
        nGram_train = Gram_train.shape[0]
        
        # Perform Tikhonov least squares regression 
        if self.regression == "L2":
            
            # Perform checks and modifications on regression parameter given
            if not np.isscalar(self.reg):
                raise TypeError("L2 Regularisation was selected but regression parameter is not scalar")
            
            # Perform regression computation for weights
            if self.pinv is False:   # Without using pseudoinverse
                self.alpha = np.linalg.inv((Gram_train + self.reg * np.identity(nGram_train))) @ training_teacher_washed
            if self.pinv is True:    # With using pseudoinverse
                self.alpha = np.linalg.pinv((Gram_train + self.reg * np.identity(nGram_train))) @ training_teacher_washed

            # Compute the weights constant shift
            self.alpha0 = np.mean(training_teacher_washed, axis=0) - np.matmul(self.alpha.T, np.mean(Gram_train, axis=0))
            
            # Assign as attributes size of the regression outputs (no. variables seen after removing washout)
            self.nfeatures_seen = self.alpha.shape[0]
            
        # Perform Lasso least squares regression using sklearn built in lasso package
        if self.regression == "Lasso":
            
            # Perform checks and modifications on regression parameter given
            if np.isscalar(self.reg):
                self.reg = [self.reg] * self.ntargets
            elif isinstance(self.reg, np.ndarray):
                if self.reg.shape != (self.ntargets, ):
                    raise TypeError("Lasso regularisation was chosen but the array given as regularisation is wrong")
            elif isinstance(self.reg, list):
                if len(self.reg) != self.ntargets:
                    raise TypeError("Lasso regularisation was chosen but the array given as regularisation is wrong")
            
            # Initialise lasso output store for all targets
            alpha_lasso = np.zeros((nGram_train, self.ntargets))
            alpha0_lasso = np.zeros((self.ntargets, ))
            nfeatures_seen_lasso = np.zeros((self.ntargets, ))
            
            # Iterate through each target then run sklearn lasso regression
            for target in range(self.ntargets):
                
                # Define target's regularisation parameter
                target_reg_param = self.reg[target]
                
                # Perform Lasso least squares regularisation using the sklearn lasso model
                lasso_target = Lasso(target_reg_param, max_iter=self.lasso_max_iter, tol=self.lasso_tol).fit(Gram_train, training_teacher_washed[:, target])
                alpha_target, alpha0_target = lasso_target.coef_, lasso_target.intercept_
                
                # Fill up the regression parameters with the regression result
                alpha_lasso[:, target] = alpha_target
                alpha0_lasso[target] = alpha0_target
                nfeatures_seen_lasso[target] = len(np.nonzero(alpha_lasso[:, target])[0])
            
            # Assign as instance attributes the Lasso results
            self.alpha = alpha_lasso
            self.alpha0 = alpha0_lasso
            
            # Assign as instance attributes the number of features seen (no. nonzero coefficients)
            self.nfeatures_seen = nfeatures_seen_lasso
            
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
        
        # Assign as instance attribute testing input related 
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
                output[t, target] = np.dot(self.alpha[:, target], Gram_new_input_col[self.washout:self.ninputs]) + self.alpha0[target]
            
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
        
        # Assign as instance attribute the testing horizon given
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
                output[t, target] = np.dot(self.alpha[:, target], Gram_new_input_col[self.washout:self.ninputs]) + self.alpha0[target]
            
            # Initialise the new last column of the Gram matrix
            Gram_last_col = Gram_new_input_col
            
        return output


