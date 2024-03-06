
### Volterra class code adapted from the methods detailed in https://arxiv.org/abs/2212.14641 
### BUT using Lasso regression instead of L2 regression

import numpy as np
from sklearn.linear_model import Lasso

class VolterraLasso:
    
    """
    Volterra object with Lasso regression. 
    
    Attributes
    ----------
    ld_coef : float
        Coefficient to multiple ld value by. Should be in (0, 1).
    tau_coef : float
        Coefficient to multiple tau value by. Should be in (0, 1).
    reg : float or list of floats or array of floats
        Reguralisation used for Lasso regression. 
        If nfeatures is None, then reg can be a scalar, array of reg value scalars, or list of reg value scalars
        If nfeatures is not None, then reg must be a list of 1D numpy arrays or a single 1D numpy array. 
    washout : int
        Amount of washout to use during training
    nfeatures : (int, list of ints) or None
        Number of nonzero coefficients in the regression weights during Lasso. 
        If nfeatures is None, then Lasso regression defaults to the usual Lasso
            nfeatures is updated to record the number of nonzero coefficients chosen by sklearn lasso. 
        If nfeatures is provided, then it must be an int or list of ints. 
            Regression is provided by going along regularisation values until number of nonzero weights in
            the regression weights is below nfeatures[target] or nfeatures. 
            Then nfeature attribute is updated to reflect the new number of nonzero weights chosen. 
    max_iter : int 
        Max number of iterations used by elastic net in sklearn. (default: 1000)
    tol : float
        Tolerance for elastic net in sklearn. (default: 1e-4)
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher.
        If nfeatures is None, performs lasso regression by sklearn.
        If nfeatures is not None, performs lasso regression iteratively until number of nonzero coefficients
            is below the given number in nfeatures per target. 
    Forecast(testing input)
        Performs testing using a new set of inputs
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    
    def __init__(self, ld_coef, tau_coef, reg, washout, nfeatures=None, max_iter=1000, tol=1e-4):
        
        # Instance attributes that are user defined
        self.ld_coef = ld_coef                      # Gram matrix hyperparameter (has to be between 0 and 1)
        self.tau_coef = tau_coef                    # Gram matrix hyperparameter (has to be between 0 and 1)
        self.reg = reg                              # Regularisation paramter to be used in Lasso regression
        self.washout = washout                      # Training washout length
        self.nfeatures = nfeatures                  # Number of nonzero features in Lasso regression
        self.max_iter = max_iter                    # Max iterations to be used in elastic net by sklearn
        self.tol = tol                              # Tolerance for elastic net in sklearn
        
        # Instance attributes storing arrays created by methods
        self.Gram = None                            # Store Gram matrix throughout training and forecasting
        self.training_input = None                  # Stores training input seen during training
        self.alpha = None                           # Stores outcome of regression - weights
        self.alpha0 = None                          # Stores outcome of regression - shift
        self.regisFound = None                      # Stores whether a good reg param for desired number of features was found for each target
        
        # Instance attributes storing data dependent values created by methods
        self.ninputs = None                         # Store training input length
        self.ntargets = None                        # Store number of targets output in testing
        self.nhorizon = None                        # Store length of forecasting horizon
        self.ld = None                              # Store the ld value used to build the Gram matrix
        self.tau = None                             # Store the tau value sued to build the Gram matrix
        self.M = None                               # Store the uniform bound of the training input data
        
        
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
        VolterraLasso : class_instance
            VolterraLasso object with training attributes initialised
        """
        
        # Assign as instance attribute that are based on training input data
        self.training_input = training_input 
        self.ninputs = training_input.shape[0]
        
        # Assign instance attributes that are related to the training teacher data
        self.ntargets = training_teacher.shape[1]
        
        # If nfeatures not provided, then reg should be scalar or list of scalars
        if self.nfeatures is None:   
            
            # If reg provided is scalar, repeat the same reg for each target
            if np.isscalar(self.reg):
                self.reg = [self.reg] * self.ntargets
            # If reg provided is a numpy array, check it has the right shape
            elif isinstance(self.reg, np.ndarray):
                if self.reg.shape != (self.ntargets, ):
                    raise TypeError("nfeatures not provided but the numpy array given as regularisation has the wrong shape")    
            # If reg provided is a list, check it is a list of scalars and has the right shape
            elif isinstance(self.reg, list):
                for reg_entry in self.reg:
                    if not np.isscalar(reg_entry):
                        raise TypeError("nfeatures not provided but the list given as regularisation contains a nonscalar")
                if len(self.reg) != self.ntargets:
                    raise TypeError("nfeatures not provided but the list given as regularisation has the wrong length")
        
        # If nfeatures is provided, then reg should be list of numpy arrays to iterate over per target or a single 1D numpy array
        if self.nfeatures is not None:
            
            # Check that reg provided is not a scalar
            if np.isscalar(self.reg):
                raise TypeError("nfeatures was provided but the reg is a scalar when it should be a range or array of ranges")
            # If reg provided is a list, check each reg entry is a 1D array and that the list is the right size
            elif isinstance(self.reg, list):
                for reg_entry in self.reg:
                    if not isinstance(reg_entry, np.ndarray):
                        raise TypeError("nfeatures was provided but the list of reg entries contains a non numpy array")
                    if reg_entry.ndim != 1:
                        raise TypeError("nfeatures was provided by the one of the reg entries is a numpy array that is not 1D")
                if len(self.reg) != self.ntargets:
                    raise ValueError("nfeatures was provided but the list of reg entries is the wrong length")
            # If reg provided is an array, check that it is 1D. If it is, multiply it per target
            elif isinstance(self.reg, np.ndarray):
                if self.reg.ndim != 1:
                    raise TypeError("nfeatures was provided but the reg array provided was not 1D")
                self.reg = [self.reg] * self.ntargets
                
            # Check that nfeatures is an int or a list of integers
            if isinstance(self.nfeatures, int):
                self.nfeatures = [self.nfeatures] * self.ntargets    # Cast integer into list
            # Check that the list is one of integers and that the shape is correct
            elif isinstance(self.nfeatures, list):
                for nfeature_entry in self.nfeatures:
                    if not isinstance(nfeature_entry, int):
                        raise TypeError("nfeatures provided contains a noninteger")
                if len(self.nfeatures) != self.ntargets:
                    raise ValueError("nfeatures provided contains the wrong number of values")
                
        # Check that the training input and training teacher sizes are the same
        nteacher = training_teacher.shape[0]
        if self.ninputs != nteacher:
            raise ValueError("The size of the training teacher and training inputs do not match")
        
        # Check that the washout is not greater than the size of the inputs
        if self.washout > self.ninputs:
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
        
        # Remove washout and record shape of Gram
        Gram_train = self.Gram[self.washout: , self.washout: ]
        nGram_train = Gram_train.shape[0]
        
        # Initialise lasso output store for all targets
        alpha = np.zeros((nGram_train, self.ntargets))
        alpha0 = np.zeros((self.ntargets, ))
        nfeatures = np.zeros((self.ntargets, ))
        
        # If nfeatures was not provided, then do regular Lasso regression
        if self.nfeatures is None:
            
            # Iterate through each target then run sklearn lasso regression
            for target in range(self.ntargets):
                
                # Define target's regularisation parameter
                target_reg_param = self.reg[target]
                
                # Perform Lasso least squares regularisation using the sklearn lasso model
                lasso_target = Lasso(target_reg_param, max_iter=self.max_iter, tol=self.tol).fit(Gram_train, training_teacher_washed[:, target])
                alpha_target, alpha0_target = lasso_target.coef_, lasso_target.intercept_
                
                # Fill up the regression parameters with the regression result
                alpha[:, target] = alpha_target
                alpha0[target] = alpha0_target
                nfeatures[target] = len(np.nonzero(alpha[:, target])[0])
        
        # If nfeatures was provided, then repeat Lasso regression over the reg entries until find one below nfeatures
        if self.nfeatures is not None:
            
            # Initialise regression information store for all targets
            reg = np.zeros((self.ntargets, ))
            regisFound = [False] * self.ntargets
        
            # Iterate through each target then run sklearn lasso regression
            for target in range(self.ntargets):
                for target_reg in self.reg[target]:
                    
                    lasso_target = Lasso(target_reg, max_iter=self.max_iter, tol=self.tol).fit(Gram_train, training_teacher_washed[:, target])
                    alpha_target, alpha0_target = lasso_target.coef_, lasso_target.intercept_
                
                    # Check for the number of nonzero coefficients
                    n_nonzeros_target = len(np.nonzero(alpha_target)[0])
                    if n_nonzeros_target <= self.nfeatures[target]:
                        # Fill up the regression parameters with the regression result
                        nfeatures[target] = n_nonzeros_target
                        reg[target] = target_reg
                        regisFound[target] = True
                        alpha[:, target] = alpha_target
                        alpha0[target] = alpha0_target
                        break
                
                # Check if no appropriate regularisation parameter for the target was found
                if regisFound[target] is False:  # Means no regularisation parameter was chosen 
                    # Default to the last found regularisation parameter
                    nfeatures[target] = n_nonzeros_target
                    reg[target] = target_reg
                    alpha[:, target] = alpha_target
                    alpha0[target] = alpha0_target
        
            # Assign as instance attribute only relevant to this lasso
            self.regisFound = regisFound
        
        # Assign as instance attributes the Lasso results
        self.alpha = alpha
        self.alpha0 = alpha0    
        self.nfeatures = nfeatures
        
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