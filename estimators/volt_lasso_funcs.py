
### Volterra class code based on the methods detailed in https://arxiv.org/abs/2212.14641 
### Additionally introduce Lasso regression such that max number of nonzero features is controlled

#TODO recomment the code
import numpy as np
from sklearn.linear_model import Lasso
from time import process_time


class VolterraFeatureLasso:
    
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
    ndesired_features : int
        Max number of nonzero coefficients in the regression weights during Lasso
    reg_range : array_like
        The regularisation range used by Lasso to reduce coefficients to zero
    lasso_max_iter : int, optional
        Number of iterations by sklearn Lasso (default 1000, same as sklearn)
    lasso_tol : float, optional
        Tolerance for optimisation (duality, updates) for sklearn Lasso (default 1e-4, same as sklearn)
        
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher in the Volterra method
    Forecast(testing input)
        Performs testing using a new set of training input 
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    def __init__(self, ld_coef, tau_coef, washout, ndesired_features, reg_range, lasso_max_iter=1000, lasso_tol=1e-4):
        
        
        # Instance attributes that are user defined
        self.ld_coef = ld_coef                      # Gram matrix hyperparameter (has to be between 0 and 1)
        self.tau_coef = tau_coef                    # Gram matrix hyperparameter (has to be between 0 and 1)
        self.washout = washout                      # Training washout length
        self.ndesiredfeatures = ndesired_features   # Store the max number of nonzero coefficients from Lasso regression
        self.reg_range = reg_range                  # Array in which to search for regularisation that gives the desired number of coefficients
        self.lasso_max_iter = lasso_max_iter        # Number of iterations by sklearn Lasso (same as default)
        self.lasso_tol = lasso_tol                  # Tolerance for optimisation (duality, updates) for sklearn Lasso (same as default)
            
        # Instance attributes storing arrays created by methods
        self.Gram = None                            # Store Gram matrix throughout training and forecasting
        self.training_input = None                  # Stores training input seen during training
        self.alpha = None                           # Stores outcome of regression - weights
        self.alpha0 = None                          # Stores outcome of regression - shift
        self.regisFound = None                      # Stores whether a good reg param for desired number of features was found for each target
        self.reg_values = None                      # Store the regularisation values chosen by Lasso training
        
        # Instance attributes storing data dependent values created by methods
        self.ninputs = None                         # Store training input length
        self.nfeatures = None                       # Store number of input features seen from training
        self.nfeatures_seen = None                  # Store number of input features actually used in training
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
        VolterraFeatureLasso : class_instance
            VolterraFeatureLasso object with training attributes initialised
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
        
        # Initialise lasso output store for all targets
        alpha_lasso = np.zeros((nGram_train, self.ntargets))
        alpha0_lasso = np.zeros((self.ntargets, ))
        nfeatures_seen_lasso = np.zeros((self.ntargets, ))
        reg_values = np.zeros((self.ntargets, ))
        regisFound = [False] * self.ntargets
        
        # Iterate through each target then run sklearn lasso regression
        for target in range(self.ntargets):
            for target_reg in self.reg_range:
                
                lasso_target = Lasso(target_reg, max_iter=self.lasso_max_iter, tol=self.lasso_tol).fit(Gram_train, training_teacher_washed[:, target])
                alpha_target, alpha0_target = lasso_target.coef_, lasso_target.intercept_
            
                # Check for the number of nonzero coefficients
                n_nonzeros_target = len(np.nonzero(alpha_target)[0])
                if n_nonzeros_target <= self.ndesiredfeatures:
                    # Fill up the regression parameters with the regression result
                    nfeatures_seen_lasso[target] = n_nonzeros_target
                    reg_values[target] = target_reg
                    regisFound[target] = True
                    alpha_lasso[:, target] = alpha_target
                    alpha0_lasso[target] = alpha0_target
                    break
            
            # Check if no appropriate regularisation parameter for the target was found
            if regisFound[target] is False:  # Means no regularisation parameter was chosen 
                # Default to the last found regularisation parameter
                nfeatures_seen_lasso[target] = n_nonzeros_target
                reg_values[target] = target_reg
                alpha_lasso[:, target] = alpha_target
                alpha0_lasso[target] = alpha0_target
    
        # Assign as instance attributes the Lasso results
        self.alpha = alpha_lasso
        self.alpha0 = alpha0_lasso
        self.reg_values = reg_values
        self.regisFound = regisFound
        
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


