
### SINDy code written as a class, adapted from Manjunath's https://github.com/MJSteynberg/ForecastingThroughCausalEmbedding/tree/main and
###                                                         https://pysindy.readthedocs.io/en/latest/api/pysindy.html 

import pysindy as ps
import numpy as np

class SINDyPolynomialSTLSQ:
    
    """
    A wrapper class for SINDy Polynomial STLSQ.
    
    Attributes
    ----------
    alpha : float
        Stores regression regularisation parameter in the LSQ
    threshold : float
        Stores the threshold at which smaller values are set to 0
    deg : int
        Sets highest degree of polynomials used
    dt : float
        Size of time steps between each training step
    
    Methods
    -------
    Train(training_input, training_teacher)
        Performs training using the training input against the training teacher in the SINDy method
    PathContinue(latest_input, nhorizon)
        Simulates forward in time using the latest input for nhorizon period of time
    """
    
    def __init__(self, alpha, threshold, deg, dt):
        
        # Instance attributes that are used defined
        self.alpha = alpha                  # Stores regression regularisation parameter in the LSQ
        self.threshold = threshold          # Stores the threshold at which smaller values are set to 0
        self.deg = deg                      # Sets highest degree of polynomials used
        self.dt = dt                        # Size of time steps between each training step
        
        # Instance attributes storing arrays/classes created by methods
        self.training = None                # Stores training data seen during training
        self.coefficients = None            # Stores outcome of regression weights
        self.model = None                   # Stores the SINDy model class
        
        # Instance attributes storing data dependent values created by methods
        self.ninputs = None                 # Store training data length
        self.nfeatures = None               # Store number of input features seen from training
        self.ntargets = None                # Store number of target features seen from training
        self.nhorizon = None                # Stores the forecasting horizon  
        
        # Instance attributes for additional SINDy parameters
        self.include_interaction = True     # To instruct SINDy Polynomial library to include cross terms
        self.interaction_only = False       # To instruct SINDy Polynomial library to use only cross terms
    
    
    # Function performs training using training data (computes derivatives)
    def Train(self, training_input, training_teacher):    # take in additional argument so it works with cv code
        
        """
        Performs training using the training input against the training teacher in the SINDy method
        
        Parameters
        ----------
        training_input : array_like
            Training input for training in SINDy. Must have format (nsamples, ndim)
        training_teacher : array_like
            Training teacher for training in SINDy. Must have format (nsamples, ndim)

        Returns
        -------
        SINDy : class_instance
            SINDy object with training attributes initialised
        """
        
        # Assign as instance attributes that are based on training data
        self.ninputs = training_input.shape[0]
        self.nfeatures = training_input.shape[1]
        self.ntargets = training_teacher.shape[0]
        
        # Define training set as it is what is needed to train SINDy - works on the assumption that the training teacher data is one off the training input
        training = np.zeros((self.ninputs+1, self.nfeatures))
        training[0:self.ninputs, :] = training_input
        training[self.ninputs, :] = training_teacher[-1]
        
        # Instantiate the STLSQ optimizer with the threshold and alpha values
        optimizer = ps.STLSQ(threshold=self.threshold, 
                             alpha=self.alpha)
        
        # Instantiate the library of choice 
        library = ps.PolynomialLibrary(degree=self.deg, 
                                       include_interaction=self.include_interaction, 
                                       interaction_only=self.interaction_only)
        
        # Instantiate the SINDy class object with the opimtizer and library and fit the model 
        self.model = ps.SINDy(optimizer=optimizer, 
                         feature_library=library)
        self.model.fit(training, t=self.dt)
        
        # Assign as instance attributes the model coefficients
        self.coefficients = self.model.coefficients()
        self.nfeatures_seen = len(np.nonzero(self.coefficients)[0])
        
        return self
    
    
    # Function that performs path continuation based on some latest input
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
        
        # Assign as instance attribute the forecasting horizon
        self.nhorizon = nhorizon
        
        # Write nhorizon and dt value back into the evaluation, shifted by 1 since SINDy includes latest input
        t_horizon = np.linspace((self.ninputs+1)*self.dt, (self.ninputs+nhorizon)*self.dt, self.nhorizon)
        
        # Simulate forward using the SINDy model class function
        output = self.model.simulate(latest_input, t_horizon)
        
        return output
