import multiprocessing
import numpy as np
from itertools import product
from utils.errors import calculate_mse
from utils.normalisation import normalise_arrays

class CrossValidate:
    
    """
    Class to wrap cross validation with multiprocessing, specially for estimators that do path continuing tasks and
    need to be robust against initial conditions. 
    
    Attributes
    ----------
    validation_parameters : array_like of ints, optional
        Sizes in which to split a single training fold, validation fold and the number of golds (default None)
        If None, then validation parameter default to 0.8 of the data for the training fold, 0.1 for the validation fold, and however many folds so that
        each starting point of the thold is 0.1 of the remaining 0.1 of the dataset
    validation_type : str, optional
        The manner in which the training and validation folds move with each fold iteration. Options: {"rolling", "expanding"}, (default "rolling")
    task : str, optional 
        Whether to perform forecasting or path continuation. Options: {"Forecast", "PathContinue"}, (default "PathContinue")
    norm_type : str, optional
        Normalisation method called based on the options available in normalise_arrays. Normalisation is carried out over each fold individually.
        If an overall normalisation is preferred, choose None for this norm_type and normalise data before input.
        Options: {"NormStd", "MinMax", "ScaleL2", "ScaleL2Shift", None}, (default "ScaleL2")
    
    Methods
    -------
    crossvalidate_per_parameters(estimator, data_in, target, estimator_parameters)
        Runs the cross validation process for a single set of parameters into the estimator
    crossvalidate_multiprocessing(estimator, data_in, target, param_ranges, param_names, param_add, num_processes=4)
        Runs cross validation for a range of input parameters. Uses multiprocessing. 
    """
    
    def __init__(self, validation_parameters=None, validation_type="rolling", task="PathContinue", norm_type="ScaleL2"):
        
        self.validation_parameters = validation_parameters
        self.validation_type = validation_type
        self.task = task
        self.norm_type = norm_type        
        self.MinMax_range = (0, 1)
    
    def crossvalidate_per_parameters(self, estimator, data_in, target, estimator_parameters):

        """
        Runs the cross validation process for a single set of parameters into the estimator

        Parameters
        -------
        data_in : array_like
            Full input data. Will be split up to form the folds
        target : array_like
            Full target data. Will be split up to form the folds
        estimator_parameters : array
            Parameters that will be rolled out and passed into the estimator
        
        Returns
        -------
        mean_validation_error : float
            Average mean square error between target and output over all folds
        """
        
        # Define the length of the incoming data inputs
        input_size = len(data_in)
        
        # Check that data input and targets are the same size
        if len(target) != input_size:
            raise ValueError("Target data and input data are not of the same size")

        # If validation parameters are not provided, use the defaults
        if self.validation_parameters is None:
            train_size = int(0.8 * input_size)
            validation_size = int(0.1 * input_size)
            nstarts = int((input_size - train_size - validation_size) * 0.1)
            # Handle the case where nstarts happens to be 0
            if nstarts == 0: 
                nstarts = 1
            # Assign them as instance attribute
            self.validation_parameters = [train_size, validation_size, nstarts]
        
        # If validation parameters are provided, roll them out.
        if self.validation_parameters is not None:
            train_size, validation_size, nstarts = self.validation_parameters
            # Handles the case when the number of starts will cause training + validation window to exceed data
            if nstarts > (input_size - train_size - validation_size):
                raise ValueError("The number of starting points is too great")
            # Handle the case when the provided number of starts is 0
            if nstarts == 0:
                raise ValueError("The number of starts needs to be at least 1")

        # Use the validation sizes to define the size between starting points
        start_size = int((input_size - train_size - validation_size)/nstarts)

        # Define store for validation errors
        validation_errors = []
        
        # Iterate through data in, training method on each fold then compute validation results
        n_folds = 0     # records number of folds
        for start_id in range(0, nstarts):
            
            # Define the starting index
            start = start_id * start_size
            
            # Define the training and validation data for rolling validation type
            if self.validation_type == "rolling":
                # Rolling window cross validation moves the starting points so the train size stays constant
                train_in = data_in[start : start+train_size]
                train_target = target[start : start+train_size]
                validation_in = data_in[start+train_size : start+train_size+validation_size]
                validation_target = target[start+train_size : start+train_size+validation_size]
                
                # Handle the case where the last block captures the remainder of the data
                if start_id == nstarts-1:   # Check when it is the last block
                    train_in = data_in[start : start+train_size]
                    train_target = target[start : start+train_size]
                    validation_in = data_in[start+train_size : ]
                    validation_target = target[start+train_size : ]

            # Define the training and validaiton data for expanding validation type
            elif self.validation_type == "expanding":
                # Expanding window cross validation allows the train size to grow with each start
                train_in = data_in[0 : start+train_size]
                train_target = target[0 : start+train_size]
                validation_in = data_in[start+train_size : start+train_size+validation_size]
                validation_target = target[start+train_size : start+train_size+validation_size]
                
                # Handle the case where the last block captures the remainder of the data
                if start_id == nstarts-1:   # Check when it is the last block
                    train_in = data_in[0 : start+train_size]
                    train_target = target[0: start+train_size]
                    validation_in = data_in[start+train_size : ]
                    validation_target = target[start+train_size : ]

            # Raise error if cross validation type input is incorrect
            else:
                raise NotImplementedError("Validation method of splitting dataset is not available")
            
            # Calls normalisation function to normalise a single validation iteration
            data_ls = [train_in, train_target, validation_in, validation_target]
            normalisation_output = normalise_arrays(data_ls, norm_type=self.norm_type, MinMax_range=self.MinMax_range)
            train_in, train_target, validation_in, validation_target = normalisation_output[0]
            shift, scale = normalisation_output[1], normalisation_output[2]

            # Instantiate the estimator to train and test on the training and validation sets
            Estimator = estimator(*estimator_parameters)
            # For path continuation task training and validating
            if self.task == "PathContinue":
                output = Estimator.Train(train_in, train_target).PathContinue(train_target[-1], validation_target.shape[0])
            # For general forecasting task training and validating
            elif self.task == "Forecast":
                output = Estimator.Train(train_in, train_target).Forecast(validation_in)
            else:
                raise NotImplementedError("Task on which to cross validate is not available")
            
            # Compute mse of method's output using the validation target
            fold_mse = calculate_mse(output, validation_target, shift, scale)
            validation_errors.append(fold_mse)

            # Increment counter for number of folds
            n_folds = n_folds + 1
        
        # Compute total average mse across validation to measure performance of hyperparameter choice
        mean_validation_error = np.mean(validation_errors)
        
        return mean_validation_error

    
    def crossvalidate_multiprocessing(self, estimator, data_in, target, param_ranges, param_names, param_add, num_processes=4):
        
        """
        Runs cross validation for a range of input parameters. Uses multiprocessing.
        
        Parameters
        ----------
        estimator : class
            Estimator for which to tune hyperparameters for. Has to have methods Train, PathContinue/Forecast. 
            Order of inputs during initialisation need to be of the same order as param_ranges and param_add when concatenated
        data_in : array_like
            Full input data. Will be split up to form the folds
        target : array_like
            Full target data. Will be split up to form the folds
        param_ranges : list of arrays or tuple of arrays
            List or tuple of the parameter values to cross validate for. Must be in the same order as param_names.
        param_names : list of str
            List of names of the parameter values to cross validate for. Must be in the same order as param_ranges.
        param_add : list
            The additional parameters taken in by the estimator that are NOT being cross validated for
            
        Returns
        -------
        best_parameters : dict
            Dictionary based on param_names of the best parameters found
        parameter_combinations : array_like
            The parameter values that were tested over, in order
        result : array_like
            All the mean validation errors for each parameter values in parameter_combinations, in order
        """
    
        # Define dictionary store for best parameters and the corresponding error
        best_parameters = {'validation_error': float('inf')}
        for param_name in param_names:
            best_parameters[param_name] = None
  
        # Intialise multiprocessing Pool object (object to execute function across multiple input values)
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Unpack parameter range values and create range of inputs to the cross_validate_per_parameters function
        combinations = []
        parameter_combinations = list(product(*param_ranges))
        for param_choice in parameter_combinations:
            input_comb = (estimator, data_in, target, (*param_choice, *param_add))
            combinations.append(input_comb)
        
        # Perform parallelised cross validation using the defined pool object
        result = pool.starmap(self.crossvalidate_per_parameters, combinations)
        
        # Iterate through the combinations to obtain the results
        for combination_id, combinations_choice in enumerate(parameter_combinations):

            # Collect validation error for choice of parameter choice
            validation_error = result[combination_id]
            
            # Update best parameters if the current combination is better
            if validation_error < best_parameters['validation_error']:
                best_parameters['validation_error'] = validation_error
                for param_id, param_name in enumerate(param_names):
                    best_parameters[param_name] = combinations_choice[param_id]
                
                print("Intermediary Best Parameters:")
                print(f"Validation errors: {best_parameters['validation_error']}")
                for param_name in param_names:
                    print(f"{param_name}: {best_parameters[param_name]}")
                print("-" * 40)  # Separator for clarity
        
        pool.close()
        pool.join()
        
        return best_parameters, parameter_combinations, result
