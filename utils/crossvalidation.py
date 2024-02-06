import multiprocessing
import numpy as np
from itertools import product
from utils.errors import calculate_mse
from utils.normalisation import normalise_arrays

class CrossValidate:
    
    """ - validation_parameters: arr of ints (train_size, validation_size, start_pts, start_jump). Integers that determine the size of 
        each training set size, the validation set size, the range spanned by the starting points of the training set, the distance between
        any two training starting points. 
        - validation_type: str {"rolling", "expanding"}. Determines the way to split the training and validation sets.
        - norm_type: str {"NormStd", "MinMax", "ScaleL2", "ScaleL2Shift}. Normalisation type. 
        - MinMax_range: tuple (desired_min, desired_max). Only kicks in if norm_type is "MinMax".
    """
    
    def __init__(self, validation_parameters=None, validation_type="rolling", norm_type="ScaleL2"):
        self.validation_parameters = validation_parameters
        self.validation_type = validation_type
        self.norm_type = norm_type        
        if self.norm_type == "MinMax":
            self.MinMax_range = (0, 1)
    
    def crossvalidate_per_parameters(self, method, data_in, target, method_parameters):
        
        ''' 
        Function that takes a single combination of parameter choices then tests them against a defined validation set. 
        This validation set differs from typical cross-validation methods in that validation is done in an autonomous manner.
        And the validation sets vary in starting point that might not be the starting of a fold necessarily. 
        This provides greater stability for the choice of parameters. 

        Parameters:
        - method: callable. Should be the method that we wish to cross validate for. This function must take in arguments in the
        following manner method(training input array, training teacher array, testing input array, method parameters)
        - data_in: array (nsamples, nfeatures). An array of data inputs that will be used for training
        - target: array (nsamples, ntargets). An array of training teacher values. 
        - method_parameters: array. Inputs for the callable method. For example, regression parameters. 
        Must follow the order in used by method. Method must take in the method parameters as a tuple. 
       
        Returns
        - Average validation error over all validation folds.
        '''

        # Define the length of the incoming data inputs
        input_size = len(data_in)
        # Check that data input and targets are the same size
        if len(target) != input_size:
            raise ValueError("Target data and input data are not of the same size")

        # If validation parameters are not provided, use the defaults
        if self.validation_parameters is None:
            train_size = 0.8 * input_size
            validation_size = 0.1 * input_size
            start_pts = 0.4 * input_size - validation_size
            start_jump = 0.2 * start_pts
        
        # If validation parameters are provided, roll them out.
        if self.validation_parameters is not None:
            train_size, validation_size, start_pts, start_jump = self.validation_parameters

        # Define store for validation errors
        validation_errors = []

        # Iterate through data in, training method on each fold then compute validation results
        n_folds = 0     # records number of folds
        for start in range(0, start_pts, start_jump):

            # Define the data sets for validation depending on the cv type chosen

            # Rolling window cross validation moves the starting points so the train size stays constant
            if self.validation_type == "rolling":
                train_in = data_in[start : start+train_size]
                train_target = target[start : start+train_size]
                validation_in = data_in[start+train_size : start+train_size+validation_size]
                validation_target = target[start+train_size : start+train_size+validation_size]

            # Expanding window cross validation allows the train size to grow with each start
            elif self.validation_type == "expanding":
                train_in = data_in[0 : start+train_size]
                train_target = target[0 : start+train_size]
                validation_in = data_in[start+train_size : start+train_size+validation_size]
                validation_target = target[start+train_size : start+train_size+validation_size]

            # Raise error if cross validation type input is incorrect
            else:
                raise NotImplementedError("Validation method of splitting dataset is not available")
            
            # Calls normalisation function to normalise a single validation iteration
            data_ls = [train_in, train_target, validation_in, validation_target]
            train_in, train_target, validation_in, validation_target = normalise_arrays(data_ls, norm_type=self.norm_type, MinMax_range=self.MinMax_range)

            # Pass defined datasets into method together with parameters
            output = method(train_in, train_target, validation_in, validation_size, method_parameters)

            # Compute mse of method's output using the validation target
            fold_mse = calculate_mse(output, validation_target)
            validation_errors.append(fold_mse)

            # Increment counter for number of folds
            n_folds = n_folds + 1
        
        # Compute total average mse across validation to measure performance of hyperparameter choice
        mean_validation_error = np.mean(validation_errors)
        
        return mean_validation_error

    def crossvalidate_multiprocessing(self, method, data_in, target, param_ranges, param_names, num_processes=4):
        
        # Define dictionary store for best parameters and the corresponding error
        best_parameters = {'validation_error': float('inf')}
        for param_name in param_names:
            best_parameters[param_name] = None

        # Intialise multiprocessing Pool object (object to execute function across multiple input values)
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Unpack parameter range values and create range of inputs to the cross_validate_per_parameters function
        combinations = []
        parameter_combinations = product(*param_ranges)
        for param_choice in parameter_combinations:
            input_comb = (method, data_in, target, param_choice)
            combinations.append(input_comb)
        
        # Perform parallelised cross validation using the defined pool object
        result = pool.starmap(self.crossvalidate_per_parameters, combinations)
        
        # Iterate through the combinations to obtain the results
        for combination_id, combinations_choice in enumerate(combinations):
            
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
        
        return best_parameters, combinations