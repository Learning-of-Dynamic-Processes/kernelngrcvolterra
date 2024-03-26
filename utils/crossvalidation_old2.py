import multiprocessing
import numpy as np
from itertools import product
from utils.errors import calculate_mse
from utils.normalisation import normalise_arrays

class CrossValidate:
    
    """
    Cross validation class with multiprocessing.  
    
    Attributes
    ----------
    validation_parameters : list of ints, optional
        List containing size of a single training fold, size of validation fold and size of jump between folds (default: None)
        If None, validation parameter defaults to 0.8 of the data for training, 0.1 of the data for validation, and 0.1 of the remaining data as jump size.
        If desire to have only one fold, set size of jumps to be 0 or any number larger than (training_input - training_size - validation_size + 1).
    validation_type : str, optional
        The manner in which the training and validation folds move with each fold iteration. 
        Rolling means start of training fold jumps with jump size. 
        Expanding means start of training fold always stays the same but validation fold start jumps with jump size. 
        Standard k-fold cross-validation is supported by the rolling option, and choosing validation parameters all equal.
        Options: {"rolling", "expanding"}. (default: "rolling").
    task : str, optional 
        Whether to perform forecasting or path continuation. Options: {"Forecast", "PathContinue"}. (default: "PathContinue").
        Estimator passed must have methods called these names, for whichever option is chosen. 
    norm_type : str, optional
        Normalisation method called based on the options available in normalise_arrays. Normalisation is carried out over each fold individually.
        If overall normalisation is preferred, choose None for this norm_type and normalise data before input.
        Options: {"NormStd", "MinMax", "ScaleL2", "ScaleL2Shift", None}. (default: None).
    ifPrint : bool, optional
        Whether to print estimators and error for each parameter. (default: False) .
    Methods
    -------
    crossvalidate_per_parameters(estimator, data_in, target, estimator_parameters)
        Runs the cross validation process for a single set of parameters into the estimator
    crossvalidate_multiprocessing(estimator, data_in, target, param_ranges, param_names, param_add, num_processes=4)
        Runs cross validation for a range of input parameters. Uses multiprocessing. 
    """
    
    def __init__(self, validation_parameters=None, validation_type="rolling", task="PathContinue", norm_type=None, ifPrint=False):
        
        self.validation_parameters = validation_parameters
        self.validation_type = validation_type
        self.task = task
        self.norm_type = norm_type        
        self.ifPrint = ifPrint
        self.MinMax_range = (0, 1)
        
    def crossvalidate_per_parameters(self, estimator, cv_datasets, estimator_parameters, param_id):

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
        
        # Iterate through the normalised training and validation sets, perform estimation, compute errors
        validation_errors = []
        for start_id in range(len(cv_datasets)):
            
            # Initalise each cv_datasets element
            train_in, train_target, validation_in, validation_target = cv_datasets[start_id][0]
            shift, scale = cv_datasets[start_id][1], cv_datasets[start_id][2]
            
            # Instantiate the estimator to train and test on the training and validation sets
            Estimator = estimator(*estimator_parameters)
            
            # For path continuation task training and validating
            if self.task == "PathContinue":
                output = Estimator.Train(train_in, train_target).PathContinue(train_target[-1], validation_target.shape[0])
            # For general forecasting task training and validating
            elif self.task == "Forecast":
                output = Estimator.Train(train_in, train_target).Forecast(validation_in)
            else:   # Raise error for any other task
                raise NotImplementedError("Task on which to cross validate is not available")

            # Compute mse of method's output using the validation target
            fold_mse = calculate_mse(validation_target, output, shift, scale)
            validation_errors.append(fold_mse)

        # Compute total average mse across validation to measure performance of hyperparameter choice
        mean_validation_error = np.mean(validation_errors)
        
        # Print parameter id if desired to track progress
        if self.ifPrint == True:
            print(param_id)   
        print(estimator_parameters, mean_validation_error)
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
        
        # Define the length of the incoming data inputs
        input_size = len(data_in)
        
        # Check that data input and targets are the same size
        if len(target) != input_size:
            raise ValueError("Target data and input data are not of the same size")

        # If validation parameters are not provided, use the defaults
        if self.validation_parameters is None:
            
            # Default sizes (0.8, 0.1, 0.1)
            train_size = int(0.8 * input_size)
            validation_size = int(0.1 * input_size)
            jump_size = int((input_size - train_size - validation_size) * 0.1)
            
            # Account for when jump size becomes 0
            if jump_size == 0:
                jump_size = input_size - train_size - validation_size + 1
                
            # Assign them as instance attribute
            self.validation_parameters = [train_size, validation_size, jump_size]
        
        # If validation parameters are provided, roll them out.
        if self.validation_parameters is not None:
            
            # Roll out provided parameters
            train_size, validation_size, jump_size = self.validation_parameters
            
            # Check if user-provided jump size is 0
            if jump_size == 0:
                jump_size = input_size - train_size - validation_size + 1
                self.validation_parameters = [train_size, validation_size, jump_size]
        
        # Use the validation sizes to define the number of folds
        nstarts = int((input_size - train_size - validation_size)/jump_size) + 1

        # Iterate through data in, training method on each fold then compute validation results
        cv_datasets = []
        for start_id in range(0, nstarts):
            
            # Define the starting index
            start = start_id * jump_size

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
            normalisation_output = normalise_arrays(data_ls, norm_type=self.norm_type, minmax_range=self.MinMax_range)
            
            # Append to cv_datas the normalised output
            cv_datasets.append(normalisation_output)

        # Define dictionary store for best parameters and the corresponding error
        best_parameters = {'validation_error': float('inf')}
        for param_name in param_names:
            best_parameters[param_name] = None
  
        # Intialise multiprocessing Pool object (object to execute function across multiple input values)
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Unpack parameter range values and create range of inputs to the cross_validate_per_parameters function
        combinations = []
        parameter_combinations = list(product(*param_ranges))
        for param_id, param_choice in enumerate(parameter_combinations):
            input_comb = (estimator, cv_datasets, (*param_choice, *param_add), param_id)
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
