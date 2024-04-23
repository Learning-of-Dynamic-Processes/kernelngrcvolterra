import multiprocessing
import numpy as np
from itertools import product
from utils.errors import calculate_mse, calculate_wasserstein1err, calculate_specdensloss
from utils.normalisation import normalise_arrays

class CrossValidate:
    
    """
    Cross-validation that utilises Python's built-in multiprocessing package. 
    
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
    error_type : str, optional
        The type of error function to use in computing error in each fold. Options: {"meansquare", "wasserstein1", "specdens"}. (default: "meansquare")
    minmax_range : tuple, optional
        Tuple containing the desired min and max when normalising using norm_type="MinMax". (default: (0, 1))
    log_interval : int, optional
        The interval at which results are saved into a txt file and intermediate best parameters are printed. (default: 10)
        
    Methods
    -------
    split_data_to_folds(data_in, target)
        Splits data into training and validation folds. Outputs based on utils.normalisation
    test_parameter_set(test_parameter_inputs)
        Helper function to test each parameter set. Used in cross validation. 
    crossvalidate(estimator, cv_datasets, param_ranges, param_add, num_processes, chunksize)
        Runs cross validation for range of input parameters. Uses multiprocessing to parallelise over grid. 
    """


    def __init__(self, validation_parameters=None, validation_type="rolling", 
                       task="PathContinue", norm_type=None, error_type="meansquare", 
                       minmax_range=(0, 1), log_interval=10):
        
        self.validation_parameters = validation_parameters
        self.validation_type = validation_type
        self.task = task
        self.norm_type = norm_type
        self.error_type = error_type
        self.minmax_range = minmax_range
        self.log_interval = log_interval


    def split_data_to_folds(self, data_in, target):
        
        """
        Takes input and target data and splits it into folds depending on the validation parameters and validation type. 
        Normalises the data using the training fold, for each training, validation combination. Normalisation depends on norm_type. 
        
        Parameters
        ----------
        data_in : array-like
            Input data. Will be split up into multiple training and validation folds with defined jump size between each fold.
        target : array-like
            Target data. Splits in the same way (same indices) as the input data. 
        
        Returns
        -------
        cv_datasets : array-like
            List of arrays where each array contains the training input fold, training target fold, validation input fold, and validation target fold,
            as well as the shifts and scales used by the normalisation function. Will be unpacked when running test_parameter_set. 

        Raises
        ------
        ValueError
            Throws error when the target and input size provided do not match. 
        NotImplementedError
            Throws error when the method of validation provided is not "rolling" or "expanding"     
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

            # Define the training and validation data for expanding validation type
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
            normalisation_output = normalise_arrays(data_ls, norm_type=self.norm_type, minmax_range=self.minmax_range)
            
            # Append to cv_datas the normalised output
            cv_datasets.append(normalisation_output)
        
        return cv_datasets
        
        
    def test_parameter_set(self, test_parameter_set_inputs):
        
        """
        Helper function to crossvalidate. Takes one set of parameters and runs the chosen estimator on every fold combination in cv_datasets.
        Inputs should be a list of the inputs because multiprocessing.pool.imap_unordered is used. 

        Parameters
        ----------
        test_parameter_set_inputs : array-like
            Should be a list containing in order the
            - estimator class that has the methods Train and PathContinue/Forecast depending on which task has been chosen
            - cv_datasets list that contains the normalisation outputs of each fold. Use split_data_to_folds to generate. 
            - estimator_parameters list of parameter inputs for the estimator. Should be in the same order as how the estimator is defined. 
    
        Returns
        -------
        estimator_parameters : tuple
            Tuple containing the parameters that were used as inputs into the estimator parameters.
        mean_validation_error : float
            The average validation error over each of the folds. Error measure depends on error_type attribute.

        Raises
        ------
        NotImplementedError
            Task is not either PathContinue or Forecast. Error is not mean-squared error or wasserstein-1 error
        """
        
        estimator, cv_datasets, estimator_parameters = test_parameter_set_inputs
        
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
            if self.error_type == "meansquare":
                fold_err = calculate_mse(validation_target, output, shift, scale)
            elif self.error_type == "wasserstein1":
                fold_err = calculate_wasserstein1err(validation_target, output, shift, scale)
            elif self.error_type == "specdens":
                fold_err = calculate_specdensloss(validation_target, output, shift, scale)
            else: 
                raise NotImplementedError("Error type is not available")
            
            # Append the validation error to fold_mse store
            validation_errors.append(fold_err)

        # Compute total average mse across validation to measure performance of hyperparameter choice
        mean_validation_error = np.mean(validation_errors)

        return estimator_parameters, mean_validation_error

    
    def crossvalidate(self, estimator, cv_datasets, param_ranges, param_add, num_processes=4, chunksize=1):
        
        """
        Crossvalidate with multiprocessing on test_parameter_set. 
        Uses imap_unordered to be able to access and store result as they arrive. 
        Stores all errors with the parameters in a cv.txt file. If run again, cv.txt file should be cleared or saved elsewhere. 
        Otherwise, a separator is written to separate different runs, but does not store information about the run itself. 
        Logs information to the file at log_interval intervals. 

        Parameters
        ----------
        estimator : class object
            Estimator for which to tune hyperparameters for. Has to have methods Train, PathContinue/Forecast. 
            Order of inputs during initialisation need to be of the same order as param_ranges and param_add when concatenated
        cv_datasets : array-like
            List of arrays containing the folds, shifts and scale outputs from normalisation. Can use split_data_to_folds function to generate.
        param_ranges : list of arrays or tuple of arrays
            List or tuple of the parameter values to cross validate for. Must be in same order as how estimator takes as inputs.
        param_add : list
            The additional parameters taken in by the estimator that are NOT being cross validated for.
        num_processes : int, optional
            Number of processes to split the work over. Wraps equivalent processes arg in multiprocessing.Pool (default: 4).
        chunksize : int, optional
            How the iterables are split approximately and each chunk is submitted as a separate task. 
            Wraps equivalent chunksize arg in pool.imap_unordered. (default: 1)
        
        Returns
        -------
        min_error : float
            The best error found.
        min_parameter : tuple
            Tuple of estimator parameters that gave the best error. 
        """
        
        # Unpack parameter range values and create range of inputs to the cross_validate_per_parameters function
        combinations = []
        parameter_combinations = list(product(*param_ranges))
        for param_choice in parameter_combinations:
            input_comb = (estimator, cv_datasets, (*param_choice, *param_add))
            combinations.append(input_comb)
        
        # Create a process pool with num_processes workers
        pool = multiprocessing.Pool(processes=num_processes)

        # Issue tasks, yielding results as soon as they are available using imap_unordered
        partial_results = []        # Stores partial results in case of crash
        count = 0
        min_error = float('inf')
        min_parameter = None
        for result in pool.imap_unordered(self.test_parameter_set, combinations, chunksize=chunksize):
            
            # Append the results to each list
            partial_results.append(result)
            
            # Count the number of results and track best parameters
            count = count + 1
            if result[1] <= min_error:
                min_error = result[1]
                min_parameter = result[0]
            
            # At every log_interval, log the partial results and empty it
            if len(partial_results) % self.log_interval == 0:
                # Write the partial results to the cv.txt file
                with open("cv.txt", "a") as file:
                    for partial_result in partial_results:
                        file.write(f"{partial_result}\n")
                    partial_results = []
                # Print current progress with best found
                print(f"Reached {count} hyperparameters") 
                print(f"Best estimate so far: {min_error} with {min_parameter}")   
            
        # Log the remaining results not covered in log_interval
        with open("cv.txt", "a") as file:
            for partial_result in partial_results:
                file.write(f"{partial_result}\n")
            file.write("-" * 40 + "\n")    # Separator in case one forgets to erase cv.txt before running
                        
        # Prevent further execution of tasks 
        pool.close()
        # Wait for worker processes to exit
        pool.join()
        
        return min_error, min_parameter
        
