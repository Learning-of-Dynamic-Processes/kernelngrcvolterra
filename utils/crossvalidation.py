import multiprocessing
import numpy as np
from itertools import product
import time
from utils.errors import calculate_mse, calculate_nmse, calculate_wasserstein1err, calculate_specdens_periodogram_err
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
        Sklearn time series split can be replicated by choosing "expanding" and all validation parameters equal. 
        Options: {"rolling", "expanding"}. (default: "rolling").
    manage_remainder : bool, optional
        What to do with remainder values. If True, appends the remainder at the end, jump size remains the same. 
        If False, ignores the remainder data (so some data is not used). Default: True. 
    task : str, optional 
        Whether to perform forecasting or path continuation. Options: {"Forecast", "PathContinue"}. (default: "PathContinue").
        Estimator passed must have methods called these names, for whichever option is chosen. 
    norm_type_in : str, optional
        Normalisation method called based on the options available in normalise_arrays. 
        Normalisation is carried out over each fold individually but with shift and scales depending only on the training input or shift_in and scale_in (prevents data leakage).
        If task is "PathContinue", then this normalisation type is used for both training and validation sets.
        If task is "Forecast", then this normalisation type is used for only the input sets. 
        If overall normalisation is preferred, choose None for this norm_type and normalise data before input.
        Options: {"NormStd", "MinMax", "ScaleL2", "ScaleL2Shift", "ShiftScale", None}. (default: None).
    norm_type_target : str, optional
        Normalisation method called based on the options available in normalise_arrays. 
        Normalisation is carried out over each fold individually but with shift and scales depending only on the input or shift_target and scale_target (prevents data leakage).
        If task is "PathContinue", then this is ignored.
        If task is "Forecast", then this normalisation type is used for the target sets. 
        If overall normalisation is preferred, choose None for this norm_type and normalise data before input.
        Options: {"NormStd", "MinMax", "ScaleL2", "ScaleL2Shift", "ShiftScale", None}. (default: None).
    error_type : str, optional
        The type of error function to use in computing error in each fold. Options: {"meansquare", "norm_meansquare", "wasserstein1", "specdens"}. (default: "meansquare")
    minmax_range_in : tuple, optional
        Tuple containing the desired min and max when normalising using norm_type_in="MinMax". (default: (0, 1))
        If task is "PathContinue" then is applied to both input and target folds. If task is "Forecast" the applies only to the input sets. 
    minmax_range_target : tuple, optional
        Tuple containing the desired min and max when normalising using norm_type_target="MinMax". (default: (0, 1))
        If task is "PathContinue" then is ignored. If task is "Forecast" the applies only to the target sets. 
    shift_in : float, optional
        Shift for when calling "ShiftScale" norm type. Is the shift for each fold if task is "PathContinue" and only the input folds if task is "Forecast". Default: 0.
    scale_in : float, optional
        Scale for when calling "ShiftScale" norm type. Is the scale for each fold if task is "PathContinue" and only the input folds if task is "Forecast". Default: 1. 
    shift_target : float, optional
        Shift for when calling "ShiftScale" norm type. Is ignored if task is "PathContinue" and only the target folds if task is "Forecast". Default: 0.
    scale_target : float, optional
        Scale for when calling "ShiftScale" norm type. Is ignored if task is "PathContinue" and only the target folds if task is "Forecast". Default: 1. 
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


    def __init__(self, validation_parameters=None, validation_type="rolling", manage_remainder=True,
                       task="PathContinue", norm_type_in=None, norm_type_target=None, 
                       error_type="meansquare", minmax_range_in=(0, 1), minmax_range_target=(0, 1),
                       shift_in=0, scale_in=1, shift_target=0, scale_target=1, log_interval=10):
        
        self.validation_parameters = validation_parameters
        self.validation_type = validation_type
        self.manage_remainder = manage_remainder
        self.task = task
        self.norm_type_in = norm_type_in
        self.norm_type_target = norm_type_target
        self.error_type = error_type
        self.minmax_range_in = minmax_range_in
        self.minmax_range_target = minmax_range_target
        self.shift_in = shift_in
        self.scale_in = scale_in 
        self.shift_target = shift_target
        self.scale_target = scale_target
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
        
        # Use the validation sizes to define the data that will remain
        remainder_size = (input_size - train_size - validation_size) % jump_size
        
        # Iterate through data in, training method on each fold then compute validation results
        cv_datasets = []
        for start_id in range(0, nstarts):
            
            # Define the starting index
            start = start_id * jump_size

            # Define the training and validation data for rolling validation type
            if self.validation_type == "rolling":
                
                # Ignore remaining data (train sizes exactly as defined)
                if self.manage_remainder is False:
                    # Rolling window cross validation moves the starting points so the train size stays constant
                    train_in = data_in[start : start+train_size]
                    train_target = target[start : start+train_size]
                    validation_in = data_in[start+train_size : start+train_size+validation_size]
                    validation_target = target[start+train_size : start+train_size+validation_size]
            
                # Add remaining data to each train size (train sizes may be larger than defined)
                if self.manage_remainder is True:
                    # Rolling window cross validation moves the starting points so the train size stays constant
                    new_train_size = train_size + remainder_size
                    train_in = data_in[start : start+new_train_size]
                    train_target = target[start : start+new_train_size]
                    validation_in = data_in[start+new_train_size : start+new_train_size+validation_size]
                    validation_target = target[start+new_train_size : start+new_train_size+validation_size] 
                    
            # Define the training and validation data for expanding validation type
            elif self.validation_type == "expanding":
                
                # Ignore remaining data (train sizes exactly as defined, except last fold)
                if self.manage_remainder is False:
                    # Expanding window cross validation allows the train size to grow with each start
                    train_in = data_in[0 : start+train_size]
                    train_target = target[0 : start+train_size]
                    validation_in = data_in[start+train_size : start+train_size+validation_size]
                    validation_target = target[start+train_size : start+train_size+validation_size]
                    
                # Add remaining data to each train size (train sizes may be larger than defined)
                if self.manage_remainder is True:
                    # Expanding window cross validation allows the train size to grow with each start
                    new_train_size = train_size + remainder_size
                    train_in = data_in[0 : start+new_train_size]
                    train_target = target[0 : start+new_train_size]
                    validation_in = data_in[start+new_train_size : start+new_train_size+validation_size]
                    validation_target = target[start+new_train_size : start+new_train_size+validation_size] 

            # Raise error if cross validation type input is incorrect
            else:
                raise NotImplementedError("Validation method of splitting dataset is not available")
            
            # If task is path continue, normalise based on the first array, so all arrays are shifted by the training inputs
            if self.task == "PathContinue":
                # Call normalisation function to normalise all arrays according to the inputs
                normed_arrays = normalise_arrays([train_in, train_target, validation_in, validation_target], norm_type=self.norm_type_in, 
                                                 minmax_range=self.minmax_range_in, shift=self.shift_in, scale=self.scale_in)
                # Append datasets in correct order to cv_datasets
                cv_datasets.append(normed_arrays)
            
            # If task is forecasting, normalise the input and target arrays separately and normalise based on training
            if self.task == "Forecast":
                # Call normalisation function to normalise a single train-validation iteration -- inputs
                normed_input = normalise_arrays([train_in, validation_in], norm_type=self.norm_type_in, 
                                                minmax_range=self.minmax_range_in, shift=self.shift_in, scale=self.scale_in)
                # Call normalisation function to normalise a single train-validation iteration -- targets
                normed_output = normalise_arrays([train_target, validation_target], norm_type=self.norm_type_target, 
                                                minmax_range=self.minmax_range_target, shift=self.shift_target, scale=self.scale_target)
                # Construct cross validation set dataset
                data_ls = [normed_input[0][0], normed_output[0][0], normed_input[0][1], normed_output[0][1]]
                # Append datasets in correct order to cv_datasets
                cv_datasets.append((data_ls, self.shift_target, self.scale_target))
                
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
            elif self.error_type == "norm_meansquare":
                fold_err = calculate_nmse(validation_target, output, shift, scale)
            elif self.error_type == "wasserstein1":
                fold_err = calculate_wasserstein1err(validation_target, output, shift, scale)
            elif self.error_type == "specdens":
                fold_err = calculate_specdens_periodogram_err(validation_target, output, shift, scale)
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
        Stores all errors with the parameters and cv related data in a estimator_timestamp.txt file.
        Does not store information about the data generation process. 
        Logs information to the file at log_interval intervals. 
        Note that log_intervals should be judiciously large enough as logging repeatedly can be expensive. 

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

        # Create and write to a text file with details of cv run - results will append to this text file later
        estimator_name = estimator.__name__                         # stores the estimator class name for use in file name
        time_stamp = time.strftime("%Y%m%d-%H%M%S")                 # stores the time stamp to differentiate files, for use in file name
        number_parameters = len(parameter_combinations)
        cvresults_filename = f"{estimator_name}_{time_stamp}.txt"   # create file name with name of class of estimator + time cv code is run
        with open(cvresults_filename, "a") as file:                 # store cross-validation details above all cv results
            file.write(f"Estimator: {estimator_name}\n")
            file.write(f"Total number of parameters: {number_parameters}\n")
            file.write(f"Time started: {time_stamp}\n")
            file.write(f"Validation parameters: {self.validation_parameters} (training, validation, jump)\n")
            file.write(f"Validation type: {self.validation_type}\n")
            file.write(f"Task: {self.task}\n")
            file.write(f"Normalisation type for inputs: {self.norm_type_in}\n")
            file.write(f"Normalisation type for targets: {self.norm_type_target}\n")
            file.write(f"Error type: {self.error_type}\n")
            if self.norm_type_in == "MinMax":
                file.write(f"MinMax range for inputs: {self.minmax_range_in} \n")
            if self.norm_type_target == "MinMax":
                file.write(f"MinMax range for targets: {self.minmax_range_target} \n")
            if self.norm_type_in == "ShiftScale":
                file.write(f"Shift for inputs: {self.shift_in} \n")
                file.write(f"Scale for inputs: {self.scale_in} \n")
            if self.norm_type_target == "ShiftScale":
                file.write(f"Shift for targets: {self.shift_target} \n")
                file.write(f"Scale for targets: {self.scale_target} \n")
            file.write("-" * 80 + "\n") 
        
        # Create a process pool with num_processes workers
        pool = multiprocessing.Pool(processes=num_processes)

        # Issue tasks, yielding results as soon as they are available using imap_unordered
        partial_results = []        # Stores partial results in txt file in case of crash
        count = 0
        min_error = float('inf')
        min_parameter = None
        start_time = time.time()
        for result in pool.imap_unordered(self.test_parameter_set, combinations, chunksize=chunksize):
            
            # Append the results to each list
            partial_results.append(result)
            
            # Count the number of results and track best parameters
            count = count + 1
            if result[1] <= min_error:
                min_error = result[1]
                min_parameter = result[0]
            
            # At the first log_interval, compute and print estimated time for process
            if count == self.log_interval:
                first_log_time = time.time()
                first_log_time_elapsed = first_log_time - start_time
                estimated_total_time = (number_parameters // self.log_interval) * first_log_time_elapsed
                with open(cvresults_filename, "a") as file:
                    file.write(f"First log reached. Amount of time elapsed is {first_log_time_elapsed} seconds.\n")
                    file.write(f"Estimated time to finish is {estimated_total_time} seconds.\n")
                    file.write("-" * 80 + "\n")
                    
            # At every log_interval, log the partial results and empty it
            if len(partial_results) % self.log_interval == 0:
                # Write the partial results to the cv.txt file
                with open(cvresults_filename, "a") as file:
                    for partial_result in partial_results:
                        file.write(f"{partial_result}\n")
                    partial_results = []
                # Print current progress with best found
                print(f"Reached {count} hyperparameters") 
                print(f"Best estimate so far: {min_error} with {min_parameter}")   
        
        end_time = time.time()    
        # Log the remaining results not covered in log_interval
        with open(cvresults_filename, "a") as file:
            for partial_result in partial_results:
                file.write(f"{partial_result}\n")
            file.write("-" * 80 + "\n")    # Separator marks end of CV run
            file.write(f"Cross validation total time taken: {end_time - start_time} seconds.")
                        
        # Prevent further execution of tasks 
        pool.close()
        # Wait for worker processes to exit
        pool.join()
        
        return min_error, min_parameter
        
