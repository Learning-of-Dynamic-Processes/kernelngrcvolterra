# Sets the default math computation in numpy to not parallelise (might be MKL)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'    

import numpy as np
from time import time
from datagen.data_generate_dde import dde_rk45
from systems.ddes import mackeyglass
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays
from estimators.polykernel_funcs import PolynomialKernel

if __name__ == "__main__":
            
    # Start wall timer
    start = time()
    
    # Create the MG dataset
    def init(t):
        return 1.2

    mg_args = {'delay': 17, 'a': 0.2, 'b': 0.1, 'n': 10 }

    h = 0.02
    n_intervals = 200
    slicing = int(1 / h)

    data = dde_rk45(n_intervals, init, mackeyglass, h, mg_args)[1][::slicing]

    # Define training and washout size
    ntrain = 3000
    washout = 0

    # Construct training input and teacher, testing input and teacher
    training_input_orig = data[0:ntrain-1] 
    training_teacher_orig = data[1:ntrain]

    # Normalise training arrays if necessary
    normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
    training_input, training_teacher = normalisation_output[0]

    # Define the range of parameters for which you want to cross validate over
    deg_range = np.arange(1, 10, 1)
    ndelays_range = [1] #np.arange(1, 102, 1)
    reg_range = np.logspace(-15, -1, 15)
    param_ranges = [deg_range, ndelays_range, reg_range]

    # Define additional input parameters
    param_add = [washout]

    # Instantiate CV, split dataset, crossvalidate in parallel
    CV = CrossValidate(validation_parameters=[2000, 500, 100], validation_type="rolling", manage_remainder=True, 
                       task="PathContinue", norm_type_in="MinMax", error_type="meansquare", log_interval=100)
    cv_datasets = CV.split_data_to_folds(training_input, training_teacher)
    min_error, best_parameters = CV.crossvalidate(PolynomialKernel, cv_datasets, param_ranges, param_add, 
                                                  num_processes=50, chunksize=1)      
    
    # Print out the best paraeter and errors found
    print(f"Best parameters found are {best_parameters} with error {min_error}")
    
    # Print amount of time taken to run cv
    print(f"Amount of time to run: {time() - start}")