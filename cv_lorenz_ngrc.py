# Sets the default math computation in numpy to not parallelise (might be MKL)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'    

import numpy as np
from time import time
from datagen.data_generate_ode import rk45
from systems.odes import lorenz
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays
from estimators.ngrc_funcs import NGRC

if __name__ == "__main__":
    
    # Start wall timer
    start = time()
    
    # Create the Lorenz dataset
    lor_args = (10, 8/3, 28)
    Z0 = (0, 1, 1.05)
    h = 0.005
    t_span = (0, 40)
    t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)

    # Define full data training and testing sizes
    ntrain = 5000 
    washout = 0

    # Construct training input and teacher, testing input and teacher
    training_input_orig = data[0:ntrain-1]
    training_teacher_orig = data[1:ntrain]

    # Normalise training arrays if necessary
    normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
    training_input, training_teacher = normalisation_output[0]

    # Define the range of parameters for which you want to cross validate over
    ndelay_range = [1, 2, 3, 4, 5] 
    deg_range = [1, 2, 3]
    reg_range = np.logspace(-15, -1, 15)
    param_ranges = [ndelay_range, deg_range, reg_range]

    # Define additional input parameters
    param_add = [washout, True]

    # Instantiate CV, split dataset, crossvalidate in parallel
    CV = CrossValidate(validation_parameters=[2500, 500, 500], validation_type="rolling", manage_remainder=True,
                       task="PathContinue", norm_type_in=None, error_type="meansquare", log_interval=10)
    cv_datasets = CV.split_data_to_folds(training_input, training_teacher)
    min_error, best_parameters = CV.crossvalidate(NGRC, cv_datasets, param_ranges, param_add, 
                                                  num_processes=50, chunksize=1)      
    
    # Print out the best parameter and errors found
    print(f"Best parameters found are {best_parameters} with error {min_error}")
    
    # Print amount of time taken to run cv
    print(f"Amount of time to run: {time() - start}")