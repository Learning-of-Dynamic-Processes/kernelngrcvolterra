
import numpy as np
from datagen.data_generate import rk45
from systems.odes import lorenz
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays
from estimators.ngrc_funcs import NGRC

if __name__ == "__main__":
    
    # Create the Lorenz dataset
    lor_args = (10, 8/3, 28)
    Z0 = (0, 1, 1.05)
    h = 0.005
    t_span = (0, 40)
    t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)

    # Define full data training and testing sizes
    ndata  = len(data)
    ntrain = 5000 
    washout = 2
    ntest = ndata - ntrain

    # Construct training input and teacher, testing input and teacher
    training_input_orig = data[0:ntrain-1, 0].reshape((-1, 1)) 
    training_teacher_orig = data[1:ntrain, 0].reshape((-1, 1))

    # Normalise training arrays if necessary
    normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
    training_input, training_teacher = normalisation_output[0]

    # Define the range of parameters for which you want to cross validate over
    ndelay_range = [2] 
    deg_range = [2]
    reg_range = np.logspace(-15, 2, 18)
    param_ranges = [ndelay_range, deg_range, reg_range]

    # Define the additional inputs taken in by the 
    param_add = [washout]

    # Instantiate CV, split dataset, crossvalidate in parallel
    CV = CrossValidate(validation_parameters=[2500, 500, 500], validation_type="rolling", 
                       task="PathContinue", norm_type=None, 
                       error_type="wasserstein1", log_interval=1)
    cv_datasets = CV.split_data_to_folds(training_input, training_teacher)
    min_error, best_parameters = CV.crossvalidate(NGRC, cv_datasets, param_ranges, param_add, 
                                                  num_processes=8, chunksize=1)      
    
    # Print out the best paraeter and errors found
    print(f"Best parameters found are {best_parameters} with error {min_error}")   