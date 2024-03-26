
import numpy as np
from datagen.data_generate import rk45
from systems.odes import lorenz
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays
from estimators.polykernel_funcs import PolynomialKernel

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
    washout = 1000
    ntest = ndata - ntrain

    # Construct training input and teacher, testing input and teacher
    training_input_orig = data[0:ntrain-1]
    training_teacher_orig = data[1:ntrain]

    # Normalise training arrays if necessary
    normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
    training_input, training_teacher = normalisation_output[0]

    # Define the range of parameters for which you want to cross validate over
    deg_range = [2]
    ndelays_range = np.arange(2, 11, 1)
    reg_range = np.logspace(-15, -1, 15)
    param_ranges = [deg_range, ndelays_range, reg_range]

    # Define the additional inputs taken in by the 
    param_add = [washout]

    # Instantiate CV, split dataset, crossvalidate in parallel
    CV = CrossValidate(validation_parameters=[2500, 500, 500], validation_type="rolling", 
                       task="PathContinue", norm_type="MinMax", 
                       error_type="wasserstein1", log_interval=1)
    cv_datasets = CV.split_data_to_folds(training_input, training_teacher)
    min_error, best_parameters = CV.crossvalidate(PolynomialKernel, cv_datasets, param_ranges, param_add, 
                                                  num_processes=8, chunksize=1)      
    
    # Print out the best paraeter and errors found
    print(f"Best parameters found are {best_parameters} with error {min_error}")