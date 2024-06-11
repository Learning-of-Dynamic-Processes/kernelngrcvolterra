# Sets the default math computation in numpy to not parallelise (might be MKL)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'    

import numpy as np
import mat73
from time import time
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays
from estimators.polykernel_funcs import PolynomialKernel

if __name__ == "__main__":
            
    # Start wall timer
    start = time()
    
    # Load BEKK dataset
    matstruct_contents = mat73.loadmat("./datagen/BEKK_d15_data.mat")

    # Extract variables of interest from data
    returns = matstruct_contents['data_sim']
    epsilons = matstruct_contents['exact_epsilons']
    Ht_sim_vech = matstruct_contents['Ht_sim_vech']

    # Assign input and output data
    ndata = 3760
    data_in = epsilons[0:ndata-1, :]
    data_out = Ht_sim_vech[1:ndata, :]

    # Define the length of training and testing sizes
    ntrain = 3007
    washout = 101

    # Construct the training input and teacher, testing input and teacher
    training_input_orig = data_in[0:ntrain] 
    training_teacher_orig = data_out[0:ntrain]

    # Normalise training arrays if necessary - input
    normed_input = normalise_arrays([training_input_orig], norm_type=None)
    training_input = normed_input[0][0]
    
    # Normalise training arrays if necessary - teacher
    normed_output = normalise_arrays([training_teacher_orig], norm_type="NormStd")
    training_teacher = normed_output[0][0]
    
    # Define the range of parameters for which you want to cross validate over
    deg_range = np.arange(2, 10, 1)
    ndelays_range = np.arange(2, 102, 1)
    reg_range = np.logspace(-15, -1, 15)
    param_ranges = [deg_range, ndelays_range, reg_range]

    # Define additional input parameters
    param_add = [washout]

    # Instantiate CV, split dataset, crossvalidate in parallel
    CV = CrossValidate(validation_parameters=[501, 501, 501], validation_type="expanding",
                       manage_remainder=True,  task="Forecast", 
                       norm_type_in="MinMax", norm_type_target="MinMax",
                       error_type="meansquare", log_interval=100)
    cv_datasets = CV.split_data_to_folds(training_input, training_teacher)
    min_error, best_parameters = CV.crossvalidate(PolynomialKernel, cv_datasets, param_ranges, param_add, 
                                                  num_processes=8, chunksize=1)      
    
    # Print out the best paraeter and errors found
    print(f"Best parameters found are {best_parameters} with error {min_error}")
    
    # Print amount of time taken to run cv
    print(f"Amount of time to run: {time() - start}")