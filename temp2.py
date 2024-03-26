import numpy as np
from datagen.data_generate import rk45
from estimators.volt_funcs import Volterra
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays
from systems.odes import lorenz

if __name__ == "__main__":
    
    # Generate dataset
    lor_args = (10, 8/3, 28)
    Z0 = (0, 1, 1.05)
    h = 0.005
    t_span = (0, 40)
    t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)

    # Define full data training and testing sizes
    ndata  = len(data)
    ntrain = 500
    washout = 10
    ntest = ndata - ntrain

    # Construct training input and teacher, testing input and teacher
    training_input_orig = data[0:ntrain-1, 0].reshape((-1, 1)) 
    training_teacher_orig = data[1:ntrain, 0].reshape((-1, 1))

    # Normalise training arrays if necessary
    normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
    training_input, training_teacher = normalisation_output[0]

    # Define the range of parameters for which you want to cross validate over
    ld_coef_range = np.linspace(0.1, 0.9, 3).round(2)
    tau_coef_range = np.linspace(0.1, 0.9, 3).round(2)
    reg_range = np.logspace(-3, -1, 3)
    param_ranges = [ld_coef_range, tau_coef_range, reg_range]

    # Define the names of the parameters -- orders must match
    param_names = ["ld coef", "tau coef", "reg"]
    # Define the additional inputs taken in by the 
    param_add = [washout]
    
    CV = CrossValidate(validation_parameters=[200, 200, 100], validation_type="rolling", task="PathContinue", norm_type=None)
    cv_datasets = CV.split_data_to_folds(training_input, training_teacher)
    best = CV.crossvalidate(Volterra, cv_datasets, param_ranges, param_add, num_processes=5, chunksize=10)      