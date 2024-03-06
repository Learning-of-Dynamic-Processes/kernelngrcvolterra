import numpy as np
from utils.crossvalidation import CrossValidate
from estimators.volt_funcs import Volterra

# Define a sample training data
data_len = 101
training_data = np.linspace(0, data_len-1, data_len).reshape((-1, 1))
training_input = training_data[0:data_len-1]
training_target = training_data[1:data_len]

# Define the range of parameters for which you want to cross validate over
ld_coef_range = np.linspace(0.1, 0.2, 5)
tau_coef_range = np.linspace(0.1, 0.2, 5)
reg_range = np.logspace(-15, -10, 5)
param_ranges = [ld_coef_range, tau_coef_range, reg_range]

# Define the names of the parameters -- orders must match
param_names = ["ld_coef", "tau_coef", "reg"]

# Define the additional inputs taken in by the 
param_add = [10]

if __name__ == "__main__":
    CV = CrossValidate(validation_parameters=[50, 20, 15], validation_type="rolling", task="PathContinue", norm_type=None, ifPrint=True)
    best_parameters, parameter_combinations, errors = CV.crossvalidate_multiprocessing(Volterra, training_input, training_target, param_ranges, param_names, param_add, num_processes=1)

