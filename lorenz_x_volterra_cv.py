import numpy as np
from time import process_time
from datagen.data_generate import rk45
from estimators.volt_funcs import Volterra
from utils.crossvalidation import CrossValidate
from utils.normalisation import normalise_arrays

# Create the Lorenz dataset
def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)

h = 0.005
t_span = (0, 40)
slicing = int(h/h)

t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)
t_eval = t_eval[::slicing]
data = data[::slicing]

# Define full data training and testing sizes
ndata  = len(data)
ntrain = 5000 
washout = 1000
ntest = ndata - ntrain

# Construct training input and teacher, testing input and teacher
training_input_orig = data[0:ntrain-1, 0].reshape((-1, 1)) 
training_teacher_orig = data[1:ntrain, 0].reshape((-1, 1))

# Normalise training arrays if necessary
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig], norm_type=None)
training_input, training_teacher = normalisation_output[0]

# Define the range of parameters for which you want to cross validate over
ld_coef_range = [0.1] #np.linspace(0.01, 0.99, 99) 
tau_coef_range = [0.1] #np.linspace(0.01, 0.99, 99) 
reg_range = [1e-15] #np.logspace(-15, -1, 15)
param_ranges = [ld_coef_range, tau_coef_range, reg_range]

# Define the names of the parameters -- orders must match
param_names = ["ld coef", "tau coef", "reg"]
# Define the additional inputs taken in by the 
param_add = [washout]

if __name__ == "__main__":
    CV = CrossValidate(validation_parameters=[2500, 100, 600], validation_type="rolling", task="PathContinue", norm_type="ScaleL2Shift", ifPrint=True)
    best_parameters, parameter_combinations, errors = CV.crossvalidate_multiprocessing(Volterra, training_input, training_teacher, param_ranges, param_names, param_add, num_processes=50)      