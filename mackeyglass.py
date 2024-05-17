# %% 
# Imports

import numpy as np

from estimators.volt_funcs import Volterra
from estimators.ngrc_funcs import NGRC
from estimators.sindy_funcs import SINDyPolynomialSTLSQ
from estimators.polykernel_funcs import PolynomialKernel

from datagen.data_generate_dde import dde_rk45
from systems.ddes import mackeyglass
from utils.normalisation import normalise_arrays
from utils.errors import calculate_mse
from utils.plotting import plot_data, plot_data_distributions

#%% 
# Generate dataset

def init(t):
    return 1.2

mg_args = {'delay': 17, 'a': 0.2, 'b': 0.1, 'n': 10 }

h = 0.02
n_intervals = 200
slicing = int(1 / h)

data = dde_rk45(n_intervals, init, mackeyglass, h, mg_args)[1][::slicing]

ndata = len(data)
ntrain = 3001
washout = 1000
ntest = ndata - ntrain

# Construct training input and teacher, testing input and teacher
training_input_orig = data[0:ntrain-1] 
training_teacher_orig = data[1:ntrain]
testing_input_orig = data[ntrain-1:ntrain+ntest-1]
testing_teacher_orig = data[ntrain:ntrain+ntest]

# %% 
# Volterra with L2 least squares regression

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.9, 0.4, 1e-09, 1000

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_volt = calculate_mse(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output], "mackeyglass_volterra_plot.pdf")
plot_data_distributions([test_teacher, output], "mackeyglass_volterra_dist.pdf")

# %%
