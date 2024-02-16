# %% 

import numpy as np

from estimators.volt_funcs import Volterra
from estimators.ngrc_funcs import NGRC
from estimators.sindy_funcs import SINDyPolynomialSTLSQ

from datagen.data_generate import rk45
from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse

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
training_input_orig = data[0:ntrain-1] 
training_teacher_orig = data[1:ntrain]
testing_input_orig = data[ntrain-1:ntrain+ntest-1]
testing_teacher_orig = data[ntrain:ntrain+ntest]


### VOLTERRA

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_in_volt, train_teach_volt, test_in_volt, test_teach_volt = normalisation_output[0]
shift_volt, scale_volt = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg = 0.8, 0.2, 1e-9 

# Run new Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
out_volt = volt.Train(train_in_volt, train_teach_volt).PathContinue(train_teach_volt[-1], test_teach_volt.shape[0])

# Compute the mse
mse_volt = calculate_mse(test_teach_volt, out_volt, shift_volt, scale_volt)

# Plot the forecast and actual
plot_data([test_teach_volt, out_volt])
plot_data_distributions([test_teach_volt, out_volt])


### NGRC

# Normalise the arrays for NGRC
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_in_ngrc, train_teach_ngrc, test_in_ngrc, test_teach_ngrc = normalisation_output[0]
shift_ngrc, scale_ngrc = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for NGRC
ndim, ndelay, reg, deg = 3, 2, 1e-4, 2

# Run the new NGRC class
NGRC = NGRC(ndelay, deg, reg, washout)
NGRC.Train(train_in_ngrc, train_teach_ngrc)
out_ngrc = NGRC.PathContinue(train_teach_ngrc[-1], test_teach_ngrc.shape[0])

# Compute the mse
mse_ngrc = calculate_mse(test_teach_ngrc, out_ngrc, shift_ngrc, scale_ngrc)

# Plot the forecast and actual
plot_data([test_teach_ngrc, out_ngrc])
plot_data_distributions([test_teach_ngrc, out_ngrc])


### SINDy

# Normalise the arrays for SINDy
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_in_sindy, train_teach_sindy, test_in_sindy, test_teach_sindy = normalisation_output[0]
shift_sindy, scale_sindy = normalisation_output[1], normalisation_output[2]

# Define the hyperparameters for SINDy
threshold, alpha, deg = 0.1, 1e-15, 2

# Run the new SINDy functions
SINDy = SINDyPolynomialSTLSQ(alpha, threshold, deg, h)
out_sindy = SINDy.Train(train_in_sindy, train_teach_sindy).PathContinue(train_teach_sindy[-1], test_teach_sindy.shape[0])

# Compute the mse
mse_sindy = calculate_mse(test_teach_sindy, out_sindy, shift_sindy, scale_sindy)

# Plot the forecast and actual
plot_data([test_teach_sindy, out_sindy])
plot_data_distributions([test_teach_sindy, out_sindy])

print("MSEs")
print("Volterra: ", mse_volt)
print("NGRC: ", mse_ngrc)
print("SINDy: ", mse_sindy)
