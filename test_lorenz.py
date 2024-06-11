# %% 
# Imports

import time

from estimators.volt_funcs import Volterra
from estimators.ngrc_funcs import NGRC
from estimators.polykernel_funcs import PolynomialKernel

from datagen.data_generate_ode import rk45 
from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse, calculate_nmse, calculate_wasserstein1err, calculate_specdensloss
from systems.odes import lorenz

# %% 
# Prepare Lorenz datatsets

# Create the Lorenz dataset
lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)
h = 0.005
t_span = (0, 75)
t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)

# Define full data training and testing sizes
ndata  = len(data)
ntrain = 5000 
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
ld_coef, tau_coef, reg, washout = 0.6, 0.4, 1e-09, 1000 

# Start timer
start = time.time()

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output_volt = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"Volterra took: {time.time() - start}")

# Compute the errors
mse_volt = calculate_mse(test_teacher, output_volt, shift, scale)
nmse_volt = calculate_nmse(test_teacher, output_volt, shift, scale)
wass1_volt = calculate_wasserstein1err(test_teacher, output_volt, shift, scale)
spec_volt = calculate_specdensloss(test_teacher, output_volt, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_volt], shift=shift, scale=scale, filename="images/lorenz_volterra.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_volt], "images/lorenz_volterra_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# Volterra with L2 least squares regression using pinv 

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.6, 0.4, 1e-09, 1000 

# Start timer
start = time.time()

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout, pinv=True)
output_volt_pinv = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"Volterra with pinv took: {time.time() - start}")

# Compute the errors
mse_volt_pinv = calculate_mse(test_teacher, output_volt_pinv, shift, scale)
nmse_volt_pinv = calculate_nmse(test_teacher, output_volt_pinv, shift, scale)
wass1_volt_pinv = calculate_wasserstein1err(test_teacher, output_volt_pinv, shift, scale)
spec_volt_pinv = calculate_specdensloss(test_teacher, output_volt_pinv, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_volt_pinv], filename="images/lorenz_volterrapinv.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_volt_pinv], "images/lorenz_volterrapinv_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# NGRC defaults with pinv

# Normalise the arrays for NGRC
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 2, 2, 0.00012224984640507843, 0 

# Start timer
start = time.time()

# Run the new NGRC class
ngrc = NGRC(ndelay, deg, reg, washout, isPathContinue=True)
output_ngrc = ngrc.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"NGRC took: {time.time() - start}")

# Compute the errors
mse_ngrc = calculate_mse(test_teacher, output_ngrc, shift, scale)
nmse_ngrc = calculate_nmse(test_teacher, output_ngrc, shift, scale)
wass1_ngrc = calculate_wasserstein1err(test_teacher, output_ngrc, shift, scale)
spec_ngrc = calculate_specdensloss(test_teacher, output_ngrc, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_ngrc], filename="images/lorenz_ngrc.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_ngrc], "images/lorenz_ngrc_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# Polynomial kernel 

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for Poly Kernel
deg, ndelays, reg, washout = 2, 7, 1e-07, 0

# Start timer
start = time.time()

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout)
output_poly = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"Polynomial kernel took: {time.time() - start}")

# Compute the errors
mse_poly = calculate_mse(test_teacher, output_poly, shift, scale)
nmse_poly = calculate_nmse(test_teacher, output_poly, shift, scale)
wass1_poly = calculate_wasserstein1err(test_teacher, output_poly, shift, scale)
spec_poly = calculate_specdensloss(test_teacher, output_poly, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_poly], filename="images/lorenz_polykernel.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_poly], "images/lorenz_polykernel_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# Polynomial kernel with pinv

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for Poly Kernel
deg, ndelays, reg, washout = 2, 6, 1e-07, 0

# Start timer
start = time.time()

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout, pinv=True)
output_poly_pinv = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"Polynomial with pinv kernel took: {time.time() - start}")

# Compute the errors
mse_poly_pinv = calculate_mse(test_teacher, output_poly_pinv, shift, scale)
nmse_poly_pinv = calculate_nmse(test_teacher, output_poly_pinv, shift, scale)
wass1_poly_pinv = calculate_wasserstein1err(test_teacher, output_poly_pinv, shift, scale)
spec_poly_pinv = calculate_specdensloss(test_teacher, output_poly_pinv, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_poly_pinv], filename="images/lorenz_polykernelpinv.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_poly_pinv], "images/lorenz_polykernelpinv_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# Print MSEs

print("Method: MSE, Normalised MSE, Wasserstein1, Spectral Density Distance")
print(f"Volterra:                    {mse_volt}, {nmse_volt}, {wass1_volt}, {spec_volt}")
print(f"Volterra with pinv:          {mse_volt_pinv}, {nmse_volt_pinv}, {wass1_volt_pinv}, {spec_volt_pinv}")
print(f"NGRC:                        {mse_ngrc}, {nmse_ngrc}, {wass1_ngrc}, {spec_ngrc}")
print(f"Polynomial Kernel:           {mse_poly}, {nmse_poly}, {wass1_poly}, {spec_poly}")
print(f"Polynomial Kernel with pinv: {mse_poly_pinv}, {nmse_poly_pinv}, {wass1_poly_pinv}, {spec_poly_pinv}")

# %%
