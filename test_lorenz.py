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
shift_volt, scale_volt = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout =  0.3, 0.3, 1e-10, 100 

# Start timer
start = time.time()

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output_volt = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"Volterra took: {time.time() - start}")

# Compute the errors
mse_volt = calculate_mse(test_teacher, output_volt, shift_volt, scale_volt)
nmse_volt = calculate_nmse(test_teacher, output_volt, shift_volt, scale_volt)
wass1_volt = calculate_wasserstein1err(test_teacher, output_volt, shift_volt, scale_volt)
spec_volt = calculate_specdensloss(test_teacher, output_volt, shift_volt, scale_volt)

# Plot the forecast and actual
plot_data([test_teacher, output_volt], shift=shift_volt, scale=scale_volt, filename="images/lorenz_volterra.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_volt], "images/lorenz_volterra_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# NGRC defaults with pinv

# Normalise the arrays for NGRC
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift_ngrc, scale_ngrc = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 3, 2, 1e-07, 0

# Start timer
start = time.time()

# Run the new NGRC class
ngrc = NGRC(ndelay, deg, reg, washout, isPathContinue=True)
output_ngrc = ngrc.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"NGRC took: {time.time() - start}")

# Compute the errors
mse_ngrc = calculate_mse(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
nmse_ngrc = calculate_nmse(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
wass1_ngrc = calculate_wasserstein1err(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
spec_ngrc = calculate_specdensloss(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)

# Plot the forecast and actual
plot_data([test_teacher, output_ngrc], filename="images/lorenz_ngrc.pdf", shift=shift_ngrc, scale=scale_ngrc, xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_ngrc], "images/lorenz_ngrc_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# Polynomial kernel 

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift_poly, scale_poly = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for Poly Kernel
deg, ndelays, reg, washout = 2, 6, 1e-06, 0 

# Start timer
start = time.time()

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout)
output_poly = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Print time taken for training and generating outputs
print(f"Polynomial kernel took: {time.time() - start}")

# Compute the errors
mse_poly = calculate_mse(test_teacher, output_poly, shift_poly, scale_poly)
nmse_poly = calculate_nmse(test_teacher, output_poly, shift_poly, scale_poly)
wass1_poly = calculate_wasserstein1err(test_teacher, output_poly, shift_poly, scale_poly)
spec_poly = calculate_specdensloss(test_teacher, output_poly, shift_poly, scale_poly)

# Plot the forecast and actual
plot_data([test_teacher, output_poly], filename="images/lorenz_polykernel.pdf", shift=shift_poly, scale=scale_poly, xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher, output_poly], "images/lorenz_polykernel_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# Print MSEs

print("Method: MSE, Normalised MSE, Wasserstein1, Spectral Density Distance")
print(f"Volterra:                    {mse_volt}, {nmse_volt}, {wass1_volt}, {spec_volt}")
print(f"Polynomial Kernel:           {mse_poly}, {nmse_poly}, {wass1_poly}, {spec_poly}")
print(f"NGRC:                        {mse_ngrc}, {nmse_ngrc}, {wass1_ngrc}, {spec_ngrc}")

# %%

# Plot the absolute difference in mse 
import numpy as np
import matplotlib.pyplot as plt

def unshiftscale(data, shift, scale):
    return (1/scale) * data - shift

def abs_diff_overtime(output, original, shift, scale):
    differences = []
    sum = 0
    unnormalised_output = unshiftscale(output, shift, scale)
    for data_id in range(len(output)):
        sum = sum + np.sum(np.abs(original[data_id, :] - unnormalised_output[data_id, :]))
        differences.append(sum)
    return differences

volt_diff = np.array(abs_diff_overtime(output_volt, testing_teacher_orig, shift_volt, scale_volt))
ngrc_diff = np.array(abs_diff_overtime(output_ngrc, testing_teacher_orig, shift_ngrc, scale_ngrc))
poly_diff = np.array(abs_diff_overtime(output_poly, testing_teacher_orig, shift_poly, scale_poly))

plt.figure(figsize=(12, 8))
plt.plot(volt_diff, label="Volterra", color="r", linewidth=0.8)
plt.plot(ngrc_diff, label="NG-RC", color="g", linewidth=0.8)
plt.plot(poly_diff, label="Polynomial kernel", color="b", linewidth=0.8)
plt.xlabel("time")
plt.ylabel("sum of absolute difference")
plt.legend()
plt.savefig("images/errors_lorenz.pdf")
plt.show()
plt.close()

# %%
