# %% 
# Imports

import time

from estimators.volt_funcs import Volterra
from estimators.ngrc_funcs import NGRC
from estimators.polykernel_funcs import PolynomialKernel

from datagen.data_generate_ode import rk45 
from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse, calculate_nmse, calculate_mae, calculate_mdae_err, calculate_r2_err, calculate_mape_err
from utils.errors import calculate_wasserstein1_nd_err, calculate_specdens_periodogram_err, calculate_specdens_welch_err
from utils.errors import valid_pred_time

from systems.odes import lorenz

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

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

# Define Lyapunov exponent and x_ranges for plotting   
lyapunov_exponent = 0.9
x_values = [index * lyapunov_exponent * h for index in range(len(testing_teacher_orig))]

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
mdae_volt = calculate_mdae_err(test_teacher, output_volt, shift_volt, scale_volt)
mae_volt = calculate_mae(test_teacher, output_volt, shift_volt, scale_volt)
mape_volt = calculate_mape_err(test_teacher, output_volt, shift_volt, scale_volt)
r2_volt = calculate_r2_err(test_teacher, output_volt, shift_volt, scale_volt)
specwelch_volt = calculate_specdens_welch_err(test_teacher, output_volt, shift_volt, scale_volt)
specpgram_volt = calculate_specdens_periodogram_err(test_teacher, output_volt, shift_volt, scale_volt)
#wass1_nd_volt = calculate_wasserstein1_nd_err(test_teacher, output_volt, shift_volt, scale_volt)

valid_pred_time_volt = valid_pred_time(test_teacher, output_volt, shift_volt, scale_volt) * h * lyapunov_exponent

# Plot the forecast and actual
plot_data([test_teacher, output_volt], shift=shift_volt, scale=scale_volt, filename="images/lorenz_volterra.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values)
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
mdae_ngrc = calculate_mdae_err(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
mae_ngrc = calculate_mae(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
mape_ngrc = calculate_mape_err(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
r2_ngrc = calculate_r2_err(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
specwelch_ngrc = calculate_specdens_welch_err(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
specpgram_ngrc = calculate_specdens_periodogram_err(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)
#wass1_nd_ngrc = calculate_wasserstein1_nd_err(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)

valid_pred_time_ngrc = valid_pred_time(test_teacher, output_ngrc, shift_ngrc, scale_ngrc) * h * lyapunov_exponent

# Plot the forecast and actual
plot_data([test_teacher, output_ngrc], filename="images/lorenz_ngrc.pdf", shift=shift_ngrc, scale=scale_ngrc, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values)
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
mdae_poly = calculate_mdae_err(test_teacher, output_poly, shift_poly, scale_poly)
mae_poly = calculate_mae(test_teacher, output_poly, shift_poly, scale_poly)
mape_poly = calculate_mape_err(test_teacher, output_poly, shift_poly, scale_poly)
r2_poly = calculate_r2_err(test_teacher, output_poly, shift_poly, scale_poly)
specwelch_poly = calculate_specdens_welch_err(test_teacher, output_poly, shift_poly, scale_poly)
specpgram_poly = calculate_specdens_periodogram_err(test_teacher, output_poly, shift_poly, scale_poly)
#wass1_nd_poly = calculate_wasserstein1_nd_err(test_teacher, output_poly, shift_poly, scale_poly)

valid_pred_time_poly = valid_pred_time(test_teacher, output_poly, shift_poly, scale_poly) * h * lyapunov_exponent

# Plot the forecast and actual
plot_data([test_teacher, output_poly], filename="images/lorenz_polykernel.pdf", shift=shift_poly, scale=scale_poly, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values)
plot_data_distributions([test_teacher, output_poly], "images/lorenz_polykernel_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %% 
# Print MSEs

errors = PrettyTable(['Method', 'MSE', 'Normalised MSE', 'MAE', 'Wasserstein1_nd', 'MdAE', 'MAPE', 'R2-score', 'SpecDens (Welch)', 'SpecDens (PGram)'])
errors.add_row(["Volterra",   mse_volt, nmse_volt, mae_volt, wass1_nd_volt, mdae_volt, mape_volt, r2_volt, specwelch_volt, specpgram_volt])
errors.add_row(["Polynomial", mse_poly, nmse_poly, mae_poly, wass1_nd_poly, mdae_poly, mape_poly, r2_poly, specwelch_poly, specpgram_poly])
errors.add_row(["NGRC",       mse_ngrc, nmse_ngrc, mae_ngrc, wass1_nd_ngrc, mdae_ngrc, mape_ngrc, r2_ngrc, specwelch_ngrc, specpgram_ngrc])
print(errors)

# %% 
# Print MSEs

errors = PrettyTable(['Method', 'MSE', 'Normalised MSE', 'MAE', 'MdAE', 'MAPE', 'R2-score', 'SpecDens (Welch)', 'SpecDens (PGram)'])
errors.add_row(["Volterra",   mse_volt, nmse_volt, mae_volt, mdae_volt, mape_volt, r2_volt, specwelch_volt, specpgram_volt])
errors.add_row(["Polynomial", mse_poly, nmse_poly, mae_poly, mdae_poly, mape_poly, r2_poly, specwelch_poly, specpgram_poly])
errors.add_row(["NGRC",       mse_ngrc, nmse_ngrc, mae_ngrc, mdae_ngrc, mape_ngrc, r2_ngrc, specwelch_ngrc, specpgram_ngrc])
print(errors)

# %%

# Plot the cumulated absolute difference in mse 
import numpy as np
import matplotlib.pyplot as plt

def unshiftscale(data, shift, scale):
    return (1/scale) * data - shift

def sum_abs_diff_overtime(output, original, shift, scale):
    differences = []
    sum = 0
    unnormalised_output = unshiftscale(output, shift, scale)
    for data_id in range(len(output)):
        sum = sum + np.sum(np.abs(original[data_id, :] - unnormalised_output[data_id, :]))
        differences.append(sum)
    return differences

volt_diff = np.array(sum_abs_diff_overtime(output_volt, testing_teacher_orig, shift_volt, scale_volt))
ngrc_diff = np.array(sum_abs_diff_overtime(output_ngrc, testing_teacher_orig, shift_ngrc, scale_ngrc))
poly_diff = np.array(sum_abs_diff_overtime(output_poly, testing_teacher_orig, shift_poly, scale_poly))

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
# Gather and plot the error values over time for visualisation

def error_vals_over_time(original, output, shift, scale):
    maes = []
    mses = []
    norm_mses = []
    mdaes = []
    mapes = []
    r2_scores = []
    for data_id in range(3, len(output)+1):
        maes.append(calculate_mae(original[0:data_id, :], output[0:data_id, :], shift, scale))
        mses.append(calculate_mse(original[0:data_id, :], output[0:data_id, :], shift, scale))
        norm_mses.append(calculate_nmse(original[0:data_id, :], output[0:data_id, :], shift, scale))
        mdaes.append(calculate_mdae_err(original[0:data_id, :], output[0:data_id, :], shift, scale))
        mapes.append(calculate_mape_err(original[0:data_id, :], output[0:data_id, :], shift, scale))
        r2_scores.append(calculate_r2_err(original[0:data_id, :], output[0:data_id, :], shift, scale))
    return maes, mses, norm_mses, mdaes, mapes, r2_scores

error_name = ["maes", "mses", "norm_mses", "mdaes", "mapes", "r2_scores"]
volt_errors = error_vals_over_time(test_teacher, output_volt, shift_volt, scale_volt)
poly_errors = error_vals_over_time(test_teacher, output_poly, shift_poly, scale_poly)
ngrc_errors = error_vals_over_time(test_teacher, output_ngrc, shift_ngrc, scale_ngrc)

for i in range(len(volt_errors)):
    plt.plot(volt_errors[i], label="Volterra", color="r", linewidth=0.8)
    plt.plot(poly_errors[i], label="Polynomial kernel", color="b", linewidth=0.8)
    plt.plot(ngrc_errors[i], label="NG-RC", color="g", linewidth=0.8)
    plt.xlabel("time")
    plt.ylabel("error")
    plt.title(f"{error_name[i]}")
    plt.legend()
    plt.savefig(f"images/bekk_{error_name[i]}.pdf")
    plt.show()
    plt.close()
