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

from prettytable import PrettyTable

# %% 
# Prepare Lorenz datatsets

# Create the Lorenz dataset
lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)
h = 0.005
t_span = (0, 200)
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
train_input_volt, train_teacher_volt, test_input_volt, test_teacher_volt = normalisation_output[0]
shift_volt, scale_volt = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout =  0.3, 0.3, 1e-10, 100 

# Start timer
start = time.time()

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output_volt = volt.Train(train_input_volt, train_teacher_volt).PathContinue(train_teacher_volt[-1], test_teacher_volt.shape[0])

# Print time taken for training and generating outputs
print(f"Volterra took: {time.time() - start}")

# %% 
# Polynomial kernel 

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input_poly, train_teacher_poly, test_input_poly, test_teacher_poly = normalisation_output[0]
shift_poly, scale_poly = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for Poly Kernel
deg, ndelays, reg, washout = 2, 6, 1e-06, 0 

# Start timer
start = time.time()

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout)
output_poly = polykernel.Train(train_input_poly, train_teacher_poly).PathContinue(train_teacher_poly[-1], test_teacher_poly.shape[0])

# Print time taken for training and generating outputs
print(f"Polynomial kernel took: {time.time() - start}")

# %% 
# NGRC defaults with pinv

# Normalise the arrays for NGRC
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
train_input_ngrc, train_teacher_ngrc, test_input_ngrc, test_teacher_ngrc = normalisation_output[0]
shift_ngrc, scale_ngrc = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 3, 2, 1e-07, 0

# Start timer
start = time.time()

# Run the new NGRC class
ngrc = NGRC(ndelay, deg, reg, washout, isPathContinue=True)
output_ngrc = ngrc.Train(train_input_ngrc, train_teacher_ngrc).PathContinue(train_teacher_ngrc[-1], test_teacher_ngrc.shape[0])

# Print time taken for training and generating outputs
print(f"NGRC took: {time.time() - start}")

# %%
# Calculate valid prediction time

epsilon = 0.2
valid_pred_time_volt = valid_pred_time(test_teacher_volt, output_volt, shift_volt, scale_volt) * h * lyapunov_exponent
valid_pred_time_poly = valid_pred_time(test_teacher_poly, output_poly, shift_poly, scale_poly) * h * lyapunov_exponent
valid_pred_time_ngrc = valid_pred_time(test_teacher_ngrc, output_ngrc, shift_ngrc, scale_ngrc) * h * lyapunov_exponent

# Print valid prediction times
valid_pred_time_table = PrettyTable(["Volterra", "Polynomial", "NG-RC"])
valid_pred_time_table.add_row([valid_pred_time_volt, valid_pred_time_poly, valid_pred_time_ngrc])
print(valid_pred_time_table)

# %% 
# Calculate time step by time step errors

# Max valid prediction time is about above 10. Round to 11 for error range.
err_range = 2444  # about 11 lyapunov times

# Volterra
mse_volt = calculate_mse(test_teacher_volt[0:err_range], output_volt[0:err_range], shift_volt, scale_volt)
nmse_volt = calculate_nmse(test_teacher_volt[0:err_range], output_volt[0:err_range], shift_volt, scale_volt)
mdae_volt = calculate_mdae_err(test_teacher_volt[0:err_range], output_volt[0:err_range], shift_volt, scale_volt)
mae_volt = calculate_mae(test_teacher_volt[0:err_range], output_volt[0:err_range], shift_volt, scale_volt)
mape_volt = calculate_mape_err(test_teacher_volt[0:err_range], output_volt[0:err_range], shift_volt, scale_volt)
r2_volt = calculate_r2_err(test_teacher_volt[0:err_range], output_volt[0:err_range], shift_volt, scale_volt)

# Polynomial 
mse_poly = calculate_mse(test_teacher_poly[0:err_range], output_poly[0:err_range], shift_poly, scale_poly)
nmse_poly = calculate_nmse(test_teacher_poly[0:err_range], output_poly[0:err_range], shift_poly, scale_poly)
mdae_poly = calculate_mdae_err(test_teacher_poly[0:err_range], output_poly[0:err_range], shift_poly, scale_poly)
mae_poly = calculate_mae(test_teacher_poly[0:err_range], output_poly[0:err_range], shift_poly, scale_poly)
mape_poly = calculate_mape_err(test_teacher_poly[0:err_range], output_poly[0:err_range], shift_poly, scale_poly)
r2_poly = calculate_r2_err(test_teacher_poly[0:err_range], output_poly[0:err_range], shift_poly, scale_poly)

# NG-RC
mse_ngrc = calculate_mse(test_teacher_ngrc[0:err_range], output_ngrc[0:err_range], shift_ngrc, scale_ngrc)
nmse_ngrc = calculate_nmse(test_teacher_ngrc[0:err_range], output_ngrc[0:err_range], shift_ngrc, scale_ngrc)
mdae_ngrc = calculate_mdae_err(test_teacher_ngrc[0:err_range], output_ngrc[0:err_range], shift_ngrc, scale_ngrc)
mae_ngrc = calculate_mae(test_teacher_ngrc[0:err_range], output_ngrc[0:err_range], shift_ngrc, scale_ngrc)
mape_ngrc = calculate_mape_err(test_teacher_ngrc[0:err_range], output_ngrc[0:err_range], shift_ngrc, scale_ngrc)
r2_ngrc = calculate_r2_err(test_teacher_ngrc[0:err_range], output_ngrc[0:err_range], shift_ngrc, scale_ngrc)

# %% 
# Plot time signals up to err_range

print("Volterra")
plot_data([test_teacher_volt[0:err_range], output_volt[0:err_range]], filename="images/lorenz_volt.pdf", shift=shift_volt, scale=scale_volt, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values[0:err_range])
print("Polynomial")
plot_data([test_teacher_poly[0:err_range], output_poly[0:err_range]], filename="images/lorenz_poly.pdf", shift=shift_poly, scale=scale_poly, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values[0:err_range])
print("NG-RC")
plot_data([test_teacher_ngrc[0:err_range], output_ngrc[0:err_range]], filename="images/lorenz_ngrc.pdf", shift=shift_ngrc, scale=scale_ngrc, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values[0:err_range])

# %%
# Print time step to time step errors

errors = PrettyTable(['Method', 'MSE', 'Normalised MSE', 'MdAE', 'MAE', 'MAPE', 'R2-score'])
errors.add_row(["Volterra",   mse_volt, nmse_volt, mdae_volt, mae_volt, mape_volt, r2_volt])
errors.add_row(["Polynomial", mse_poly, nmse_poly, mdae_poly, mae_poly, mape_poly, r2_poly])
errors.add_row(["NGRC",       mse_ngrc, nmse_ngrc, mdae_ngrc, mae_ngrc, mape_ngrc, r2_ngrc])
print(errors)

# %% 
# Plot data signals after err_range

print("Volterra")
plot_data([test_teacher_volt[err_range: ], output_volt[err_range: ]], shift=shift_volt, scale=scale_volt, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values[err_range: ])
print("Polynomial")
plot_data([test_teacher_poly[err_range: ], output_poly[err_range: ]], shift=shift_poly, scale=scale_poly, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values[err_range: ])
print("NG-RC")
plot_data([test_teacher_ngrc[err_range: ], output_ngrc[err_range: ]], shift=shift_ngrc, scale=scale_ngrc, xlabel=["x", "y", "z"], datalabel=['actual', 'output'], x_values=x_values[err_range: ])

# %% 
# Plot overall data distribution 

print("Volterra")
plot_data_distributions([test_teacher_volt, output_volt], "images/lorenz_volt_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'], figsize=(8, 5))
print("Polynomial")
plot_data_distributions([test_teacher_poly, output_poly], "images/lorenz_poly_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'], figsize=(8, 5))
print("NG-RC")
plot_data_distributions([test_teacher_ngrc, output_ngrc], "images/lorenz_ngrc_dist.pdf", xlabel=["x", "y", "z"], datalabel=['actual', 'output'], figsize=(8, 5))

# %% 
# Compute climate metrics

# Volterra 
print("Volterra")
specwelch_volt = calculate_specdens_welch_err(test_teacher_volt, output_volt, shift_volt, scale_volt, fs=200, nperseg=1000, stop=30, ifPlot=True, figname="images/lorenz_volt_welch.pdf", leg_loc="upper right", leg_bbox_anchor=(1.12, 1))
specpgram_volt = calculate_specdens_periodogram_err(test_teacher_volt, output_volt, shift_volt, scale_volt, fs=200, stop=30, ifPlot=True, leg_loc="upper right", leg_bbox_anchor=(1.12, 1))

# Polynomial
print("Polynomial")
specwelch_poly = calculate_specdens_welch_err(test_teacher_poly, output_poly, shift_poly, scale_poly, fs=200, nperseg=1000, stop=30, ifPlot=True, figname="images/lorenz_poly_welch.pdf", leg_loc="upper right", leg_bbox_anchor=(1.12, 1))
specpgram_poly = calculate_specdens_periodogram_err(test_teacher_poly, output_poly, shift_poly, scale_poly, fs=200, stop=30, ifPlot=True, leg_loc="upper right", leg_bbox_anchor=(1.12, 1))

# NG-RC
print("NG-RC")
specwelch_ngrc = calculate_specdens_welch_err(test_teacher_ngrc, output_ngrc, shift_ngrc, scale_ngrc, fs=200, nperseg=1000, stop=30, ifPlot=True, figname="images/lorenz_ngrc_welch.pdf", leg_loc="upper right", leg_bbox_anchor=(1.12, 1))
specpgram_ngrc = calculate_specdens_periodogram_err(test_teacher_ngrc, output_ngrc, shift_ngrc, scale_ngrc, fs=200, stop=30, ifPlot=True, leg_loc="upper right", leg_bbox_anchor=(1.12, 1))

# %% 
# Wasserstein_nd takes too long with the full dataset, sampling is needed

slice = 50
print("Volterra")
plot_data_distributions([test_teacher_volt[::slice], output_volt[::slice]], xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
print("Polynomial")
plot_data_distributions([test_teacher_poly[::slice], output_poly[::slice]], xlabel=["x", "y", "z"], datalabel=['actual', 'output'])
print('NG-RC')
plot_data_distributions([test_teacher_ngrc[::slice], output_ngrc[::slice]], xlabel=["x", "y", "z"], datalabel=['actual', 'output'])

# %%
# Compute Wasserstein distance for sampled distributions

wass1_nd_volt = calculate_wasserstein1_nd_err(test_teacher_volt[::slice], output_volt[::slice], shift_volt, scale_volt)
wass1_nd_poly = calculate_wasserstein1_nd_err(test_teacher_poly[::slice], output_poly[::slice], shift_poly, scale_poly)
wass1_nd_ngrc = calculate_wasserstein1_nd_err(test_teacher_ngrc[::slice], output_ngrc[::slice], shift_ngrc, scale_ngrc)

# %% 
# Climate error metrics

climate_err_table = PrettyTable(['Method', 'Wasserstein1', 'SpecDens (Welch)', 'SpecDens (PGram)'])
climate_err_table.add_row(["Volterra",   wass1_nd_volt, specwelch_volt, specpgram_volt])
climate_err_table.add_row(["Polynomial", wass1_nd_poly, specwelch_poly, specpgram_poly])
climate_err_table.add_row(["NGRC",       wass1_nd_ngrc, specwelch_ngrc, specpgram_ngrc])
print(climate_err_table)

# %% 
# Print all errors together

errors = PrettyTable(['Method', 'ValidPredTime', 'MSE', 'Normalised MSE', 'MAE', 'MdAE', 'MAPE', 'R2-score', 'SpecDens (Welch)', 'SpecDens (PGram)', 'Wass1_nd'])
errors.add_row(["Volterra",   valid_pred_time_volt, mse_volt, nmse_volt, mae_volt, mdae_volt, mape_volt, r2_volt, specwelch_volt, specpgram_volt, wass1_nd_volt])
errors.add_row(["Polynomial", valid_pred_time_poly, mse_poly, nmse_poly, mae_poly, mdae_poly, mape_poly, r2_poly, specwelch_poly, specpgram_poly, wass1_nd_poly])
errors.add_row(["NGRC",       valid_pred_time_ngrc, mse_ngrc, nmse_ngrc, mae_ngrc, mdae_ngrc, mape_ngrc, r2_ngrc, specwelch_ngrc, specpgram_ngrc, wass1_nd_ngrc])
print(errors)
