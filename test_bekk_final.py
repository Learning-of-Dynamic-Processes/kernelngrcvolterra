# %%
# Imports 

import mat73
import time

from estimators.volt_funcs import Volterra
from estimators.polykernel_funcs import PolynomialKernel
from estimators.ngrc_funcs import NGRC

from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse, calculate_nmse, calculate_mae, calculate_mdae_err, calculate_r2_err, calculate_mape_err
from utils.errors import calculate_wasserstein1_nd_err, calculate_specdens_periodogram_err, calculate_specdens_welch_err

from prettytable import PrettyTable
import numpy as np

# %% 
# Preparing datasets

# Load BEKK dataset
matstruct_contents = mat73.loadmat("./datagen/BEKK_d15_data.mat")

# Extract variables of interest from data
returns = matstruct_contents['data_sim']
epsilons = matstruct_contents['exact_epsilons']
Ht_sim_vech = matstruct_contents['Ht_sim_vech']

# Assign input and output data
ndata = 3760
data_in = epsilons[0:ndata-1, :]
data_out = Ht_sim_vech[1:ndata, :] * 1000

# Define the length of training and testing sizes
ntrain = 3007
ntest = ndata - ntrain

# Construct the training input and teacher, testing input and teacher
training_input_orig = data_in[0:ntrain] 
training_teacher_orig = data_out[0:ntrain]
testing_input_orig = data_in[ntrain:]
testing_teacher_orig = data_out[ntrain:]

# %%
# Volterra reservoir kernel with L2 least-squares regression 

# Normalise arrays -- inputs, shift so that L2 norm is 0 and mean is 0, as needed by Volterra kernels
normed_inputs = normalise_arrays([training_input_orig, testing_input_orig], norm_type="ScaleL2Shift")
train_input_volt, test_input_volt = normed_inputs[0]
shift_input_volt, scale_input_volt = normed_inputs[1], normed_inputs[2]

# Normalise arrays -- outputs, standardised to standard normal distribution
normed_outputs = normalise_arrays([training_teacher_orig, testing_teacher_orig], norm_type="NormStd")
train_teacher_volt, test_teacher_volt = normed_outputs[0]
shift_output_volt, scale_output_volt = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.9, 0.6, 0.001, 100

# Start timer
start = time.time()

# Run Volterra class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output_volt = volt.Train(train_input_volt, train_teacher_volt).Forecast(test_input_volt)

# Print time taken for training and generating outputs
print(f"Volterra took: {time.time() - start}")

# %%
# Polynomial kernel with least squares regression

# Normalise arrays -- inputs, shift so that L2 norm is 0 and mean is 0, as needed by Polynomial kernels
normed_inputs = normalise_arrays([training_input_orig, testing_input_orig], norm_type="MinMax")
train_input_poly, test_input_poly = normed_inputs[0]
shift_input, scale_input = normed_inputs[1], normed_inputs[2]

# Normalise arrays -- outputs, standardised to standard normal distribution
normed_outputs = normalise_arrays([training_teacher_orig, testing_teacher_orig], norm_type="NormStd")
train_teacher_poly, test_teacher_poly = normed_outputs[0]
shift_output_poly, scale_output_poly = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for Polynomial Kernel
deg, ndelays, reg, washout = 2, 1, 0.1, 0

# Start timer
start = time.time()

# Run Polynomial kernel class
poly = PolynomialKernel(deg, ndelays, reg, washout)
output_poly = poly.Train(train_input_poly, train_teacher_poly).Forecast(test_input_poly)

# Print time taken for training and generating outputs
print(f"Polynomial kernel took: {time.time() - start}")

# %%
# NGRC least squares regression

# Normalise arrays -- inputs
normed_inputs = normalise_arrays([training_input_orig, testing_input_orig], norm_type=None)
train_input_ngrc, test_input_ngrc = normed_inputs[0]
shift_input_ngrc, scale_input_ngrc = normed_inputs[1], normed_inputs[2]

# Normalise arrays -- outputs
normed_outputs = normalise_arrays([training_teacher_orig, testing_teacher_orig], norm_type="NormStd")
train_teacher_ngrc, test_teacher_ngrc = normed_outputs[0]
shift_output_ngrc, scale_output_ngrc = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 1, 2, 0.1, 0

# Start timer
start = time.time()

# Run NGRC class
ngrc = NGRC(ndelay, deg, reg, washout, isPathContinue=False)
output_ngrc = ngrc.Train(train_input_ngrc, train_teacher_ngrc).Forecast(test_input_ngrc)

# Print time taken for training and generating outputs
print(f"NGRC took: {time.time() - start}")

# %% 
# Plot time series evolution

t_display = 752 #300
target_display = 1
plot_data([test_teacher_volt[0:t_display, target_display].reshape(-1, 1), output_volt[0:t_display, target_display].reshape(-1, 1)], shift=shift_output_volt[target_display], scale=scale_output_volt[target_display], figsize=(15, 4), filename="images/bekk_volt.pdf", xlabel=[f"$H_{target_display}$"], datalabel=['actual', 'output'])
plot_data([test_teacher_poly[0:t_display, target_display].reshape(-1, 1), output_poly[0:t_display, target_display].reshape(-1, 1)], shift=shift_output_poly[target_display], scale=scale_output_poly[target_display], figsize=(15, 4), filename="images/bekk_poly.pdf", xlabel=[f"$H_{target_display}$"], datalabel=['actual', 'output'])
plot_data([test_teacher_ngrc[0:t_display, target_display].reshape(-1, 1), output_ngrc[0:t_display, target_display].reshape(-1, 1)], shift=shift_output_ngrc[target_display], scale=scale_output_ngrc[target_display], figsize=(15, 4), filename="images/bekk_ngrc.pdf", xlabel=[f"$H_{target_display}$"], datalabel=['actual', 'output'])

# %% 
# Calculate time step to time step errors 

# Volterra
mse_volt = calculate_mse(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)
nmse_volt = calculate_nmse(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)
mdae_volt = calculate_mdae_err(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)
mape_volt = calculate_mape_err(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)
mae_volt = calculate_mae(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)
r2_volt = calculate_r2_err(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)

# Polynomial
mse_poly = calculate_mse(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)
nmse_poly = calculate_nmse(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)
mdae_poly = calculate_mdae_err(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)
mape_poly = calculate_mape_err(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)
mae_poly = calculate_mae(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)
r2_poly = calculate_r2_err(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)

# NG-RC
mse_ngrc = calculate_mse(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)
nmse_ngrc = calculate_nmse(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)
mdae_ngrc = calculate_mdae_err(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)
mape_ngrc = calculate_mape_err(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)
mae_ngrc = calculate_mae(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)
r2_ngrc = calculate_r2_err(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)

# %%
# Print time step to time step errors

errors = PrettyTable(['Method', 'MSE', 'Normalised MSE', 'MdAE', 'MAE', 'MAPE', 'R2-score'])
errors.add_row(["Volterra",   mse_volt, nmse_volt, mdae_volt, mae_volt, mape_volt, r2_volt])
errors.add_row(["Polynomial", mse_poly, nmse_poly, mdae_poly, mae_poly, mape_poly, r2_poly])
errors.add_row(["NGRC",       mse_ngrc, nmse_ngrc, mdae_ngrc, mae_ngrc, mape_ngrc, r2_ngrc])
print(errors)

# %%
# Compute climate metrics

# Volterra
specwelch_volt = calculate_specdens_welch_err(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt, nperseg=128)
specpgram_volt = calculate_specdens_periodogram_err(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)

# Polymomial
specwelch_poly = calculate_specdens_welch_err(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly, nperseg=128)
specpgram_poly = calculate_specdens_periodogram_err(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)

# NG-RC
specwelch_ngrc = calculate_specdens_welch_err(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc, nperseg=128)
specpgram_ngrc = calculate_specdens_periodogram_err(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)

# %% 
# Climate metrics plots isolate most significant dimension

# Select indices to plot display
indices = [84, 117]

# Volterra
calculate_specdens_welch_err(np.take(test_teacher_volt, indices, axis=1) , np.take(output_volt, indices, axis=1), np.take(shift_output_volt, indices), np.take(scale_output_volt, indices), nperseg=128, ifPlot=True, figname="images/bekk_volt_welch.pdf", dimlabel=[85, 118])
calculate_specdens_periodogram_err(np.take(test_teacher_volt, indices, axis=1), np.take(output_volt, indices, axis=1), np.take(shift_output_volt, indices), np.take(scale_output_volt, indices), ifPlot=True, dimlabel=[85, 118])

# Polymomial
calculate_specdens_welch_err(np.take(test_teacher_poly, indices, axis=1), np.take(output_poly, indices, axis=1), np.take(shift_output_poly, indices), np.take(scale_output_poly, indices), nperseg=128, ifPlot=True, figname="images/bekk_poly_welch.pdf", dimlabel=[85, 118])
calculate_specdens_periodogram_err(np.take(test_teacher_poly, indices, axis=1), np.take(output_poly, indices, axis=1), np.take(shift_output_poly, indices), np.take(scale_output_poly, indices), ifPlot=True, dimlabel=[85, 118])

# NG-RC
calculate_specdens_welch_err(np.take(test_teacher_ngrc, indices, axis=1), np.take(output_ngrc, indices, axis=1), np.take(shift_output_ngrc, indices), np.take(scale_output_ngrc, indices), nperseg=128,  ifPlot=True, figname="images/bekk_ngrc_welch.pdf", dimlabel=[85, 118])
calculate_specdens_periodogram_err(np.take(test_teacher_ngrc, indices, axis=1), np.take(output_ngrc, indices, axis=1), np.take(shift_output_ngrc, indices), np.take(scale_output_ngrc, indices), ifPlot=True, dimlabel=[85, 118])

# %%
# Plot distributions

target_display = 117#1
plot_data_distributions([test_teacher_volt[:, target_display].reshape(-1, 1), output_volt[:, target_display].reshape(-1, 1)], "images/bekk_volt_dist.pdf", xlabel=[f"$\Sigma_{{{target_display+1}}}$"], datalabel=['actual', 'output'], figsize=(8,5))
plot_data_distributions([test_teacher_poly[:, target_display].reshape(-1, 1), output_poly[:, target_display].reshape(-1, 1)], "images/bekk_poly_dist.pdf", xlabel=[f"$\Sigma_{{{target_display+1}}}$"], datalabel=['actual', 'output'], figsize=(8,5))
plot_data_distributions([test_teacher_ngrc[:, target_display].reshape(-1, 1), output_ngrc[:, target_display].reshape(-1, 1)], "images/bekk_ngrc_dist.pdf", xlabel=[f"$\Sigma_{{{target_display+1}}}$"], datalabel=['actual', 'output'], figsize=(8,5))

# %%
# Compute distribution metrics

wass1_nd_volt = calculate_wasserstein1_nd_err(test_teacher_volt, output_volt, shift_output_volt, scale_output_volt)
wass1_nd_poly = calculate_wasserstein1_nd_err(test_teacher_poly, output_poly, shift_output_poly, scale_output_poly)
wass1_nd_ngrc = calculate_wasserstein1_nd_err(test_teacher_ngrc, output_ngrc, shift_output_ngrc, scale_output_ngrc)

# %% 
# Climate error metrics

climate_err_table = PrettyTable(['Method', 'Wasserstein1', 'SpecDens (Welch)', 'SpecDens (PGram)'])
climate_err_table.add_row(["Volterra",   wass1_nd_volt, specwelch_volt, specpgram_volt])
climate_err_table.add_row(["Polynomial", wass1_nd_poly, specwelch_poly, specpgram_poly])
climate_err_table.add_row(["NGRC",       wass1_nd_ngrc, specwelch_ngrc, specpgram_ngrc])
print(climate_err_table)

# %%
# Print errors into table

errors = PrettyTable(['Method', 'MSE', 'Normalised MSE', 'MAE', 'MdAE', 'MAPE', 'R2-score', 'SpecDens (Welch)', 'SpecDens (PGram)', 'Wass1_nd'])
errors.add_row(["Volterra",   mse_volt, nmse_volt, mae_volt, mdae_volt, mape_volt, r2_volt, specwelch_volt, specpgram_volt, wass1_nd_volt])
errors.add_row(["Polynomial", mse_poly, nmse_poly, mae_poly, mdae_poly, mape_poly, r2_poly, specwelch_poly, specpgram_poly, wass1_nd_poly])
errors.add_row(["NGRC",       mse_ngrc, nmse_ngrc, mae_ngrc, mdae_ngrc, mape_ngrc, r2_ngrc, specwelch_ngrc, specpgram_ngrc, wass1_nd_ngrc])
print(errors)
