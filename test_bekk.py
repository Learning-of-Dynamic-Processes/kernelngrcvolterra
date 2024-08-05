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
import matplotlib.pyplot as plt

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
train_input, test_input = normed_inputs[0]
shift_input, scale_input = normed_inputs[1], normed_inputs[2]

# Normalise arrays -- outputs, standardised to standard normal distribution
normed_outputs = normalise_arrays([training_teacher_orig, testing_teacher_orig], norm_type="NormStd")
train_teacher, test_teacher = normed_outputs[0]
shift_output_volt, scale_output_volt = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.9, 0.6, 0.001, 100

# Start timer
start = time.time()

# Run Volterra class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output_volt = volt.Train(train_input, train_teacher).Forecast(test_input)

# Print time taken for training and generating outputs
print(f"Volterra took: {time.time() - start}")

# Calculate errors
mse_volt = calculate_mse(test_teacher, output_volt, shift_output_volt, scale_output_volt)
nmse_volt = calculate_nmse(test_teacher, output_volt, shift_output_volt, scale_output_volt)
mdae_volt = calculate_mdae_err(test_teacher, output_volt, shift_output_volt, scale_output_volt)
mape_volt = calculate_mape_err(test_teacher, output_volt, shift_output_volt, scale_output_volt)
mae_volt = calculate_mae(test_teacher, output_volt, shift_output_volt, scale_output_volt)
r2_volt = calculate_r2_err(test_teacher, output_volt, shift_output_volt, scale_output_volt)
wass1_nd_volt = calculate_wasserstein1_nd_err(test_teacher, output_volt, shift_output_volt, scale_output_volt)
specwelch_volt = calculate_specdens_welch_err(test_teacher, output_volt, shift_output_volt, scale_output_volt)
specpgram_volt = calculate_specdens_periodogram_err(test_teacher, output_volt, shift_output_volt, scale_output_volt)

# Plot the forecast and actual
t_display = 300
target_display = 1
plot_data([test_teacher[0:t_display, 0:target_display], output_volt[0:t_display, 0:target_display]], shift=shift_output_volt[0:target_display], scale=scale_output_volt[0:target_display], filename="images/bekk_volterra.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher[:, 0:target_display], output_volt[:, 0:target_display]], "images/bekk_volterra_dist.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])

# %%
# Polynomial kernel with least squares regression

# Normalise arrays -- inputs, shift so that L2 norm is 0 and mean is 0, as needed by Polynomial kernels
normed_inputs = normalise_arrays([training_input_orig, testing_input_orig], norm_type="MinMax")
train_input, test_input = normed_inputs[0]
shift_input, scale_input = normed_inputs[1], normed_inputs[2]

# Normalise arrays -- outputs, standardised to standard normal distribution
normed_outputs = normalise_arrays([training_teacher_orig, testing_teacher_orig], norm_type="NormStd")
train_teacher, test_teacher = normed_outputs[0]
shift_output_poly, scale_output_poly = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for Polynomial Kernel
deg, ndelays, reg, washout = 2, 1, 0.1, 0

# Start timer
start = time.time()

# Run Polynomial kernel class
poly = PolynomialKernel(deg, ndelays, reg, washout)
output_poly = poly.Train(train_input, train_teacher).Forecast(test_input)

# Print time taken for training and generating outputs
print(f"Polynomial kernel took: {time.time() - start}")

# Calculate errors
mse_poly = calculate_mse(test_teacher, output_poly, shift_output_poly, scale_output_poly)
nmse_poly = calculate_nmse(test_teacher, output_poly, shift_output_poly, scale_output_poly)
mdae_poly = calculate_mdae_err(test_teacher, output_poly, shift_output_poly, scale_output_poly)
mape_poly = calculate_mape_err(test_teacher, output_poly, shift_output_poly, scale_output_poly)
mae_poly = calculate_mae(test_teacher, output_poly, shift_output_poly, scale_output_poly)
r2_poly = calculate_r2_err(test_teacher, output_poly, shift_output_poly, scale_output_poly)
wass1_nd_poly = calculate_wasserstein1_nd_err(test_teacher, output_poly, shift_output_poly, scale_output_poly)
specwelch_poly = calculate_specdens_welch_err(test_teacher, output_poly, shift_output_poly, scale_output_poly)
specpgram_poly = calculate_specdens_periodogram_err(test_teacher, output_poly, shift_output_poly, scale_output_poly)

# Plot the forecast and actual
t_display = 300
target_display = 1
plot_data([test_teacher[0:t_display, 0:target_display], output_poly[0:t_display, 0:target_display]], shift=shift_output_poly[0:target_display], scale=scale_output_poly[0:target_display], filename="images/bekk_polykernel.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher[:, 0:target_display], output_poly[:, 0:target_display]], "images/bekk_polykernel_dist.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])

# %%
# NGRC least squares regression

# Normalise arrays -- inputs
normed_inputs = normalise_arrays([training_input_orig, testing_input_orig], norm_type=None)
train_input, test_input = normed_inputs[0]
shift_input, scale_input = normed_inputs[1], normed_inputs[2]

# Normalise arrays -- outputs
normed_outputs = normalise_arrays([training_teacher_orig, testing_teacher_orig], norm_type="NormStd")
train_teacher, test_teacher = normed_outputs[0]
shift_output_ngrc, scale_output_ngrc = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 1, 2, 0.1, 0

# Start timer
start = time.time()

# Run NGRC class
ngrc = NGRC(ndelay, deg, reg, washout, isPathContinue=False)
output_ngrc = ngrc.Train(train_input, train_teacher).Forecast(test_input)

# Print time taken for training and generating outputs
print(f"NGRC took: {time.time() - start}")

# Calculate errors
mse_ngrc = calculate_mse(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
nmse_ngrc = calculate_nmse(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
mdae_ngrc = calculate_mdae_err(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
mape_ngrc = calculate_mape_err(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
mae_ngrc = calculate_wasserstein1_nd_err(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
r2_ngrc = calculate_r2_err(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
wass1_nd_ngrc = calculate_wasserstein1_nd_err(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
specwelch_ngrc = calculate_specdens_welch_err(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)
specpgram_ngrc = calculate_specdens_periodogram_err(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)

# Plot the forecast and actual
t_display = 300
target_display = 1
plot_data([test_teacher[0:t_display, 0:target_display], output_ngrc[0:t_display, 0:target_display]], shift=shift_output_ngrc[0:target_display], scale=scale_output_ngrc[0:target_display], filename="images/bekk_ngrc.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher[:, 0:target_display], output_ngrc[:, 0:target_display]], "images/bekk_ngrc_dist.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])

# %%
# Print errors into table

errors = PrettyTable(['Method', 'MSE', 'Normalised MSE', 'MAE', 'Wasserstein1_nd', 'MdAE', 'MAPE', 'R2-score', 'SpecDens (Welch)', 'SpecDens (PGram)'])
errors.add_row(["Volterra",   mse_volt, nmse_volt, mae_volt, wass1_nd_volt, mdae_volt, mape_volt, r2_volt, specwelch_volt, specpgram_volt])
errors.add_row(["Polynomial", mse_poly, nmse_poly, mae_poly, wass1_nd_poly, mdae_poly, mape_poly, r2_poly, specwelch_poly, specpgram_poly])
errors.add_row(["NGRC",       mse_ngrc, nmse_ngrc, mae_ngrc, wass1_nd_ngrc, mdae_ngrc, mape_ngrc, r2_ngrc, specwelch_ngrc, specpgram_ngrc])
print(errors)

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

volt_diff = np.array(abs_diff_overtime(output_volt, testing_teacher_orig, shift_output_volt, scale_output_volt))
ngrc_diff = np.array(abs_diff_overtime(output_ngrc, testing_teacher_orig, shift_output_ngrc, scale_output_ngrc))
poly_diff = np.array(abs_diff_overtime(output_poly, testing_teacher_orig, shift_output_poly, scale_output_poly))

plt.figure(figsize=(12, 8))
plt.plot(volt_diff, label="Volterra", color="r", linewidth=0.8)
plt.plot(ngrc_diff, label="NG-RC", color="g", linewidth=0.8)
plt.plot(poly_diff, label="Polynomial kernel", color="b", linewidth=0.8)
plt.xlabel("time")
plt.ylabel("sum of absolute difference")
plt.yscale("log")
plt.legend()
plt.savefig("images/bekk_cum_ae.pdf")
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
volt_errors = error_vals_over_time(test_teacher, output_volt, shift_output_volt, scale_output_volt)
poly_errors = error_vals_over_time(test_teacher, output_poly, shift_output_poly, scale_output_poly)
ngrc_errors = error_vals_over_time(test_teacher, output_ngrc, shift_output_ngrc, scale_output_ngrc)

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

# %%
