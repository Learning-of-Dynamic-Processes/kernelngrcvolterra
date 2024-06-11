# %%
# Imports 

import mat73
import time

from estimators.volt_funcs import Volterra
from estimators.polykernel_funcs import PolynomialKernel
from estimators.ngrc_funcs import NGRC

from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse, calculate_wasserstein1err, calculate_specdensloss, calculate_nmse

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
shift_output, scale_output = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.7240596996877566, 0.6866051722990268, 0.0024184687663713032, 0 

# Start timer
start = time.time()

# Run Volterra class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output_volt = volt.Train(train_input, train_teacher).Forecast(test_input)

# Print time taken for training and generating outputs
print(f"Volterra took: {time.time() - start}")

# Calculate errors
mse_volt = calculate_mse(test_teacher, output_volt, shift_output, scale_output)
nmse_volt = calculate_nmse(test_teacher, output_volt, shift_output, scale_output)
wass1_volt = calculate_wasserstein1err(test_teacher, output_volt, shift_output, scale_output)
spec_volt = calculate_specdensloss(test_teacher, output_volt, shift_output, scale_output)

# Plot the forecast and actual
t_display = 300
target_display = 2
plot_data([test_teacher[0:t_display, 0:target_display], output_volt[0:t_display, 0:target_display]], shift=shift_output[0:target_display], scale=scale_output[0:target_display], filename="images/bekk_volterra.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher[:, 0:target_display], output_volt[:, 0:target_display]], "images/bekk_volterra_dist.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])

# %%
# Polynomial kernel with least squares regression

# Normalise arrays -- inputs, shift so that L2 norm is 0 and mean is 0, as needed by Volterra kernels
normed_inputs = normalise_arrays([training_input_orig, testing_input_orig], norm_type="MinMax")
train_input, test_input = normed_inputs[0]
shift_input, scale_input = normed_inputs[1], normed_inputs[2]

# Normalise arrays -- outputs, standardised to standard normal distribution
normed_outputs = normalise_arrays([training_teacher_orig, testing_teacher_orig], norm_type="NormStd")
train_teacher, test_teacher = normed_outputs[0]
shift_output, scale_output = normed_outputs[1], normed_outputs[2]

# Define input hyperparameters for Polynomial Kernel
deg, ndelays, reg, washout = 2, 1, 0.4147626610198409, 0

# Start timer
start = time.time()

# Run Polynomial kernel class
poly = PolynomialKernel(deg, ndelays, reg, washout)
output_poly = poly.Train(train_input, train_teacher).Forecast(test_input)

# Print time taken for training and generating outputs
print(f"Polynomial kernel took: {time.time() - start}")

# Calculate errors
mse_poly = calculate_mse(test_teacher, output_poly, shift_output, scale_output)
nmse_poly = calculate_nmse(test_teacher, output_poly, shift_output, scale_output)
wass1_poly = calculate_wasserstein1err(test_teacher, output_poly, shift_output, scale_output)
spec_poly = calculate_specdensloss(test_teacher, output_poly, shift_output, scale_output)

# Plot the forecast and actual
t_display = 300
target_display = 2
plot_data([test_teacher[0:t_display, 0:target_display], output_poly[0:t_display, 0:target_display]], shift=shift_output[0:target_display], scale=scale_output[0:target_display], filename="images/bekk_polykernel.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])
plot_data_distributions([test_teacher[:, 0:target_display], output_poly[:, 0:target_display]], "images/bekk_polykernel_dist.pdf", xlabel=["$H_1$", "$H_2$"], datalabel=['actual', 'output'])

# %%
# NGRC least squares regression

