# %% 
# Imports

from estimators.volt_funcs import Volterra
from estimators.ngrc_funcs import NGRC
from estimators.polykernel_funcs import PolynomialKernel

from datagen.data_generate_dde import dde_rk45
from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse, calculate_wasserstein1err, calculate_specdensloss
from systems.ddes import mackeyglass

from time import process_time

#%% 
# Generate dataset

def init(t):
    return 1.2

mg_args = {'delay': 17, 'a': 0.2, 'b': 0.1, 'n': 10 }

h = 0.02
n_intervals = 350
slicing = int(1 / h)

data = dde_rk45(n_intervals, init, mackeyglass, h, mg_args)[1][::slicing]

ndata = len(data)
ntrain = 3000
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
output_volt = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_volt = calculate_mse(test_teacher, output_volt, shift, scale)
wass1_volt = calculate_wasserstein1err(test_teacher, output_volt, shift, scale)
spec_volt = calculate_specdensloss(test_teacher, output_volt, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_volt], "images/mg_volterra_plot.pdf", figsize=(13, 3))
plot_data_distributions([test_teacher, output_volt], "images/mg_volterra_dist.pdf", figsize=(5,3))

# %% 
# Volterra with L2 least squares regression using pinv 

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.9, 0.4, 1e-09, 1000 

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout, pinv=True)
output_volt_pinv = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_volt_pinv = calculate_mse(test_teacher, output_volt_pinv, shift, scale)
wass1_volt_pinv = calculate_wasserstein1err(test_teacher, output_volt_pinv, shift, scale)
spec_volt_pinv = calculate_specdensloss(test_teacher, output_volt_pinv, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_volt_pinv], "images/mg_volterrapinv.pdf", figsize=(13, 3))
plot_data_distributions([test_teacher, output_volt_pinv], "images/mg_volterrapinv_dist.pdf", figsize=(5,3))

# %% 
# NGRC defaults with pinv

# Normalise the arrays for NGRC
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 2, 2, 0.1, 2

# Run the new NGRC class
ngrc = NGRC(ndelay, deg, reg, washout)
start_ngrc_time = process_time()
output_ngrc = ngrc.Train(train_input, train_teacher-train_input).PathContinue(train_teacher[-1], test_teacher.shape[0])
end_ngrc_time = process_time()

# Print the total training and forecasting time
print(f"Amount of time taken to train NGRC: {end_ngrc_time - start_ngrc_time}")

# Compute the errors
mse_ngrc = calculate_mse(test_teacher, output_ngrc, shift, scale)
wass1_ngrc = calculate_wasserstein1err(test_teacher, output_ngrc, shift, scale)
spec_ngrc = calculate_specdensloss(test_teacher, output_ngrc, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_ngrc], "images/mg_ngrc.pdf", figsize=(13, 3))
plot_data_distributions([test_teacher, output_ngrc], "images/mg_ngrc_dist.pdf", figsize=(5,3))

# %% 
# Polynomial kernel 

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for PolyKernel
deg, ndelays, reg, washout = 4, 17, 1e-06, 1000  #4, 18, 1e-05, 101 

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout)
start_poly_time = process_time()
output_poly = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])
end_poly_time = process_time()

# Print the total training and forecasting time
print(f"Amount of time taken to train polynomial kernel: {end_poly_time - start_poly_time}")

# Compute the errors
mse_poly = calculate_mse(test_teacher, output_poly, shift, scale)
wass1_poly = calculate_wasserstein1err(test_teacher, output_poly, shift, scale)
spec_poly = calculate_specdensloss(test_teacher, output_poly, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_poly], "images/mg_polykernel.pdf", figsize=(13, 3))
plot_data_distributions([test_teacher, output_poly], "images/mg_polykernel_dist.pdf", figsize=(5,3))

# %% 
# Polynomial kernel with pinv

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for PolyKernel
deg, ndelays, reg, washout = 4, 17, 1e-06, 1000 #4, 18, 1e-05, 101

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout, pinv=True)
start_poly_pinv_time = process_time()
output_poly_pinv = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])
end_poly_pinv_time = process_time()

# Print the total training and forecasting time
print(f"Amount of time taken to train polynomial kernel with pinv: {end_poly_pinv_time - start_poly_pinv_time}")

# Compute the errors
mse_poly_pinv = calculate_mse(test_teacher, output_poly_pinv, shift, scale)
wass1_poly_pinv = calculate_wasserstein1err(test_teacher, output_poly_pinv, shift, scale)
spec_poly_pinv = calculate_specdensloss(test_teacher, output_poly_pinv, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output_poly_pinv], "images/mg_polykernelpinv.pdf", figsize=(13, 3))
plot_data_distributions([test_teacher, output_poly_pinv], "images/mg_polykernelpinv_dist.pdf", figsize=(5,3))

# %% 
# Print MSEs

print("Method: MSE, Wasserstein1, Spectral Density Distance")
print(f"Volterra:                    {mse_volt}, {wass1_volt}, {spec_volt}")
print(f"Volterra with pinv:          {mse_volt_pinv}, {wass1_volt_pinv}, {spec_volt_pinv}")
print(f"NGRC:                        {mse_ngrc}, {wass1_ngrc}, {spec_ngrc}")
print(f"Polynomial Kernel:           {mse_poly}, {wass1_poly}, {spec_poly}")
print(f"Polynomial Kernel with pinv: {mse_poly_pinv}, {wass1_poly_pinv}, {spec_poly_pinv}")
