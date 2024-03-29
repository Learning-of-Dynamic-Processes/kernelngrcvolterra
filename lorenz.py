# %% 
# Imports

from estimators.volt_funcs import Volterra
from estimators.ngrc_funcs import NGRC
from estimators.sindy_funcs import SINDyPolynomialSTLSQ
from estimators.polykernel_funcs import PolynomialKernel

from datagen.data_generate import rk45 
from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse, calculate_wasserstein1err
from systems.odes import lorenz

# %% 
# Prepare Lorenz datatsets

# Create the Lorenz dataset
lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)
h = 0.005
t_span = (0, 40)
t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)

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

# %% 
# Volterra with L2 least squares regression

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.8, 0.2, 1e-10, 1000

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_volt = calculate_mse(test_teacher, output, shift, scale)
wasserstein_volt = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# Volterra with L2 least squares regression using pinv 

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.8, 0.2, 1e-10, 1000

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout, pinv=True)
output = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_volt_pinv = calculate_mse(test_teacher, output, shift, scale)
wasserstein_volt_pinv = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])

# %% 
# Volterra with L2 least squares regression using pinv and only the last 28 features of the Gram matrix

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.8, 0.2, 1e-10, 1000

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout, nfeatures=28, pinv=True)
output = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_volt_pinv_28 = calculate_mse(test_teacher, output, shift, scale)
wasserstein_volt_pinv_28 = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])

# %% 
# NGRC defaults with pseudo inverse

# Normalise the arrays for NGRC
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 2, 2, 0.00012224984640507843, 2 # provided by Gauthier was 2.5e-6, washout 2

# Run the new NGRC class
ngrc = NGRC(ndelay, deg, reg, washout)
output = ngrc.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_ngrc = calculate_mse(test_teacher, output, shift, scale)
wasserstein_ngrc = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# SINDy

# Normalise the arrays for SINDy
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define the hyperparameters for SINDy
threshold, alpha, deg, washout = 0.1, 1e-15, 2, 1000

# Run the new SINDy functions
sindy = SINDyPolynomialSTLSQ(alpha, threshold, deg, h)
output = sindy.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_sindy = calculate_mse(test_teacher, output, shift, scale)
wasserstein_sindy = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# Polynomial kernel 

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for Poly Kernel
deg, ndelays, reg, washout = 2, 6, 1e-07, 1000

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout)
output = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_poly = calculate_mse(test_teacher, output, shift, scale)
wasserstein_poly = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# Polynomial kernel with pseudo inverse

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for Poly Kernel
deg, ndelays, reg, washout = 2, 6, 1e-07, 1000

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout, pinv=True)
output = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_poly_pinv = calculate_mse(test_teacher, output, shift, scale)
wasserstein_poly_pinv = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# Remove second and third dimensions from data

training_input_orig_x = training_input_orig[:, 0].reshape((-1, 1))
training_teacher_orig_x = training_teacher_orig[:, 0].reshape((-1, 1))
testing_input_orig_x = testing_input_orig[:, 0].reshape((-1, 1))
testing_teacher_orig_x = testing_teacher_orig[:, 0].reshape((-1, 1))

# %% 
# Volterra observing only the first dimension

# Normalise the arrays for Volterra
normalisation_output = normalise_arrays([training_input_orig_x, training_teacher_orig_x, testing_input_orig_x, testing_teacher_orig_x], norm_type="ScaleL2Shift")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
ld_coef, tau_coef, reg, washout = 0.4, 0.3, 1e-09, 1000 #0.99, 0.23, 0.1, 1000 

# Run Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
output = volt.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_volt_x = calculate_mse(test_teacher, output, shift, scale)
wasserstein_volt_x = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# NGRC observing only the first dimension

normalisation_output = normalise_arrays([training_input_orig_x, training_teacher_orig_x, testing_input_orig_x, testing_teacher_orig_x], norm_type=None)
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for NGRC
ndelay, deg, reg, washout = 2, 2, 15.6771196224781, 2

# Run the new NGRC class
ngrc = NGRC(ndelay, deg, reg, washout)
output = ngrc.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_ngrc_x = calculate_mse(test_teacher, output, shift, scale)
wasserstein_ngrc_x = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# SINDy observing only the first dimension

normalisation_output = normalise_arrays([training_input_orig_x, training_teacher_orig_x, testing_input_orig_x, testing_teacher_orig_x], norm_type=None)
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define the hyperparameters for SINDy
threshold, alpha, deg = 0.1, 1e-15, 2

# Run the new SINDy functions
sindy = SINDyPolynomialSTLSQ(alpha, threshold, deg, h)
output = sindy.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the mse
mse_sindy_x = calculate_mse(test_teacher, output, shift, scale)
wasserstein_sindy_x = calculate_wasserstein1err(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# Polynomial observing only the first dimension

# Normalise the arrays for Polykernel
normalisation_output = normalise_arrays([training_input_orig_x, training_teacher_orig_x, testing_input_orig_x, testing_teacher_orig_x], norm_type="MinMax")
train_input, train_teacher, test_input, test_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define hyperparameters for Poly Kernel
deg, ndelays, reg, washout = 2, 2, 1e-10, 1000

# Run the new polynomial functinos
polykernel = PolynomialKernel(deg, ndelays, reg, washout)
output = polykernel.Train(train_input, train_teacher).PathContinue(train_teacher[-1], test_teacher.shape[0])

# Compute the errors
mse_poly_x = calculate_mse(test_teacher, output, shift, scale)
wasserstein_poly_x = calculate_mse(test_teacher, output, shift, scale)

# Plot the forecast and actual
plot_data([test_teacher, output])
plot_data_distributions([test_teacher, output])

# %% 
# Print MSEs

print("Method: MSE, Wasserstein")
print(f"Volterra: {mse_volt}, {wasserstein_volt}")
print(f"Volterra with pinv: {mse_volt_pinv}, {wasserstein_volt_pinv}")
print(f"Volterra with pinv and 28 features: {mse_volt_pinv_28}, {wasserstein_volt_pinv_28}")
print(f"NGRC: {mse_ngrc}, {wasserstein_ngrc}, (CVed)")
print(f"SINDy: {mse_sindy}, {wasserstein_sindy}")
print(f"Polynomial Kernel: {mse_poly}, {wasserstein_poly}, (CVed)")
print(f"Polynomial Kernel with pinv: {mse_poly_pinv}, {wasserstein_poly_pinv}, (CVed)")

print()

print("Method with first dimension: MSE, Wasserstein")
print(f"Volterra with first dimension: {mse_volt_x}, {wasserstein_volt_x}")
print(f"NGRC with first dimension: {mse_ngrc_x}, {wasserstein_ngrc_x}, (CVed)")
print(f"Polynomial kernel with first dimension: {mse_poly_x}, {wasserstein_poly_x}, (CVed)")
print(f"SINDy with first dimension: {mse_sindy_x}, {wasserstein_sindy_x}")
