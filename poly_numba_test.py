### Comparison of polynomial kernel with and without numba.
### Can check that with kernel values pass np.allclose but are not fully the same
### Results in alpha values being different when numba is used and not. 
### Can generate print statements with np.dot to realise that the dot function has rounding errors
### that are handled differently regardless of whether numba is used or not. 
### Meaning that np.dot is inconsistent with rounding errors with the actual implementation
### Whether or not numba is used. But it looks like implementation is consistent so long as
### one is consistent when numba is used and when not (i.e. different runs of the same code)
### So just fix a decision and just be consistent. 

# %% 
import time

from estimators.polykernel_funcs import PolynomialKernel
from estimators.polykernel_funcs_old import PolynomialKernel as PolynomialKernel2

from datagen.data_generate_ode import rk45 
from utils.normalisation import normalise_arrays
from utils.plotting import plot_data, plot_data_distributions
from utils.errors import calculate_mse, calculate_nmse, calculate_wasserstein1err
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
output_poly = polykernel.Train(train_input, train_teacher)
end = time.time()
print(f"Original time: {end - start}")

# With numba 
polykernel2 = PolynomialKernel2(deg, ndelays, reg, washout)
output_poly2 = polykernel2.Train(train_input, train_teacher)
print(f"New time: {time.time() - end}")

#%% 

start = time.time()
forecast = polykernel.Forecast(test_input)
end = time.time()
print(f"Original: {end - start}")
forecast2 = polykernel2.Forecast(test_input)
print(f"New: {end - time.time()}")

# %% 

start = time.time()
pathcontinue = polykernel.PathContinue(train_teacher[-1], test_teacher.shape[0])
end = time.time()
print(f"Original: {end - start}")
pathcontinue2 = polykernel2.PathContinue(train_teacher[-1], test_teacher.shape[0])
print(f"New: {end - time.time()}")

# %% 

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

