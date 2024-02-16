# %% Data generation demonstration-- ODEs

import numpy as np
import datagen.data_generate as data_gen

# Ordinary differential equation
def lorenz(t, Z, args):
    
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

# Define the arguments that go into the Lorenz equation
lor_args = (10, 8/3, 28)

# Define the initial conditions
Z0 = (0, 1, 1.05)

# Integration time steps
h = 0.005

# Start and end times of integration
t_span = (0, 40)

# Slice data depending on how many steps to observe
slicing = int(h/h)

# RK45 to obtain numerical solution
data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)[1]
data = data[::slicing]

# Check that dataset outputs as (nsamples, nfeatures)
print(data.shape)

# %% Data generation demonstration -- DDEs

import numpy as np 
import datagen.data_generate_delay as data_gen

# Delay differential equation
def mackeyglass(t, z, z_lag, mg_args):
    a = mg_args['a']
    b = mg_args['b']
    n = mg_args['n']
    return (a * z_lag) / (1 + z_lag**n) - b*z

# Define an initial function
def init(t):
    return 1.2

# Arguments that go into the differential equation
mg_args = {'delay': 17, 'a': 0.2, 'b': 0.1, 'n': 10 }

# Integration time steps
h = 0.02

# Number of delay intervals to integrate for
n_intervals = 200

# Slice data depending on how many steps to observe
slicing = int(1/h)

# RK45 for DDES to obtain numerical solution
data = data_gen.dde_rk45(n_intervals, init, mackeyglass, h, mg_args)[1]
data = data[::slicing]

# Check that dataset outputs as (nsamples, nfeatures)
print(data.shape)

# %% Plotting functions for datasets

from utils.plotting import plot_data

# Check how the datasets look like when plotted 
plot_data(data, plot_mode="1d")
plot_data(data, plot_mode="nd")

# %% Split data

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

# %% TODO: Data normalisation check

from utils.normalisation import normalise_arrays

## Scale by mean and standard deviation
print("NormStd check")
training_input, training_teacher, testing_input, testing_teacher = normalise_arrays(
    [training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="NormStd")
# Check that mean and std of training input is correct 
std_check = np.std(training_input, axis=0)
mean_check = np.mean(training_input, axis=0)
print(std_check, "\n", mean_check)
# Check that remaining datasets have been normed and scaled
print(np.allclose(training_teacher, training_teacher_orig))     # Should throw back False
print(np.allclose(testing_input, testing_input_orig))           # Should throw back False
print(np.allclose(testing_teacher, testing_teacher_orig))       # Should throw back False


## Scale by min and max with min max range (0, 1)
print("MinMax check")
training_input, training_teacher, testing_input, testing_teacher = normalise_arrays(
    [training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax")
# Check that mean and std of training input is correct 
min_check = np.min(training_input, axis=0)
max_check = np.max(training_input, axis=0)
print(min_check, "\n", max_check)
# Check that remaining datasets have been normed and scaled
print(np.allclose(training_teacher, training_teacher_orig))     # Should throw back False
print(np.allclose(testing_input, testing_input_orig))           # Should throw back False
print(np.allclose(testing_teacher, testing_teacher_orig))       # Should throw back False


## Scale by min and max with min max range (0, 1/2)
print("MinMax check")
training_input, training_teacher, testing_input, testing_teacher = normalise_arrays(
    [training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="MinMax", MinMax_range=(0, 0.5))
# Check that mean and std of training input is correct 
min_check = np.min(training_input, axis=0)
max_check = np.max(training_input, axis=0)
print(min_check, "\n", max_check)
# Check that remaining datasets have been normed and scaled
print(np.allclose(training_teacher, training_teacher_orig))     # Should throw back False
print(np.allclose(testing_input, testing_input_orig))           # Should throw back False
print(np.allclose(testing_teacher, testing_teacher_orig))       # Should throw back False


## Scale by L2 norm but without shifting
print("ScaleL2 check")
training_input, training_teacher, testing_input, testing_teacher = normalise_arrays(
    [training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2")
# Check that mean and norm of training input is correct 
norm_check = np.max([np.linalg.norm(z) for z in training_input])
mean_check = np.mean(training_input, axis=0)                    # Check that norm is not 0
print(norm_check, "\n", mean_check)
# Check that remaining datasets have been normed and scaled
print(np.allclose(training_teacher, training_teacher_orig))     # Should throw back False
print(np.allclose(testing_input, testing_input_orig))           # Should throw back False
print(np.allclose(testing_teacher, testing_teacher_orig))       # Should throw back False


## Scale by the L2 norm with shift
print("ScaleL2Shift check")
training_input, training_teacher, testing_input, testing_teacher = normalise_arrays(
    [training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="ScaleL2Shift")
# Check that mean and norm of training input is correct 
norm_check = np.max([np.linalg.norm(z) for z in training_input])
mean_check = np.mean(training_input, axis=0)
print(norm_check, "\n", mean_check)
# Check that remaining datasets have been normed and scaled
print(np.allclose(training_teacher, training_teacher_orig))     # Should throw back False
print(np.allclose(testing_input, testing_input_orig))           # Should throw back False
print(np.allclose(testing_teacher, testing_teacher_orig))       # Should throw back False


## Check to make sure that no normalisation also outputs properly
print("No normalisation")
training_input, training_teacher, testing_input, testing_teacher = normalise_arrays(
    [training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig])
print(np.allclose(training_input, training_input_orig))         # Should throw back True
print(np.allclose(training_teacher, training_teacher_orig))     # Should throw back True
print(np.allclose(testing_input, testing_input_orig))           # Should throw back True
print(np.allclose(testing_teacher, testing_teacher_orig))       # Should throw back True

## Check to make sure error raises properly
training_input, training_teacher, testing_input, testing_teacher = normalise_arrays(
    [training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="Apples")

# %% Volterra class check

import numpy as np
from time import process_time
import estimators.volt_funcs as volt
import estimators.volt_funcs2 as volt2
from utils.plotting import plot_data
from utils.normalisation import normalise_arrays
from utils.errors import calculate_mse

# Define parameters needed to build the reservoir
tau_coef = 0.2
ld_coef = 0.8

# Define regularisation parameter for training
reg = 1e-10 

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

# Normalise data so that L2 norm is 1 and mean is 0
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type="NormStd")
training_input, training_teacher, testing_input, testing_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define the instance of Volterra class - volt
volt_train_start = process_time()
Volterra = volt.Volterra(ld_coef=ld_coef, tau_coef=tau_coef, reg=reg, washout=washout, regression="L2", pinv=False)
Volterra.Train(training_input=training_input, training_teacher=training_teacher)
volt_train_end = process_time()
output_forecast = Volterra.Forecast(testing_input=testing_input)
volt_forecast_end = process_time()
output_pathcontinue = Volterra.PathContinue(latest_input=training_teacher[-1], nhorizon=testing_teacher.shape[0])
volt_end = process_time()
print(f"Volterra training took {volt_train_end - volt_train_start}")
print(f"Volterra forecasting took {volt_forecast_end - volt_train_end}")
print(f"Volterra path continuing took {volt_end - volt_forecast_end}")

# Train using the old functions - volt2

# Define parameters needed to build the reservoir
M = np.max([np.linalg.norm(z) for z in training_input])
tau = np.sqrt(1 / M**2)
tau = tau * tau_coef
ld = np.sqrt(1 - (tau**2) * (M**2)) * ld_coef

# Define regularisation parameter for training
reg = 1e-10 

# Run training and forecasting 
volt2_train_start = process_time()
alpha_ols, alpha0_ols, K = volt2.Train(training_input, training_teacher, washout, ld, tau, reg, pinv=False)
volt2_train_end = process_time()
pred_nonauto = volt2.Forecast(K, training_input, testing_input, alpha_ols, alpha0_ols, washout, ld, tau)  
volt2_forecast_end = process_time()
pred_auto = volt2.ForecastAuto(K, training_input, training_teacher[-1], alpha_ols, alpha0_ols, washout, ntest, ld, tau)
volt2_end = process_time()
print(f"Volterra training took {volt2_train_end - volt2_train_start}")
print(f"Volterra forecasting took {volt2_forecast_end - volt2_train_end}")
print(f"Volterra path continuing took {volt2_end - volt2_forecast_end}")

# Check that volt and volt2 return the same things
print(np.allclose(alpha_ols, Volterra.alpha))
print(np.allclose(alpha0_ols, Volterra.alpha0))
print(np.allclose(output_forecast, pred_nonauto))
print(np.allclose(output_pathcontinue, pred_auto))

# Compute the mse considering the amount that was shifted and scaled by
print(calculate_mse(output_pathcontinue, testing_teacher, shift, scale))
      
# Plot and check forecast superimposed on training and testing teacher
plot_data([output_forecast, testing_teacher], plot_mode="1d")
plot_data([output_pathcontinue, testing_teacher], plot_mode="1d")

# %% NGRC class check

import numpy as np
import estimators.ngrc_funcs as ngrc
from utils.plotting import plot_data
from utils.normalisation import normalise_arrays
from utils.errors import calculate_mse

# Define parameters to built NGRC feature space
ndelay = 2
deg = 2

# Define regularisation parameters
reg = 1e-4

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

# Normalise data so that L2 norm is 1 and mean is 0
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
training_input, training_teacher, testing_input, testing_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define the instance of the NGRC class
NGRC = ngrc.NGRC(ndelay=ndelay, deg=deg, reg=reg, washout=washout)
NGRC.Train(training_input, training_teacher)
output_pathcontinue = NGRC.PathContinue(latest_input=training_teacher[-1], nhorizon=testing_teacher.shape[0])

# Compute the mse considering the amount that was shifted and scaled by
print(calculate_mse(output_pathcontinue, testing_teacher, shift, scale))
      
# Plot and check forecast superimposed on training and testing teacher
plot_data([output_pathcontinue, testing_teacher], plot_mode="1d")

# %% SINDy class check

import numpy as np
import estimators.sindy_funcs as SINDy
from utils.plotting import plot_data
from utils.normalisation import normalise_arrays
from utils.errors import calculate_mse

# Define regression parameters 
alpha = 1e-15
threshold = 0.1
deg = 5

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

# Normalise data so that L2 norm is 1 and mean is 0
normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
training_input, training_teacher, testing_input, testing_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define and train the SINDy model
SINDy = SINDy.SINDyPolynomialSTLSQ(alpha=alpha, threshold=threshold, deg=deg, dt=h*slicing)
SINDy.Train(training_input, training_teacher)
output_pathcontinue = SINDy.PathContinue(training_teacher[-1], testing_teacher.shape[0])

# Compute the mse considering the amount that was shifted and scaled by
print(calculate_mse(output_pathcontinue, testing_teacher, shift, scale))
      
# Plot and check forecast superimposed on training and testing teacher
plot_data([output_pathcontinue, testing_teacher], plot_mode="1d")

# %% TODO: Check that the codes run the same as before 

# %% TODO: Small examples to ensure that everything runs correctly

# %% CV code check

import numpy as np
from utils.crossvalidation import CrossValidate
from estimators.volt_funcs import Volterra

# Define a sample training data
data_len = 97
training_data = np.linspace(0, data_len-1, data_len).reshape((-1, 1))
training_input = training_data[0:data_len-1]
training_target = training_data[1:data_len]

# Use Volterra as example
'''
# Check cross-validation for one set of parameters - using defaults only
Volterra_params = (0.8, 0.2, 1e-10, 2)

mean_mse_paramset = CV.crossvalidate_per_parameters(Volterra, training_input, training_target, Volterra_params)
'''
# Check cv for range of params
CV = CrossValidate()
ld_coef_range = np.linspace(0.1, 0.2, 2)
tau_coef_range = np.linspace(0.1, 0.2, 2)
reg_range = np.logspace(-15, -14, 1)
param_ranges = [ld_coef_range, tau_coef_range, reg_range]
param_names = ["ld", "tau", "reg"]
param_add = [2, "L2", False]
CV.crossvalidate_multiprocessing(Volterra, training_input, training_target, param_ranges, param_names, param_add, num_processes=1)
