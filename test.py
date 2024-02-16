# %% Test normalisation and error functions 

import numpy as np
from utils.normalisation import normalise_arrays
from utils.errors import calculate_mse

# Define a sample dataset
sample_1 = np.linspace(1, 10, 10).reshape((-1, 2))
sample_2 = 2 * np.linspace(1, 10, 10).reshape((-1, 2))
sample_ls = [sample_1, sample_2]

#output, shift, scale = normalise_arrays(sample_ls, norm_type="NormStd")
output, shift, scale = normalise_arrays(sample_ls, norm_type="MinMax")
#output, shift, scale = normalise_arrays(sample_ls, norm_type="MinMax", MinMax_range=(0, 2))
#output, shift, scale = normalise_arrays(sample_ls, norm_type="ScaleL2")
#output, shift, scale = normalise_arrays(sample_ls, norm_type="ScaleL2Shift")
#output, shift, scale = normalise_arrays(sample_ls, norm_type=None)

print(np.allclose(output[0] * (1/scale) + shift, sample_1))

output, shift, scale = normalise_arrays(sample_ls, norm_type=None)
print(calculate_mse(sample_1, output[0], shift, scale))

# %% Test to check that estimator codes functions output as before

import numpy as np
from estimators.volt_funcs import Volterra
import estimators.volt_funcs2 as Volterra2
from estimators.ngrc_funcs import NGRC
import estimators.ngrc_funcs2 as NGRC2
from estimators.sindy_funcs import SINDyPolynomialSTLSQ
import pysindy as ps

# Define small sample dataset
sample = np.linspace(1, 40, 40).reshape((-1, 2))
ntrain = 10
washout = 2
ntest = len(sample) - ntrain

# Define the training and testing, teacher and input sets
training_input = sample[0:ntrain-1] 
training_teacher = sample[1:ntrain]
testing_input = sample[ntrain-1:ntrain+ntest-1]
testing_teacher = sample[ntrain:ntrain+ntest]

# Define input hyperparameters for Volterra
tau_coef = 0.1
ld_coef = 0.1
reg = 1e-10 

# Run new Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
pred_nonauto = volt.Train(training_input, training_teacher).Forecast(testing_input)
pred_auto = volt.PathContinue(training_teacher[-1], testing_teacher.shape[0])

# Run old Volterra functions
M = np.max([np.linalg.norm(z) for z in training_input])
tau = np.sqrt(1 / M**2) * tau_coef
ld = np.sqrt(1 - (tau**2) * (M**2)) * ld_coef
alpha_ols, alpha0_ols, K = Volterra2.Train(training_input, training_teacher, washout, ld, tau, reg, pinv=False)
pred_nonauto2 = Volterra2.Forecast(K, training_input, testing_input, alpha_ols, alpha0_ols, washout, ld, tau)  
pred_auto2 = Volterra2.ForecastAuto(K, training_input, training_teacher[-1], alpha_ols, alpha0_ols, washout, ntest, ld, tau)

# Check that the outputs match
print(np.allclose(pred_nonauto, pred_nonauto2))
print(np.allclose(pred_auto, pred_auto2))

# Define input hyperparameters for NGRC
ndim = 2
ndelay = 2
reg = 1e-4
deg = 2

# Run the new NGRC class
NGRC = NGRC(ndelay, deg, reg, washout)
NGRC.Train(training_input, training_teacher)
prediction = NGRC.PathContinue(training_teacher[-1], testing_teacher.shape[0])


# Run the old NGRC functions
training_input_ngrc, training_teacher_ngrc = training_input.T, training_teacher.T
W_out, X = NGRC2.Train(ndelay, deg, reg, training_input_ngrc, training_teacher_ngrc, washout)
prediction2 = NGRC2.Forecast(W_out, X, training_teacher[-1], deg, ntest)

print(np.allclose(prediction, prediction2.T))

# Define the hyperparameters for SINDy
threshold = 0.1
alpha = 1e-15
deg = 2

# Run the new SINDy functions
SINDy = SINDyPolynomialSTLSQ(alpha, threshold, deg, 1)
output = SINDy.Train(training_input, training_teacher).PathContinue(training_teacher[-1], testing_teacher.shape[0])

# Run the old SINDy functions
training = sample[0:ntrain]
testing = sample[ntrain:]
testing_t_eval = np.arange(10, 20, 1)
stlsq_optim = ps.STLSQ(threshold=threshold, alpha=alpha)
library = ps.PolynomialLibrary(degree=deg, include_interaction=True, interaction_only=False)
sindy_model = ps.SINDy(optimizer=stlsq_optim, feature_library=library)
sindy_model.fit(training, t=1)
output2 = sindy_model.simulate(training_teacher[-1], testing_t_eval)

print(np.allclose(output, output2))

# %%

import numpy as np
from estimators.volt_funcs import Volterra
import estimators.volt_funcs2 as Volterra2
from estimators.ngrc_funcs import NGRC
import estimators.ngrc_funcs2 as NGRC2
from estimators.sindy_funcs import SINDyPolynomialSTLSQ
import pysindy as ps
from datagen.data_generate import rk45
from utils.normalisation import normalise_arrays

def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)

h = 0.005
t_span = (0, 40)
slicing = int(h/h)

t_eval, data = rk45(lorenz, t_span, Z0, h, lor_args)
t_eval = t_eval[::slicing]
data = data[::slicing]

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

normalisation_output = normalise_arrays([training_input_orig, training_teacher_orig, testing_input_orig, testing_teacher_orig], norm_type=None)
training_input, training_teacher, testing_input, testing_teacher = normalisation_output[0]
shift, scale = normalisation_output[1], normalisation_output[2]

# Define input hyperparameters for Volterra
tau_coef = 0.2
ld_coef = 0.8
reg = 1e-10 

# Run new Volterra as a class
volt = Volterra(ld_coef, tau_coef, reg, washout)
pred_nonauto = volt.Train(training_input, training_teacher).Forecast(testing_input)
pred_auto = volt.PathContinue(training_teacher[-1], testing_teacher.shape[0])

# Run old Volterra functions
M = np.max([np.linalg.norm(z) for z in training_input])
tau = np.sqrt(1 / M**2) * tau_coef
ld = np.sqrt(1 - (tau**2) * (M**2)) * ld_coef
alpha_ols, alpha0_ols, K = Volterra2.Train(training_input, training_teacher, washout, ld, tau, reg, pinv=False)
pred_nonauto2 = Volterra2.Forecast(K, training_input, testing_input, alpha_ols, alpha0_ols, washout, ld, tau)  
pred_auto2 = Volterra2.ForecastAuto(K, training_input, training_teacher[-1], alpha_ols, alpha0_ols, washout, ntest, ld, tau)

# Check that the outputs match
print(np.allclose(pred_nonauto, pred_nonauto2))
print(np.allclose(pred_auto, pred_auto2))

# Define input hyperparameters for NGRC
ndim = 3
ndelay = 2
reg = 1e-4
deg = 2

# Run the new NGRC class
NGRC = NGRC(ndelay, deg, reg, washout)
NGRC.Train(training_input, training_teacher)
prediction = NGRC.PathContinue(training_teacher[-1], testing_teacher.shape[0])

# Run the old NGRC functions
training_input_ngrc, training_teacher_ngrc = training_input.T, training_teacher.T
W_out, X = NGRC2.Train(ndelay, deg, reg, training_input_ngrc, training_teacher_ngrc, washout)
prediction2 = NGRC2.Forecast(W_out, X, training_teacher[-1], deg, ntest)

print(np.allclose(prediction, prediction2.T))

# Define the hyperparameters for SINDy
threshold = 0.1
alpha = 0.5
deg = 2

# Run the new SINDy functions
SINDy = SINDyPolynomialSTLSQ(alpha, threshold, deg, h)
output = SINDy.Train(training_input, training_teacher).PathContinue(training_teacher[-1], testing_teacher.shape[0])

# Run the old SINDy functions
training = data[0:ntrain]
testing = data[ntrain:]
testing_t_eval = t_eval[ntrain:]
stlsq_optim = ps.STLSQ(threshold=threshold, alpha=alpha)
library = ps.PolynomialLibrary(degree=deg, include_interaction=True, interaction_only=False)
sindy_model = ps.SINDy(optimizer=stlsq_optim, feature_library=library)
sindy_model.fit(training, t=h)
output2 = sindy_model.simulate(training_teacher[-1], testing_t_eval)

print(np.allclose(output, output2))

# %%
