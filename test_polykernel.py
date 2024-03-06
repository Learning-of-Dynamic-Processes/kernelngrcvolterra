# %% 

import numpy as np
from estimators.polykernel_funcs import PolynomialKernel

# Define a sample training data
data_len = 10
data = np.linspace(1, data_len, data_len).reshape((-1, 1))

# Define full data training and testing sizes
ndata  = len(data)
ntrain = 5
washout = 1
ntest = ndata - ntrain

# Construct training input and teacher, testing input and teacher
training_input = data[0:ntrain-1] 
training_teacher = data[1:ntrain]
testing_input = data[ntrain-1:ntrain+ntest-1]
testing_teacher = data[ntrain:ntrain+ntest]

# Define polynomial kernel hyperparameters
deg = 1
ndelays = 2
reg = 1e-4

# Train and forecast/pathcontinue using poly kernel
polykernel = PolynomialKernel(deg, ndelays, reg, washout)
polykernel.Train(training_input, training_teacher)
output = polykernel.Forecast(testing_input)
pathcontinue = polykernel.PathContinue(training_teacher[-1], testing_teacher.shape[0])
