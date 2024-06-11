# %% 

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import perf_funcs as perf
import hickle as hkl
from numba import njit
from functools import partial
import time
import mat73
from scipy.stats import loguniform, wasserstein_distance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import KernelCenterer, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

# %%
#################### Scorers #########################
######################################################

def neg_calculate_nmse(y_true, y_pred):
    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2, axis=0)
    factor = np.mean((y_true)**2, axis=0)
    neg_nmse = -np.mean(mse / factor)
    return neg_nmse

def mean_wasserstein_dist(y_true, y_pred):
    was_dist=[]
    ndim=np.shape(y_true)[1]
    for i in range(ndim):
        was_dist = np.append(was_dist, wasserstein_distance(y_true[:,i], y_pred[:,i]))
    return np.mean(was_dist)

def neg_calculate_mse(y_true, y_pred):
    # Calculate MSE
    neg_mse = -np.mean((y_true - y_pred)**2)
    return neg_mse

def set_scorer(scorer='None'):
    if scorer=='neg_mse':
        my_scorer = {'neg_mse': make_scorer(neg_calculate_mse, greater_is_better=True)}  
        my_refit = 'neg_mse' 
        score_str = 'mean_test_neg_mse'
    elif scorer=='neg_nmse':
        my_scorer = {'neg_nmse': make_scorer(neg_calculate_nmse, greater_is_better=True)}  
        my_refit = 'neg_nmse' 
        score_str = 'mean_test_neg_nmse'
    elif scorer=='score':
        my_scorer = None
        my_refit = True
        score_str = 'mean_test_score'
    return my_scorer, my_refit, score_str

# %% 
################ Volterra Class ######################
######################################################

@njit
def _volt_gram_fast_njit(X, Y, omega, tau):
    nT_X, nT_Y = X.shape[0], Y.shape[0]
    # Compute once only instead of in the loop
    omega, tau = omega ** 2, tau ** 2
    # Initialize the gram matrix with ones
    Gram = np.ones((nT_X, nT_Y))
    tau_XY = 1 - X @ Y.T * tau
    
    # Compute the first row and column of the Gram matrix
    Gram[0, 0] += omega / (1 - omega) / tau_XY[0, 0]
    for i in range(1, nT_X):
        Gram[i, 0] += omega / (1 - omega) / tau_XY[i, 0]
    for i in range(1, nT_Y):
        Gram[0, i] += omega / (1 - omega) / tau_XY[0, i]

    # Compute the rest of the Gram matrix
    for i in range(1, nT_X):
        tau_XY_i = tau_XY[i]
        for j in range(1, nT_Y):
            Gram[i, j] += omega * Gram[i-1, j-1] / tau_XY_i[j]
    
    return Gram

def _volt_gram(X, Y, omega, tau):
    nT_X = np.shape(X)[0]
    nT_Y = np.shape(Y)[0]
    Gram = np.zeros((nT_X, nT_Y))
    Gram0 = 1/(1-omega**2)
    for i in range(nT_X):
        for j in range(nT_Y):
            if i==0 or j==0:
                Gram[i, j] = 1 + omega**2 * Gram0/(1-(tau**2)*(np.dot(X[i,:], Y[j,:])))
            else:
                Gram[i, j] = 1 + omega**2 * Gram[i-1,j-1]/(1-(tau**2)*(np.dot(X[i,:], Y[j,:])))
    return Gram

def Volterra_kernel_callable(X, Y, omega, tau, nwashout):
    # print(tau)
    # print(omega)
    volt_K = _volt_gram_fast_njit(X, Y, omega, tau)
    return volt_K

class VolterraKernel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1, omega=0.1, tau=0.1, nwashout=0, ifscale_kernel=False, ifscale_y=False, ifcenter_kernel=False):
        self.alpha = alpha
        self.omega = omega
        self.tau = tau
        self.nwashout = nwashout
        self.Volterra_kernel = partial(Volterra_kernel_callable, omega=self.omega, tau=self.tau, nwashout=self.nwashout)
        self.regressor = KernelRidge(kernel="precomputed", alpha=self.alpha)
        self.ifscale_kernel = ifscale_kernel
        self.ifscale_y = ifscale_y
        self.ifcenter_kernel = ifcenter_kernel

    def fit(self, X, y):
        # Transform to obtain Volterra kernel and fit kernel ridge regression
        K_fit = self.Volterra_kernel(X, X)
        K_fit = K_fit[self.nwashout:, self.nwashout:]
        if self.ifcenter_kernel:
            centerer = KernelCenterer()
            K_fit = centerer.fit_transform(K_fit)
            self.centerer = centerer
            
        self.X_fit_ = X.copy()
        # Rescale kernel if needed
        if self.ifscale_kernel:
            self.scaler_K = StandardScaler().fit(K_fit)
            K_fit = self.scaler_K.transform(K_fit)

        if self.ifscale_y:
            self.scaler_y = StandardScaler().fit(y)
            y = self.scaler_y.transform(y)

        self.K_fit_ = K_fit
        #print(self.Volterra_kernel)
        self.regressor.fit(self.K_fit_, y[self.nwashout:,:])
        return self
    
    def predict(self, X):
        # Transform to obtain Volterra kernel and predict
        K_test = self.Volterra_kernel(X, self.X_fit_) # Use X_fit, not K_fit
        K_test = K_test[:, self.nwashout:]
        if self.ifcenter_kernel:
            centerer = self.centerer
            K_test = centerer.transform(K_test)
            
        if self.ifscale_kernel:
            K_test = self.scaler_K.transform(K_test)

        if self.ifscale_y:
            return self.scaler_y.inverse_transform(self.regressor.predict(K_test))
            
        return self.regressor.predict(K_test)
    
    # Make alpha parameter 'cv-able'
    def set_params(self, **params):
        for k, v in params.items():
            if k == "nwashout":
                self.nwashout = v
            if k == "omega":
                self.omega = v
            elif k == "tau":
                self.tau = v
            elif k == "alpha":
                self.alpha=v
                self.regressor.alpha = self.alpha    
            else:
                setattr(self.regressor, k, v)
        if self.omega > np.sqrt(1 - (self.tau**2)):
            print(self.tau)
            print(self.omega)
            self.omega = np.sqrt(1 - (self.tau**2))*0.99
            print('became:')
            print(self.omega)
        self.Volterra_kernel = partial(Volterra_kernel_callable, 
                                       omega=self.omega, 
                                       tau=self.tau, 
                                       nwashout=self.nwashout)
        self.regressor = KernelRidge(kernel="precomputed", alpha = self.alpha)
        #print(self.Volterra_kernel)
        
        return self

# %% 
################ Load data ###########################
######################################################

matstruct_contents = mat73.loadmat("./datagen/BEKK_d15_data.mat")
returns = matstruct_contents['data_sim']
epsilons = matstruct_contents['exact_epsilons']
Ht_sim_vech = matstruct_contents['Ht_sim_vech']

data_in = epsilons
#data_out = returns
data_out = Ht_sim_vech

ndim = data_in.shape[1]
nT = data_in.shape[0]
ndim_output = data_out.shape[1]

if nT>3760:
    nT = 3760

x = data_in[0:nT-1,:]
#y = data_out[0:nT-1,:]**2*1000
y = data_out[1:nT,0:ndim_output]*1000
t_dispay = 300

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    shuffle=False, 
    random_state=1)

scaler_y = StandardScaler().fit(y_train)
y_train_demeaned = scaler_y.transform(y_train)

train_len = np.shape(x_train)[0]
test_len = np.shape(x_test)[0]
total_len = test_len + train_len

M = np.max([np.linalg.norm(z) for z in x])
x_train /= M
x_test /= M

set_my_scorer="score"
my_scorer, my_refit, score_str = set_scorer(set_my_scorer)
     
n_jobs = 100

ifsave=True
ifRandomSearch=False

# %% 
# Check that the Volterra kernel gives the same results on the testing set for some set of hyperparameters
# With modified Gram matrix using my Gram matrix code
test_volt_kernel1 = VolterraKernel()
test_volt_kernel1.fit(x_train, y_train_demeaned)
test_prediction1 = test_volt_kernel1.predict(x_test)

# %% 
# Check for my Volterra kernel prediction set
from estimators.volt_funcs_temp import Volterra
test_volt_kernel2 = Volterra(ld_coef=0.1, tau_coef=0.1, reg=1, washout=0)   # Follow the same defaults
test_volt_kernel2.Train(x_train, y_train_demeaned)
test_prediction2 = test_volt_kernel2.Forecast(x_test)

# %% 

print(np.allclose(test_prediction1, test_prediction2))
print(test_prediction1)
print(test_prediction2)

# %% 
# Check to make sure that the Gram matrix is the same
print(np.allclose(test_volt_kernel1.K_fit_, test_volt_kernel2.Gram))
print(test_volt_kernel1.K_fit_)
print(test_volt_kernel2.Gram)

# %% 
# Check that the error functions all run correctly 

from utils.errors import calculate_mse, calculate_nmse, calculate_wasserstein1err

print(neg_calculate_mse(y_test, test_prediction1))
print(calculate_mse(y_test, test_prediction2))

print(neg_calculate_nmse(y_test, test_prediction1))
print(calculate_nmse(y_test, test_prediction2))

print(mean_wasserstein_dist(y_test, test_prediction1))
print(calculate_wasserstein1err(y_test, test_prediction2))

# %% 
# Check the CV training split dataset so that they are the same 

cv_datasets1 = []
for train_id, validation_id in TimeSeriesSplit(n_splits=5).split(x_train, y_train):
    cv_datasets1.append((x_train[train_id], y_train[train_id], x_train[validation_id], y_train[validation_id]))

from utils.crossvalidation import CrossValidate
CV = CrossValidate(validation_parameters=[501, 501, 501], validation_type="expanding", 
                   manage_remainder=True, task="Forecast")
cv_split_output = CV.split_data_to_folds(x_train, y_train)
cv_datasets2 = []
for split_id in range(len(cv_split_output)):
    cv_datasets2.append(cv_split_output[split_id][0])

for split_id in range(len(cv_datasets1)):
    for fold in range(0, 3):
        print(np.allclose(cv_datasets1[split_id][fold], cv_datasets2[split_id][fold]))

# %% 
# Check that the CV generates the same results

from utils.crossvalidation import CrossValidate

if __name__ == "__main__":
    
    CV = CrossValidate(validation_parameters=[501, 501, 501], validation_type="expanding", 
                    manage_remainder=True, task="Forecast")
    cv_datasets =  CV.split_data_to_folds(x_train, y_train)
    ld_range = [0.01]
    tau_range = [0.1]
    reg_range = [0.0]
    param_ranges = [ld_range, tau_range, reg_range]
    param_add = [0]
    min_error, best_parameters = CV.crossvalidate(Volterra, cv_datasets, param_ranges, param_add, 
                                                    num_processes=1, chunksize=1) 


# %% 
################ VOLTERRA kernel #####################
######################################################

est_string = 'Volterra'

loc_log_alpha = -5
scale_log_alpha = 1
loc_log_omega = -2
scale_log_omega = 0
loc_log_tau = -1
scale_log_tau = -0.002
num_params = 3
n_iter = 500
num_tau = 100
tau_set = np.logspace(loc_log_tau, scale_log_tau, num_tau)
best_score_value=-1e8

# Start the timer
start_time = time.time()
if ifRandomSearch:
    for tau in tau_set:
        kr = VolterraKernel(ifscale_y=False, tau=tau)
        pipe_kr_Volterra = Pipeline(
            #[('scl', StandardScaler()),
            [('kr_est', kr)]
        )
        #print(np.sqrt(1 - (tau**2))*0.99)
        grid_param_Volterra = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                                kr_est__omega=loguniform(a=10**loc_log_omega, b=np.sqrt(1 - (tau**2))*0.999))
        krCV_Volterra = RandomizedSearchCV(
            estimator=pipe_kr_Volterra,
            param_distributions=grid_param_Volterra, 
            cv=TimeSeriesSplit(n_splits=5), 
            n_jobs=n_jobs,
            scoring=my_scorer,
            refit=my_refit,
            random_state=0,
            n_iter = n_iter
        )
        krCV_Volterra.fit(x_train, y_train_demeaned)
        if krCV_Volterra.best_score_>best_score_value:
            print(krCV_Volterra.best_score_)
            # print(krCV_Volterra.best_estimator_)
            # print(tau)
            #print(pd.DataFrame(krCV_Volterra.cv_results_).sort_values(by=score_str))
            best_score_value = krCV_Volterra.best_score_
            best_krCV_Volterra= krCV_Volterra
            best_tau = tau
else:
    for tau in tau_set:
        kr = VolterraKernel(ifscale_y=False, tau=tau)
        pipe_kr_Volterra = Pipeline(
            #[('scl', StandardScaler()),
            [('kr_est', kr)]
        )
        print(np.sqrt(1 - (tau**2))*0.99)
        grid_param_Volterra = {
            'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int((n_iter)**(num_params**-1))-1)),
            'kr_est__omega': np.logspace( loc_log_omega, np.log10(np.sqrt(1 - (tau**2))*0.99), int((n_iter)**(num_params**-1)))
        }
        krCV_Volterra = GridSearchCV(
            estimator=pipe_kr_Volterra,
            param_grid=grid_param_Volterra, 
            cv=TimeSeriesSplit(n_splits=5), 
            n_jobs=n_jobs,
            scoring=my_scorer,
            refit=my_refit
        )
        krCV_Volterra.fit(x_train, y_train_demeaned)
        print(krCV_Volterra.best_score_)
        if krCV_Volterra.best_score_>best_score_value:
            print(krCV_Volterra.best_score_)
            print(krCV_Volterra.best_estimator_)
            print(tau)
            #print(pd.DataFrame(krCV_Volterra.cv_results_).sort_values(by=score_str))
            best_score_value = krCV_Volterra.best_score_
            best_krCV_Volterra= krCV_Volterra
            best_tau = tau
    
    
krCV_Volterra=best_krCV_Volterra       
print(krCV_Volterra.best_estimator_)       
print(best_tau)                      

#krCV_Volterra.fit(x_train, y_train_demeaned)

# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = (end_time - start_time)/len(tau_set)
print("SearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_Volterra.cv_results_[score_str]
print(pd.DataFrame(krCV_Volterra.cv_results_).sort_values(by=score_str))
pred_y_test_Volterra=krCV_Volterra.predict(x_test)
pred_y_test_Volterra = scaler_y.inverse_transform(pred_y_test_Volterra)

nmsekr_Volterra = perf.calculate_nmse(y_test, pred_y_test_Volterra)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_Volterra)
plt.plot(pred_y_test_Volterra[0:t_dispay,0])
plt.plot(y_test[0:t_dispay,0])
plt.savefig("bekk_volterra.pdf")
plt.close()

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([ifRandomSearch, krCV_Volterra.best_estimator_, krCV_Volterra.best_params_,
                     krCV_Volterra.best_score_,
                    nmsekr_Volterra, elapsed_time, pred_y_test_Volterra, y_test], file_path)
    print("The variable 'data' has been saved successfully.")
print(krCV_Volterra.best_params_)
print(krCV_Volterra.best_score_)
print('Volerra done')

# %% 
############### Polynomial kernel ###################
#####################################################

est_string = 'poly'
kr = KernelRidge(kernel="poly")
pipe_kr_poly = Pipeline([('scl', StandardScaler()),
        ('kr_est', kr)])

loc_log_alpha = -3
scale_log_alpha = 0
loc_log_gamma = -4
scale_log_gamma = 0
loc_log_coef0 = -4
scale_log_coef0 = 0
low_degree = 2
high_degree = 4
num_params = 3
num_degree = 2
n_iter = 56000

if ifRandomSearch:
    grid_param_kr_poly = dict(kr_est__alpha=loguniform(a=10**loc_log_alpha, b=10**scale_log_alpha),
                             kr_est__gamma=loguniform(a=10**loc_log_gamma, b=10**scale_log_gamma),
                             kr_est__coef0=loguniform(a=10**loc_log_coef0, b=10**scale_log_coef0),
                             kr_est__degree=[2,3])#randint(low=low_degree, high=high_degree))
    krCV_poly = RandomizedSearchCV(
        estimator=pipe_kr_poly,
        param_distributions=grid_param_kr_poly, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit,
        random_state=0,
        n_iter = n_iter
    )
else:
    grid_param_kr_poly = {
        'kr_est__alpha': np.append(0, np.logspace(loc_log_alpha, scale_log_alpha, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__gamma': np.append(1, np.logspace(loc_log_gamma, scale_log_gamma, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__coef0': np.append(0, np.logspace(loc_log_coef0, scale_log_coef0, int((n_iter/num_degree)**(num_params**-1))-1)),
        'kr_est__degree': [2,3]
    }
    krCV_poly = GridSearchCV(
        estimator=pipe_kr_poly,
        param_grid=grid_param_kr_poly, 
        cv=TimeSeriesSplit(n_splits=5), 
        n_jobs=n_jobs,
        scoring=my_scorer,
        refit=my_refit
    )

# Start the timer
start_time = time.time()
krCV_poly.fit(x_train, y_train_demeaned)
# Stop the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print("RandomizedSearchCV took {:.2f} seconds.".format(elapsed_time))

score = krCV_poly.cv_results_[score_str]
#print(pd.DataFrame(krCV_poly.cv_results_).sort_values(by=score_str))
pred_y_test_poly=krCV_poly.predict(x_test)
pred_y_test_poly = scaler_y.inverse_transform(pred_y_test_poly)

nmsekr_poly = perf.calculate_nmse(y_test, pred_y_test_poly)
# Print testing normalized mean squared error
print("Testing NMSE: ", nmsekr_poly)
plt.plot(pred_y_test_poly[0:t_dispay,0])
plt.plot(y_test[0:t_dispay,0])
plt.savefig("bekk_polynomial.pdf")
plt.clear()
plt.close()

time_string = 'mean_'+ set_my_scorer+'_time'
mean_fit_time= krCV_poly.cv_results_['mean_fit_time']
mean_est_time= krCV_poly.cv_results_[time_string]
n_splits  = krCV_poly.n_splits_ #number of splits of training data
n_iter = pd.DataFrame(krCV_poly.cv_results_).shape[0] #Iterations per split

print(np.mean(mean_fit_time + mean_est_time) * n_splits * n_iter)

if ifsave:
    # Saving Variables
    file_path = 'data_score_' + str(ndim)+'d_'+set_my_scorer+ '_'+ est_string+ '_'+ '.pickle'

    # write variables to filename
    hkl.dump([ifRandomSearch, elapsed_time, krCV_poly.best_estimator_, krCV_poly.best_params_,
                     krCV_poly.best_score_,
                    nmsekr_poly, pred_y_test_poly, y_test], file_path)
    print("The variable 'data' has been saved successfully.")

print(krCV_poly.best_params_)
print(krCV_poly.best_score_)
print('poly done')