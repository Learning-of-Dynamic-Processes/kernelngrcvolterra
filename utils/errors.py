import numpy as np
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from scipy.signal import periodogram, welch
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def calculate_mse(y_true, y_pred, shift=None, scale=None):

    """
    Calculate Mean Squared Error (MSE) between true and predicted values.
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    mse : float
        Mean Squared Error.
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
    
    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2)
    
    return mse

def calculate_nmse(y_true, y_pred, shift=None, scale=None):
    
    """
    Calculate normalised MSE between true and predicted values
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    nmse : float
        Normalised mean square error. 
    """
  
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
    
    # Compute the nmse
    mse = np.mean((y_true - y_pred)**2, axis=0)
    factor = np.mean((y_true)**2, axis=0)
    nmse = np.mean(mse / factor)

    return nmse

def calculate_mae(y_true, y_pred, shift=None, scale=None):
    
    """
    Compute mean absolute error (MAE) between true and predicted values. 
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    mae : float
        Mean Absolute Error.
    """
        
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
        
    return np.mean(np.abs(y_true - y_pred))

def calculate_mdae_err(y_true, y_pred, shift=None, scale=None):
    
    """
    Computes median absolute error (MdAE). Wrapper function for sklearn.metrics.median_absolute_error.
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    MdAE : float
        Median Absolute Error.
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
    
    return median_absolute_error(y_true, y_pred)

def calculate_r2_err(y_true, y_pred, shift=None, scale=None):
    
    """
    Computes R2 score. Wrapper function for sklearn.metrics.r2_score.
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    r2_score : float
       R2 score.
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
        
    return r2_score(y_true, y_pred)    

def calculate_mape_err(y_true, y_pred, shift=None, scale=None):
    
    """
    Computes mean absolute percentage error score. Wrapper function for sklearn.metrics.mean_absolute_percentage_error.
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    mape_score : float
       Mean absolute percentage error.
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
        
    return mean_absolute_percentage_error(y_true, y_pred)



# Distribution error metrics
def calculate_wasserstein1err(y_true, y_pred, shift=None, scale=None):
    
    """
    Calculate Wasserstein1 error between true and predicted values. 
    Orders over each dimension and sums total over dimensions. 
    Non-mathematical implementation to handle multiple dimensions without using linear programming. 
    If one dimensional, should coincide with Wasserstein-1 distance up to the 15th decimal place,
    possibly due to floating point precision errors(?). 
    For actual Wasserstein distance over multiple dimensions, use calculate_wasserstein1_nd_err.
    If shift and scale are not None, then unshifts and unscales the data. 
    
    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    error : float
        Wasserstein1 error summed over all dimensions. 
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
    
    # Infer the dimension of the data
    ndim = y_true.shape[1]
    
    # Compute wasserstein distance for each dimension then sum
    error = 0
    for dim in range(ndim):
        dim_error = wasserstein_distance(y_true[:, dim], y_pred[:, dim])
        error = error + dim_error 
    
    return (1/ndim) * error

    
def calculate_wasserstein1_nd_err(y_true, y_pred, shift=None, scale=None):
    
    """
    Compute Wasserstein1 distance for n-dimensions. Wrapper function for sklearn.stats.wasserstein_distance_nd. 
    More correct version of Wasserstein distance for multiple dimensions that uses linear programming. 
    If working with 1d data, then use the calculate_wasserstein1_err function is much faster. 
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    wasserstein_distance_nd : float
        Wasserstein distance in n dimensions.
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
        
    return wasserstein_distance_nd(y_true, y_pred)
    
    
    
# Climate error metrics     
def calculate_specdens_welch_err(y_true, y_pred, shift=None, scale=None, fs=1, nperseg=256, stop=None, ifPlot=False, figname=None):
    
    """
    Calculate difference between normalised spectral density of true and predicted values using Welch's method (like Wilkner et. al.). 
    Computes spectral density over each dimension, normalises it then takes absolute difference and sums. 
    If shift and scale are not None, then unshifts and unscales the data. 
    
    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    error : float
        Absolute difference of normalised spectral density summed over all dimensions. 
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
        
    # Infer the dimension of the data
    ndim = y_true.shape[1]
    
    # Set stop to length of data if None
    if stop is None:
        stop = y_true.shape[0]
    
    # Compute absolute difference in normalised spectral density for each dimension then sum
    error = 0
    for dim in range(ndim):
        psd_true = welch(y_true[:, dim], fs=fs, nperseg=nperseg, window="hann", scaling="spectrum")[1] 
        psd_pred = welch(y_pred[:, dim], fs=fs, nperseg=nperseg, window="hann", scaling="spectrum")[1] 
        error = error + np.sum((np.abs(psd_true[0:stop] - psd_pred[0:stop]))/psd_true[0:stop])
        if ifPlot is True:
            plt.plot(psd_true[0:stop], label=f"PSD True (Dimension 84)") #{dim})")
            plt.plot(psd_pred[0:stop], label=f"PSD Pred (Dimension 117)", linestyle="dashed") #dim})", linestyle="dashed")
    if ifPlot is True:
        plt.legend()
        if figname is not None:
            plt.savefig(figname)
        plt.show()
        plt.close()
    return error


def calculate_specdens_periodogram_err(y_true, y_pred, shift=None, scale=None, fs=1, stop=None, ifPlot=False, figname=None):
    
    """
    Calculate difference between normalised spectral density of true and predicted values using periodogram method. 
    Computes spectral density over each dimension, normalises it then takes absolute difference and sums. 
    If shift and scale are not None, then unshifts and unscales the data. 
    
    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    error : float
        Absolute difference of normalised spectral density summed over all dimensions. 
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
        
    # Infer the dimension of the data
    ndim = y_true.shape[1]
    
    # Set stop to length of data if None
    if stop is None:
        stop = y_true.shape[0]
    
    # Compute absolute difference in normalised spectral density for each dimension then sum
    error = 0
    for dim in range(ndim):
        psd_true = periodogram(y_true[:, dim], fs=fs, window="hann", scaling="spectrum")[1]
        psd_pred = periodogram(y_pred[:, dim], fs=fs, window="hann", scaling="spectrum")[1]
        error = error + np.sum((np.abs(psd_true[0:stop] - psd_pred[0:stop]))/psd_true[0:stop])
        if ifPlot is True:
            plt.plot(psd_true[0:stop], label=f"PSD True (Dimension {dim})")
            plt.plot(psd_pred[0:stop], label=f"PSD Pred (Dimension {dim})", linestyle="dashed")
    if ifPlot is True:
        plt.legend()
        if figname is not None:
            plt.savefig(figname)
        plt.show()
        plt.close()
    return error


# Valid prediction time
def valid_pred_time(y_true, y_pred, shift=None, scale=None, epsilon=0.2):
    
    """
    Min time in which the R2-norm to deviates by epsilon from the true values. 

    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters
    ----------
    y_true : array_like
        Numpy array of true target values.
    y_pred : array_like
        Numpy array of predicted target values.
    shift : float, optional 
        The shift that was implemented in the normalisation process. (default: None).
    scale : float, optional 
        The scale that was implemented in the normalisation process. (default: None).

    Returns
    -------
    valid_pred_time : float
        Valid prediction time. 
    """
    
    # Destandardize the data if required
    if shift is not None and scale is not None:
        y_true = y_true * (1/scale) - shift
        y_pred = y_pred * (1/scale) - shift
    
    valid_pred_time = 0
    for t in range(1, len(y_true)+1):
        err_t = np.linalg.norm(y_true[t, :] - y_pred[t, :])/np.linalg.norm(y_true[t, :])
        if err_t >= epsilon:
            valid_pred_time = t
            break
        
    return valid_pred_time