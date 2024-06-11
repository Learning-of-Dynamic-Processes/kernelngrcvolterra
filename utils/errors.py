import numpy as np
from scipy.stats import wasserstein_distance
from scipy.signal import periodogram

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

def calculate_wasserstein1err(y_true, y_pred, shift=None, scale=None):
    
    """
    Calculate Wasserstein1 error between true and predicted values. 
    Orders over each dimension and sums total over dimensions. 
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


def calculate_specdensloss(y_true, y_pred, shift=None, scale=None):
    
    """
    Calculate difference between normalised spectral density of true and predicted values. 
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
    
    # Compute absolute difference in normalised spectral density for each dimension then sum
    error = 0
    for dim in range(ndim):
        psd_true = periodogram(y_true[:, dim], window="hann", scaling="spectrum")[1]
        psd_pred = periodogram(y_pred[:, dim], window="hann", scaling="spectrum")[1]
        error = error + np.sum(np.abs(psd_true - psd_pred))
    error = error * 1000
        
    return error