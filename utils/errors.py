import numpy as np
from scipy.stats import wasserstein_distance

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
        y_true = y_true * (1/scale) + shift
        y_pred = y_pred * (1/scale) + shift
    
    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2)

    return mse

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
        y_true = y_true * (1/scale) + shift
        y_pred = y_pred * (1/scale) + shift
    
    # Infer the dimension of the data
    ndim = y_true.shape[1]
    
    # Compute wasserstein distance for each dimension
    error = 0
    for dim in range(ndim):
        dim_error = wasserstein_distance(y_true[:, dim], y_pred[:, dim])
        error = error + dim_error 
    
    return error
