import numpy as np

def calculate_mse(y_true, y_pred, shift=None, scale=None):

    """
    Calculate Mean Squared Error (MSE) between true and predicted values.
    If shift and scale are not None, then unshifts and unscales the data. 

    Parameters:
    - y_true: numpy array of true target values.
    - y_pred: numpy array of predicted target values.
    - shift: the shift that was implemented in the normalisation process
    - scale: the scale that was implemented in the normalisation process

    Returns:
    - mse: Mean Squared Error.
    """
    
    if shift is not None and scale is not None:
        # Destandardize the data if required
        y_true = y_true * (1/scale) + shift
        y_pred = y_pred * (1/scale) + shift

    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2)

    return mse

