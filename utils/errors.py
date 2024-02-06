import numpy as np

def calculate_mse(y_true, y_pred, mean=None, std=None):

    """
    Calculate Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    - y_true: numpy array of true target values.
    - y_pred: numpy array of predicted target values.
    - mean: Mean of the target variable (optional).
    - std: Standard deviation of the target variable (optional).

    Returns:
    - mse: Mean Squared Error.
    """
    
    if mean is not None and std is not None:
        # Destandardize the data if required
        y_true = (y_true * std) + mean
        y_pred = (y_pred * std) + mean

    # Calculate MSE
    mse = np.mean((y_true - y_pred)**2)

    return mse