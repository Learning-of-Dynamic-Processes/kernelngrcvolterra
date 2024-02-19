
import numpy as np

def normalise_arrays(arrays, norm_type=None, MinMax_range=(0, 1)): 
    
    """
    Function that takes in a list of numpy arrays then takes the data from the array at index 0 
    and normalises the datasets according to this data. Meant to use so that all testing and training data
    is normalised with respect to the training input (prevents data leakage).
    
    Parameters
    ----------
    arrays : list of array_like
        [arr1, arr2, ...]. Takes in a list of numpy arrays. Each numpy array must have the shape (nsamples, nfeatures).
        The array that you want to normalise with respect to must be the first array. 
    norm_type : str
        Normalisation method to choose. Options: {"NormStd", "MinMax", "ScaleL2", "ScaleL2Shift, None}, default None.
    MinMax_range : (float, float)
        (desired_min, desired_max). Only kicks in if norm_type is "MinMax".

    Raises
    ------
    NotImplementedError: Is raised if the normalisation method provided does not match one of the norm_types available.

    Returns
    -------
    arrays_out : list of array_like
        The list of arrays after normalisation in the same order
    """
    
    # Function that takes the arrays given and shifts and scales it by a given amount
    def ShiftScale(arrays, shift, scale):
        arrays_out = []
        for array in arrays:
            array_out = scale * (array - shift)
            arrays_out.append(array_out)
        return arrays_out

    if norm_type == "NormStd":
        # Normalises so that data of the first array is centered at 0 and standard deviation is 1
        mean0 = np.mean(arrays[0], axis=0)   
        std0 = np.std(arrays[0], axis=0)
        shift = mean0
        scale = (1/std0)
        arrays_out = ShiftScale(arrays, shift, scale)
    
    elif norm_type == "MinMax":
        # Normalises so that data of first array lies between the given MinMax_range
        des_min = MinMax_range[0]
        des_max = MinMax_range[1]
        min0 = np.min(arrays[0], axis=0)
        max0 = np.max(arrays[0], axis=0)
        shift = min0 - (des_min*(max0-min0)) / (des_max-des_min)
        scale = (des_max - des_min) / (max0 - min0)
        arrays_out = ShiftScale(arrays, shift, scale)
        
    elif norm_type == "ScaleL2":
        # Normalises without shifting so that the data of the first array has norm 1
        max_l2_norm0 = np.max([np.linalg.norm(z) for z in arrays[0]])
        shift = 0
        scale = 1/max_l2_norm0
        arrays_out = ShiftScale(arrays, shift, scale)
    
    elif norm_type == "ScaleL2Shift":
        # Normalising with shifting so that the data of the first array has norm 1 and mean 0
        mean0 = np.mean(arrays[0], axis=0)
        array0_shifted = arrays[0] - mean0
        max_l2_norm0_shifted = np.max([np.linalg.norm(z) for z in array0_shifted])
        shift = mean0
        scale = 1/max_l2_norm0_shifted
        arrays_out = ShiftScale(arrays, shift, scale)
        
    elif norm_type is None:
        # Does not normalise the arrays at all
        shift = 0
        scale = 1
        arrays_out = ShiftScale(arrays, shift, scale)
        
    else:
        raise NotImplementedError("Normalisation method is not available")
    
    return arrays_out, shift, scale

