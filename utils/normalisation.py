
import numpy as np

def normalise_arrays(arrays, norm_type=None, MinMax_range=(0, 1)): 
    
    """Function that takes in a list of numpy arrays then takes the data from the array at index 0 
    and normalises the datasets according to this data. Meant to use so that all testing and training data
    is normalised with respect to the training input (prevents data leakage).
    
    Parameters:
        - arrays: [arr1, arr2, ...]. Takes in a list of numpy arrays. Each numpy array must have the shape (nsamples, nfeatures).
          The array that you want to normalise with respect to must be the first array. 
        - norm_type: {"NormStd", "MinMax", "ScaleL2", "ScaleL2Shift}. Normalisation method to choose. Defaults to no normalisation.
        - MinMax_range: (desired_min, desired_max). Only kicks in if norm_type is "MinMax".

    Raises:
        NotImplementedError: Is raised if the normalisation method provided does not match one of the norm_types available.

    Returns:
        arrays_out: the list of arrays after normalisation
    """
    
    arrays_out = []
    
    if norm_type == "NormStd":
        # Normalises so that data of the first array is centered at 0 and standard deviation is 1
        mean0 = np.mean(arrays[0], axis=0)   
        std0 = np.std(arrays[0], axis=0)
        for array in arrays:
            array_out = (1/std0) * (array - mean0)
            arrays_out.append(array_out)
            
    elif norm_type == "MinMax":
        # Normalises so that data of first array lies between the given MinMax_range
        desired_min = MinMax_range[0]
        desired_max = MinMax_range[1]
        minimum0 = np.min(array[0], axis=0)
        maximum0 = np.max(array[0], axis=0)
        for array in arrays:
            array_std = (array - minimum0) / (maximum0 - minimum0)
            array_out = array_std * (desired_max - desired_min) + desired_min
            arrays_out.append(array_out)
        
    elif norm_type == "ScaleL2":
        # Normalises without shifting so that the data of the first array has norm 1
        max_l2_norm0 = np.max([np.linalg.norm(z) for z in array[0]])
        for array in arrays:
            array_out = (1/max_l2_norm0) * array
            arrays_out.append(array)
    
    elif norm_type == "ScaleL2Shift":
        # Normalising with shifting so that the data of the first array has norm 1 and mean 0
        mean0 = np.mean(array[0], axis=0)
        array0_shifted = array[0] - mean0
        max_l2_norm0_shifted = np.max([np.linalg.norm(z) for z in array0_shifted[0]])
        for array in arrays:
            array_out = (1/max_l2_norm0_shifted) * (array - mean0)
        
    elif norm_type is None:
        # Does not normalise the arrays at all
        for array in arrays:
            array_out.append(array)
        
    else:
        raise NotImplementedError("Normalisation method is not available")
    
    return arrays_out

