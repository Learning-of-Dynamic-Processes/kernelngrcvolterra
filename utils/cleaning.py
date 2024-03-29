
from operator import itemgetter

def err_sort(filename):
    
    """
    Takes outcomes of cross validation and sorts it by the error values.
    
    Parameters
    ----------
    filename : str
        Name of file containing the cv output. Cv file should be lines where each line has the form
        (estimator_parameter_tuple, error_value).

    Returns
    -------
    sort_file_names : list
        List of the parameters sorted by the error associated to each parameter.
        Also writes to a new file named filename_sortedbyerr.txt.
    """
    
    # Pulls each line of file except for separator. Strips the line and converts it to a tuple
    separator_str = "-" * 40
    file_lines = []
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            if line == separator_str:
                continue
            file_lines.append(eval(line))
    
    # Isolate the errors for sorting
    err_vals = []
    for comb in file_lines:
        err_vals.append(comb[1])
    
    # Sort the errors and return the sorted errors and indices
    sort_indices, sort_err = zip(*sorted(enumerate(err_vals), key=itemgetter(1)))

    # Resorts the errors together with the combinations
    sort_file_lines = [file_lines[i] for i in sort_indices]
    
    # Rewrite new file name - takes the exact same name, appends with sortedbyerr
    new_filename = f"{filename[:-4]}_sortedbyerr.txt"

    # Write the sorted lines to a new file
    with open(new_filename, "w") as sorted_file:
        for sorted_line in sort_file_lines:
            sorted_file.write(f"{sorted_line}\n")
    
    return sort_file_lines
            
def param_sort(filename):
    
    """
    Takes outcomes of cross validation and sorts it by the parameter values. Sorts lexicographically. 
    
    Parameters
    ----------
    filename : str
        Name of file containing the cv output. Cv file should be lines where each line has the form
        (estimator_parameter_tuple, error_value).

    Returns
    -------
    sort_file_names : list
        List of the parameters and their errors sorted by the parameters in lexicographical order. 
        Also writes to a new file named filename_sortedbyparam.txt.
    """
    
    # Pulls each line of file except for separator. Strips the line and converts it to a tuple
    separator_str = "-" * 40
    file_lines = []
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            if line == separator_str:
                continue
            file_lines.append(eval(line))
    
    # Isolate the estimator parameters for sorting
    param_vals = []
    for comb in file_lines:
        param_vals.append(comb[0])
    
    # Sort the parameters lexicographically and return the sorted parameters and indices
    sort_indices, sort_param = zip(*sorted(enumerate(param_vals), key=itemgetter(1)))

    # Resorts the errors together with the combinations
    sort_file_lines = [file_lines[i] for i in sort_indices]
    
    # Rewrite new file name - takes the exact same name, appends with sortedbyparam
    new_filename = f"{filename[:-4]}_sortedbyparam.txt"

    # Write the sorted lines to a new file
    with open(new_filename, "w") as sorted_file:
        for sorted_line in sort_file_lines:
            sorted_file.write(f"{sorted_line}\n")
    
    return sort_file_lines
