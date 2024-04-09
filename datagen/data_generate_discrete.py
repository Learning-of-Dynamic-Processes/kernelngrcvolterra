
import numpy as np

def data_iterate(func, init_cond, nsteps, fargs=None):
    
    func_values = [init_cond]
    prev_step = init_cond
    for step in range(nsteps):
        curr_step = func(prev_step, fargs)
        func_values.append(curr_step)
        prev_step = curr_step
    
    return np.array(func_values).reshape((-1, 1))