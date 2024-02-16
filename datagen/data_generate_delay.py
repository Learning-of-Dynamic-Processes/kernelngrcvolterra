import numpy as np

def iter_rk45(prev, lag, t, h, f, fargs=None): 
    
    """
    One step of forward of Runge-Kutta 45 method for delay differential equations (DDEs). Passed into for dde_rk45. 

    Parameters
    ----------
    prev : array_like
        Vector with the values from the previous step
    lag : array_like
        Vector with the lag values needed to generate the vector field
    t : int
        Current time step for input into the DDE vector field function
    h : int
        Time step needed for RK45 interation
    f : callable
        Spits out the vector field given inputs which should be of the format (t, z, z_lag, fargs)
    fargs : dict, optional
        Arguments that are passed into f DDE to generate the vector field
    
    Returns
    -------
    curr : array_like
        One step forward of RK45 iteration 
    """
    
    if fargs == None: 
        z1 = prev
        z2 = prev + (h/2)*f(t, z1, lag)
        z3 = prev + (h/2)*f(t + 0.5*h, z2, lag)
        z4 = prev + h*f(t + 0.5*h, z3, lag)
        
        z = (h/6)*(f(t, z1, lag) + 2*f(t + 0.5*h, z2, lag) + 2*f(t + 0.5*h, z3, lag) + f(t + h, z4, lag))
        curr = prev + z
     
    else:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1, lag, fargs)
        z3 = prev + (h/2)*f(t + 0.5*h, z2, lag, fargs)
        z4 = prev + h*f(t + 0.5*h, z3, lag, fargs)
        
        z = (h/6)*(f(t, z1, lag, fargs) + 2*f(t + 0.5*h, z2, lag, fargs) + 2*f(t + 0.5*h, z3, lag, fargs) + f(t + h, z4, lag, fargs))
        curr = prev + z
        
    return curr


def dde_rk45(n_intervals, func_init, f, h, fargs=None):
    
    """
    Runge-Kutta 45 for delay differential equations
    
    Parameters
    ----------
    n_intervals : int
        Number of delay intervals to integrate over
    func_init : callable
        Function that gives the initial values for the first interval
    f : callable
        Spits out the vector field given inputs which should be of the format (t, z, z_lag, fargs)
    h : int
        Time step needed for RK45 interation
    f : callable
    fargs : dict, optional
        Arguments that are passed into f DDE to generate the vector field
        
    Returns
    -------
    t_eval : array_like
        Time steps at which RK45 solved the DDE at
    solution : array_like
        Solution flattened into a 1D array and reshaped into (nsamples, 1)
    """
    
    delay = fargs['delay']
    discretisation = int(delay / h)
    
    prev = np.array([ func_init(t) for t in range(0, discretisation) ])
    curr = np.zeros(shape=(discretisation, ))
    solution = np.zeros(shape=(n_intervals, discretisation))
    t_eval = []
    
    for interval in range(0, n_intervals):
        time = interval * delay
        curr[0] = iter_rk45(prev[discretisation-1], prev[0], time, h, f, fargs)
        t_eval.append(time)
        for step in range(1, discretisation):
            time = time + h
            curr[step] = iter_rk45(curr[step-1], prev[step], time, h, f, fargs)
            t_eval.append(time)
        solution[interval, :] = curr
        prev = curr
        
    return np.array(t_eval), solution.flatten().reshape((-1, 1))


