import numpy as np

def iter_rk45(prev, t, h, f, fargs=None):
    
    """
    One step of forward of Runge-Kutta 45 method for ODEs. Passed into for rk45. 

    Parameters
    ----------
    prev : array_like
        Vector with the values from the previous step
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
        z2 = prev + (h/2)*f(t, z1)
        z3 = prev + (h/2)*f(t + 0.5*h, z2)
        z4 = prev + h*f(t + 0.5*h, z3)

        z = (h/6)*(f(t, z1) + 2*f(t + 0.5*h, z2) + 2*f(t + 0.5*h, z3) + f(t + h, z4))
        curr = prev + z
    
    else:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1, fargs)
        z3 = prev + (h/2)*f(t + 0.5*h, z2, fargs)
        z4 = prev + h*f(t + 0.5*h, z3, fargs)

        z = (h/6)*(f(t, z1, fargs) + 2*f(t + 0.5*h, z2, fargs) + 2*f(t + 0.5*h, z3, fargs) + f(t + h, z4, fargs))
        curr = prev + z
    
    return curr

def rk45(f, t_span, sol_init, h, fargs=None):
    
    """
    Runge-Kutta 45 for ODEs. 
    
    Parameters
    ----------
    f : callable
        Function that outputs the vector field of the ODE taking in inputs with format (t, z, fargs)
    t_span : tuple of ints
        (start, end) indicating the start and end time for numerical integration
    sol_init : array_like
        Solution to start integrating from
    h : int
        Time step needed for RK45 interation
    fargs : dict, optional 
        Arguments that are passed into f DDE to generate the vector field

    Returns
    -------
    t_eval : array_like
        Time steps at which RK45 solved the ODE
    solution : array_like
        Solution of the ODE with format (nsamples, ndim)
    """
    
    start = t_span[0]
    end = t_span[1]
    
    t_eval = np.arange(start, end+h, h)
    sol_len = len(t_eval)
    solution = [0] * len(t_eval)
    
    solution[0] = sol_init
    prev = sol_init
    
    for t_id in range(1, sol_len):
        t = t_eval[t_id-1]
        curr = iter_rk45(prev, t, h, f, fargs)
        solution[t_id] = curr
        prev = curr
    
    return t_eval, np.array(solution)
