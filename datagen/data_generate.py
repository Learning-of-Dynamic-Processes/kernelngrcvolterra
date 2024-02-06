import numpy as np

def iter_rk45(prev, t, h, f, fargs=None):
    
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
    start = t_span[0]
    end = t_span[1]
    
    t_eval = np.arange(start, end+h, h)
    sol_len = len(t_eval)
    solution = [0] * len(t_eval)
    
    solution[0] = sol_init
    prev = sol_init
    
    for t_id in range(1, sol_len):
        t = t_eval[t_id]
        curr = iter_rk45(prev, t, h, f, fargs)
        solution[t_id] = curr
        prev = curr
    
    return t_eval, np.array(solution)
