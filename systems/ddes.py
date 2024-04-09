
# Store dde systems to test on here
# Functions should take arguments (t, tuple of variables, tuple of lag_vars, additional arguments)

def mackeyglass(t, z, z_lag, mg_args):
    
    a = mg_args['a']
    b = mg_args['b']
    n = mg_args['n']
    
    return (a * z_lag) / (1 + z_lag**n) - b*z
