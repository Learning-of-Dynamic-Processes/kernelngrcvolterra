
# Store discrete systems to test on here
# Functions should take arguments (tuple of prev variables, tuple of additional arguments)

def logistic_map(prev, r):
    return r * prev * (1-prev)