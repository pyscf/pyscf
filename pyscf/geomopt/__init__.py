from .grad import gen_grad_scanner
#from . import berny_solver as berny

def optimize(method, *args, **kwargs):
    from . import berny_solver
    return berny_solver.optimize(method, *args, **kwargs)
