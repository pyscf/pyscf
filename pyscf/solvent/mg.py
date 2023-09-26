import numpy as np
from pyscf import lib

def v_cycle(mg, x0, b, mesh):
    # Pre-Smoothing
    x, r = mg.smoothing(x0, b, mesh);

    # Get coarser grid
    submesh = mg.coarsening(mesh) 

    # Restriction
    r_sub = mg.restriction(r, mesh, submesh);

    eps = np.zeros_like(r_sub);

    # stop recursion at smallest grid size, otherwise continue recursion
    if grid_small_enough(submesh):
        eps, r = mg.smoothing(eps, r_sub, submesh);
    else
        eps, r = v_cycle(mg, eps, r_sub, submesh);

    # Prolongation and Correction
    x = x + mg.prolongation(eps, mesh, submesh);

    # Post-Smoothing
    x, r = mg.smoothing(x, b, mesh);
    return x, r

def w_cycle(mg):
    pass

def f_cycle(mg):
    pass

def check_convergence(r, b, conv_tol):
    r_norm = np.sqrt(lib.vdot(r,r)) / np.sqrt(lib.vdot(b,b))
    if r_norm < conv_tol:
        return True
    else:
        return False

def kernel(mg, b, x0=None, mesh=None, cycle_fun=v_cycle,
           check_convergence=None, conv_tol=1e-5, max_cycle=100):
    assert callable(cycle_fun)
    assert callable(check_convergence)
    if x0 is None:
        x0 = np.zeros_like(b)
    if mesh is None:
        mesh = mg.mesh

    x = x0
    cnvged = False
    for i in range(max_cycle):
        x, r = cycle_fun(mg, x, b, mesh)
        if check_convergence(r, b, conv_tol):
            break
    return x, r

class MultiGrid(lib.StreamObject):
    def __init__(self):
        self.smoothing = None
        self.coarsening = None
        self.restriction = None
        self.prolongation = None
        self.check_convergence = check_convergence
        self.mesh = None
        self.cycle_type = 'V'
        self.max_cycle = 100
        self.conv_tol = 1e-5

    def sanity_check(self):
        logger.info(self, f"mesh = {self.mesh}")
        logger.info(self, f"cycle_type = {self.cycle_type}")
        logger.info(self, f"max_cycle = {self.max_cycle}")
        logger.info(self, f"conv_tol = {self.conv_tol}")
        if not callable(self.smoothing):
            logger.warn(self, f"smoothing function not callable: {self.smoothing}")
        if not callable(self.coarsening):
            logger.warn(self, f"coarsening function not callable: {self.coarsening}")
        if not callable(self.restriction):
            logger.warn(self, f"restriction function not callable: {self.restriction}")
        if not callable(self.prolongation):
            logger.warn(self, f"prolongation function not callable: {self.prolongation}")
        if not callable(self.check_convergence):
            logger.warn(self, f"check_convergence function not callable: {self.check_convergence}")

    def kernel(self, b, x0=None, mesh=None, cycle_type=None, cycle_fun=None,
               check_convergence=None, conv_tol=None, max_cycle=None, sanity_check=True):
        if sanity_check:
            self.sanity_check()
        if mesh is None:
            mesh = self.mesh
        if cycle_type is None:
            cycle_type = self.cycle_type
        if check_convergence is None:
            check_convergence = self.check_convergence
        if conv_tol is None:
            conv_tol = self.conv_tol
        if max_cycle is None:
            max_cycle = self.max_cycle

        if callable(cycle_fun):
            msg = (f"Ignoring the cycle_type argument: {cycle_type}, " +
                   f"since cycle_fun: {cycle_fun} is callable.")
            logger.info(self, msg)
        else:
            if cycle_type.upper() == 'V':
                cycle_fun = v_cycle
            else:
                raise NotImplementedError

        return kernel(self, b, x0=x0, mesh=mesh, cycle_fun=cycle_fun,
                      check_convergence=check_convergence,
                      conv_tol=conv_tol, max_cycle=max_cycle)
