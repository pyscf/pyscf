'''
Self-consistente continuum solvation model

Reference:
1. J. Chem. Phys. 136, 064102 (2012); https://doi.org/10.1063/1.3676407
2. J. Chem. Phys. 144, 014103 (2016); https://doi.org/10.1063/1.4939125
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.tools import write_cube

libpbc = lib.load_library('libpbc')

def _get_eps(rho_elec, rho_aux, rho_min, rho_max, eps0):
    if rho_aux is not None:
        rho_elec = rho_elec + rho_aux
    ng = rho_elec.size
    eps = numpy.empty_like(rho_elec, order='C', dtype=float)
    deps_intermediate = numpy.empty_like(rho_elec, order='C', dtype=float)
    fun = getattr(libpbc, 'get_eps')
    fun(eps.ctypes.data_as(ctypes.c_void_p),
        deps_intermediate.ctypes.data_as(ctypes.c_void_p),
        rho_elec.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(rho_min), ctypes.c_double(rho_max),
        ctypes.c_double(eps0), ctypes.c_size_t(ng))
    return eps, deps_intermediate

def _get_log_eps_gradient(eps, Gv, mesh, method='FFT'):
    log_eps = numpy.log(eps)
    if method.upper() == 'FFT':
        out = tools.gradient_by_fft(log_eps, Gv, mesh)
    elif method.upper() == 'FDIFF':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return out

def _get_log_eps1_fdiff(cell, eps, rhoR, rho_core, rho_min, rho_max, mesh):
    if rho_core is not None:
        rhoR += rho_core
    a = cell.lattice_vectors()
    nx, ny, nz = mesh
    hx = a[0,0] / nx
    hy = a[1,1] / ny
    hz = a[2,2] / nz
    log_eps = numpy.asarray(numpy.log(eps).reshape(*mesh), order='C')
    out = numpy.empty([3,*mesh], order='C', dtype=float)
    fun = getattr(libpbc, 'fdiff_gradient')
    fun(out.ctypes.data_as(ctypes.c_void_p),
        log_eps.ctypes.data_as(ctypes.c_void_p),
        rhoR.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(rho_min), ctypes.c_double(rho_max),
        ctypes.c_size_t(nx), ctypes.c_size_t(ny), ctypes.c_size_t(nz),
        ctypes.c_double(hx), ctypes.c_double(hy), ctypes.c_double(hz))
    return out.reshape(3,-1)

def _get_deps_drho(eps, deps_intermediate):
    return lib.multiply(eps, deps_intermediate)

def _pcg(sccs, rho_solute, eps, coulG=None, Gv=None, mesh=None,
         conv_tol=1e-5, max_cycle=20):
    cell = sccs.cell
    if mesh is None:
        mesh = sccs.mesh
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if Gv is None:
        Gv = cell.get_Gv(mesh=mesh)

    sqrt_eps = numpy.sqrt(eps)
    q = lib.multiply(sqrt_eps, tools.laplacian_by_fft(sqrt_eps, Gv, mesh))
    rho_tot = rho_solute / eps
    phi_tot, dphi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh, compute_gradient=True)
    lap_phi_tot = tools.laplacian_by_fft(phi_tot, Gv, mesh)
    fac = 4 * numpy.pi

    deps = tools.gradient_by_fft(eps, Gv, mesh)
    tmp = eps*lap_phi_tot
    for x in range(3):
        tmp += deps[x]*dphi_tot[x]

    r = -fac * rho_solute - tmp
    print("r0:", abs(r).max()) 
    for i in range(max_cycle):
        fake_rho = r / sqrt_eps
        v = tools.solve_poisson(cell, fake_rho, coulG=coulG, Gv=Gv, mesh=mesh)[0] / sqrt_eps
        v = r
        if i == 0:
            p = v
        else:
            beta = lib.vdot(v, r) / lib.vdot(v_old, r_old)
            p = v + beta * p_old
        Ap = -p * q - fac * r
        alpha = lib.vdot(v, r) / lib.vdot(p, Ap)

        r_old = r.copy()
        v_old = v.copy()
        p_old = p.copy()
        r = r - alpha * Ap
        phi_tot = alpha * p + phi_tot
        print("res = ", abs(r).max())
        if abs(r).max() < conv_tol:
            break

    return rho_tot


def _mixing(sccs, rho_solute, eps, coulG=None, Gv=None, mesh=None,
            conv_tol=1e-5, max_cycle=20):
    cell = sccs.cell
    mixing_factor = sccs.mixing_factor
    if mesh is None:
        mesh = sccs.mesh
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if Gv is None:
        Gv = cell.get_Gv(mesh=mesh)

    log_eps1 = _get_log_eps_gradient(eps, Gv, mesh)
    fac = 1. / (4. * numpy.pi)
    log_eps1 = lib.multiply(fac, log_eps1, out=log_eps1)

    rho_solute_over_eps = numpy.divide(rho_solute, eps)
    rho_iter = numpy.zeros_like(rho_solute)
    rho_iter_old = lib.copy(rho_iter)
    for i in range(max_cycle):
        rho_tot = lib.add(rho_solute_over_eps, rho_iter)
        _, dphi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh, compute_gradient=True)
        rho_tot = None
        for x in range(3):
            if x == 0:
                rho_iter = lib.multiply(dphi_tot[x], log_eps1[x], out=rho_iter)
            else:
                tmp = lib.multiply(dphi_tot[x], log_eps1[x])
                rho_iter = lib.add(rho_iter, tmp, out=rho_iter)
        rho_iter = lib.multiply(mixing_factor, rho_iter, out=rho_iter)
        rho_iter = lib.add(rho_iter, (1.-mixing_factor)*rho_iter_old, out=rho_iter)
        res = abs(rho_iter - rho_iter_old).max()
        logger.info(sccs, 'cycle= %d  res= %4.3g', i+1, res)
        if res < conv_tol:
            break
        rho_iter_old = lib.copy(rho_iter, out=rho_iter_old)
    if res > conv_tol:
        logger.warn(sccs, 'SCCS did not converge.')

    rho_tot = lib.add(rho_solute_over_eps, rho_iter)
    return rho_tot


def kernel(sccs, rho_elec, rho_core=None, method="mixing",
           rho_min=1e-4, rho_max=1.5e-3, conv_tol=1e-5, max_cycle=20):
    if rho_core is None:
        rho_core = 0
    cell = sccs.cell
    mesh = sccs.mesh
    ng = numpy.prod(mesh)
    eps0 = sccs.eps
    Gv = cell.get_Gv(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)

    rho_elec = rho_elec.reshape(-1, ng)
    if len(rho_elec) == 1:
        rho_elec = rho_elec[0]
        drho_elec = None
    elif len(rho_elec) >= 4:
        drho_elec = rho_elec[1:4]
        rho_elec = rho_elec[0]
    else:
        raise ValueError("Input density has the wrong shape."
                         "Expect either (1, ngrid) or (4, ngrid).")
    #rho_elec[rho_elec<0] = 0

    rho_solute = lib.add(rho_elec, rho_core)
    eps, deps_intermediate = _get_eps(rho_elec, None, rho_min, rho_max, eps0)

    rho_tot = None
    if method.upper() == "PCG":
        rho_tot = _pcg(sccs, rho_solute, eps, coulG=coulG, Gv=Gv, mesh=mesh,
                       conv_tol=conv_tol, max_cycle=max_cycle)
    elif method.upper() == "MIXING":
        rho_tot = _mixing(sccs, rho_solute, eps, coulG=coulG, Gv=Gv, mesh=mesh,
                          conv_tol=conv_tol, max_cycle=max_cycle)
    else:
        raise KeyError(f"Unrecognized method: {method}.")

    deps_drho = _get_deps_drho(eps, deps_intermediate)
    deps_intermediate = None

    if rho_tot is None:
        raise RuntimeError

    rho_pol = sccs.rho_pol = lib.subtract(rho_tot, rho_solute)
    phi_pol = tools.solve_poisson(cell, rho_pol, coulG=coulG, Gv=Gv, mesh=mesh)[0]
    weight = cell.vol / ng 
    e_pol = .5 * lib.sum(lib.multiply(phi_pol, rho_solute)) * weight
    rho_pol = None
    logger.info(sccs, 'Polarization energy = %.8g', e_pol)

    _, dphi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh, compute_gradient=True)
    dphi_tot_square = None
    for x in range(3):
        if x == 0:
            dphi_tot_square = lib.multiply(dphi_tot[x], dphi_tot[x])
        else:
            tmp = lib.multiply(dphi_tot[x], dphi_tot[x])
            dphi_tot_square = lib.add(dphi_tot_square, tmp, out=dphi_tot_square)
    dphi_tot = None

    fac = -1. / (8.*numpy.pi)
    deps_drho = lib.multiply(fac, deps_drho, out=deps_drho)
    sccs.phi_eps = phi_eps = lib.multiply(deps_drho, dphi_tot_square)
    dphi_tot_square = None
    phi_sccs = lib.add(phi_pol, phi_eps)
    return e_pol, phi_sccs

class SCCS(lib.StreamObject):
    def __init__(self, cell, mesh, eps=78.3553, rho_min=1e-4, rho_max=1.5e-3):
        self.cell = cell
        self.mesh = mesh
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.eps = eps
        self.method = 'mixing'
        self.mixing_factor = 0.6
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.max_cycle = 50
        self.conv_tol = 1e-6
        self.rho_pol = None
        self.phi_eps = None

    def kernel(self, rho, rho_core=None):
        return kernel(self, rho, rho_core=rho_core, method=self.method,
                      rho_min=self.rho_min, rho_max=self.rho_max,
                      conv_tol=self.conv_tol, max_cycle=self.max_cycle)
