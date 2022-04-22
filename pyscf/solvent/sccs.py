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

def _get_log_eps_gradient(cell, eps, Gv=None, mesh=None, method='FFT'):
    log_eps = numpy.log(eps)
    if method.upper() == 'FFT':
        out = tools.gradient_by_fft(log_eps, Gv, mesh)
    elif method.upper() == 'FDIFF':
        out = tools.gradient_by_fdiff(cell, log_eps, mesh)
    else:
        raise KeyError
    return out

def _get_deps_drho(eps, deps_intermediate):
    return lib.multiply(eps, deps_intermediate)

def _pcg(sccs, rho_solute, eps, coulG=None, Gv=None, mesh=None,
         gradient_method=None, conv_tol=1e-5, max_cycle=50):
    cell = sccs.cell
    if mesh is None:
        mesh = sccs.mesh
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if Gv is None:
        Gv = cell.get_Gv(mesh=mesh)
    if gradient_method is None:
        gradient_method = sccs.gradient_method

    sqrt_eps = numpy.sqrt(eps)
    q = lib.multiply(sqrt_eps, tools.laplacian_by_fft(sqrt_eps, Gv, mesh))

    if sccs.phi_tot is None:
        phi_tot = tools.solve_poisson(cell, rho_solute, coulG=coulG, Gv=Gv, mesh=mesh)[0]
    else:
        phi_tot = sccs.phi_tot

    if gradient_method.upper() == "FFT":
        dphi_tot = tools.gradient_by_fft(phi_tot, Gv, mesh)
    elif gradient_method.upper() == "FDIFF":
        dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh)
    else:
        raise NotImplementedError
        
    fac = 4 * numpy.pi
    tmp = None
    if gradient_method.upper() == "FFT":
        for x in range(3):
            eps_dphi_x = lib.multiply(eps, dphi_tot[x])
            if x == 0:
                tmp = tools.gradient_by_fft(eps_dphi_x, Gv, mesh)[x]
            else:
                tmp = lib.add(tmp, tools.gradient_by_fft(eps_dphi_x, Gv, mesh)[x], out=tmp)
    elif gradient_method.upper() == "FDIFF":
        for x in range(3):
            eps_dphi_x = lib.multiply(eps, dphi_tot[x])
            if x == 0:
                tmp = tools.gradient_by_fdiff(cell, eps_dphi_x, mesh)[x]
            else:
                tmp = lib.add(tmp, tools.gradient_by_fdiff(cell, eps_dphi_x, mesh)[x], out=tmp)
    else:
        raise NotImplementedError

    dphi_tot = None

    r = lib.multiply(-fac, rho_solute) 
    r = lib.subtract(r, tmp, out=r)
    tmp = None

    invs_sqrt_eps = lib.reciprocal(sqrt_eps)
    for i in range(max_cycle):
        r_norm = numpy.linalg.norm(lib.vdot(r, r))
        logger.info(sccs, 'cycle= %d  res= %4.3g', i, r_norm)
        if r_norm < conv_tol:
            break

        fake_rho = lib.multiply(r, invs_sqrt_eps)
        v = lib.multiply(tools.solve_poisson(cell, fake_rho, coulG=coulG, Gv=Gv, mesh=mesh)[0], invs_sqrt_eps)
        fake_rho = None
        if i == 0:
            beta = 0.0
            p = v
        else:
            beta = lib.vdot(v, r) / lib.vdot(v_old, r_old)
            p = lib.add(v, lib.multiply(beta, p, out=p), out=p)
        vq = lib.multiply(v, q)
        vq = lib.subtract(lib.multiply(-fac, r), vq, out=vq)
        if i == 0:
            Ap = vq
        else:
            Ap = lib.add(vq, lib.multiply(beta, Ap, out=Ap), out=Ap)
        vq = None
        alpha = lib.vdot(v, r) / lib.vdot(p, Ap)

        r_old = lib.copy(r)
        v_old = v

        r = lib.subtract(r, lib.multiply(alpha, Ap), out=r)
        phi_tot = lib.add(phi_tot, lib.multiply(alpha, p), out=phi_tot)

    if r_norm > conv_tol:
        logger.warn(sccs, 'SCCS did not converge.')

    return phi_tot

def _mixing(sccs, rho_solute, eps, rho_pol=None, coulG=None, Gv=None, mesh=None,
            gradient_method=None, conv_tol=1e-5, max_cycle=50):
    cell = sccs.cell
    mixing_factor = sccs.mixing_factor
    if mesh is None:
        mesh = sccs.mesh
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if Gv is None:
        Gv = cell.get_Gv(mesh=mesh)
    if gradient_method is None:
        gradient_method = sccs.gradient_method

    log_eps1 = _get_log_eps_gradient(cell, eps, Gv, mesh, gradient_method)
    fac = 1. / (4. * numpy.pi)
    log_eps1 = lib.multiply(fac, log_eps1, out=log_eps1)

    rho_solute_over_eps = numpy.divide(rho_solute, eps)
    if rho_pol is not None:
        # use the polarization density from previous scf step
        # as the initial guess
        tmp = lib.subtract(rho_pol, rho_solute_over_eps)
        rho_iter = lib.add(tmp, rho_solute)
    else:
        #rho_iter = numpy.zeros_like(rho_solute)
        rho_iter = lib.subtract(rho_solute, rho_solute_over_eps)
    rho_iter_old = lib.copy(rho_iter)
    for i in range(max_cycle):
        rho_tot = lib.add(rho_solute_over_eps, rho_iter)
        if gradient_method.upper() == "FFT":
            _, dphi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh,
                                              compute_potential=False, compute_gradient=True)
        elif gradient_method.upper() == "FDIFF":
            phi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh)[0]
            dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh)
            phi_tot = None
        else:
            raise NotImplementedError

        rho_tot = None
        for x in range(3):
            if x == 0:
                rho_iter = lib.multiply(dphi_tot[x], log_eps1[x], out=rho_iter)
            else:
                tmp = lib.multiply(dphi_tot[x], log_eps1[x])
                rho_iter = lib.add(rho_iter, tmp, out=rho_iter)
        rho_iter = lib.multiply(mixing_factor, rho_iter, out=rho_iter)
        rho_iter = lib.add(rho_iter, (1.-mixing_factor)*rho_iter_old, out=rho_iter)
        diff = lib.subtract(rho_iter, rho_iter_old)
        diff_norm = numpy.linalg.norm(lib.vdot(diff, diff))
        logger.info(sccs, 'cycle= %d  res= %4.3g', i+1, diff_norm)
        if diff_norm < conv_tol:
            break
        rho_iter_old = lib.copy(rho_iter, out=rho_iter_old)
    if diff_norm > conv_tol:
        logger.warn(sccs, 'SCCS did not converge.')

    rho_tot = lib.add(rho_solute_over_eps, rho_iter)
    return rho_tot


def kernel(sccs, rho_elec, rho_core=None, method="mixing",
           rho_min=1e-4, rho_max=1.5e-3, conv_tol=1e-5, max_cycle=50):
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
    phi_tot = None
    if method.upper() == "PCG":
        phi_tot = _pcg(sccs, rho_solute, eps, coulG=coulG, Gv=Gv, mesh=mesh,
                       conv_tol=conv_tol, max_cycle=max_cycle)
        sccs.phi_tot = phi_tot
    elif method.upper() == "MIXING":
        rho_tot = _mixing(sccs, rho_solute, eps, rho_pol=sccs.rho_pol,
                          coulG=coulG, Gv=Gv, mesh=mesh,
                          conv_tol=conv_tol, max_cycle=max_cycle)
    else:
        raise KeyError(f"Unrecognized method: {method}.")

    deps_drho = _get_deps_drho(eps, deps_intermediate)
    deps_intermediate = None

    e_pol, phi_sccs = get_veff(sccs, rho_solute, eps, deps_drho, rho_tot, phi_tot,
                               coulG, Gv, mesh)
    return e_pol, phi_sccs


def get_veff(sccs, rho_solute, eps, deps_drho, rho_tot=None, phi_tot=None,
             coulG=None, Gv=None, mesh=None, gradient_method=None):
    cell = sccs.cell
    if mesh is None:
        mesh = sccs.mesh
    if Gv is None:
        Gv = cell.get_Gv(mesh)
    if coulG is None:
        coulG = tools.get_coulG(cell, mesh=mesh)
    if gradient_method is None:
        gradient_method = sccs.gradient_method
    if rho_tot is None and phi_tot is None:
        raise KeyError("Either rho_tot or phi_tot need to be specified.")

    rho_pol = None
    if rho_tot is not None:
        sccs.rho_pol = rho_pol = lib.subtract(rho_tot, rho_solute)
        phi_pol = tools.solve_poisson(cell, rho_pol, coulG=coulG, Gv=Gv, mesh=mesh)[0]
    if phi_tot is None:
        phi_tot = tools.solve_poisson(cell, rho_tot, coulG=coulG, Gv=Gv, mesh=mesh)[0]

    if gradient_method.upper() == "FFT":
        dphi_tot = tools.gradient_by_fft(phi_tot, Gv, mesh)
    elif gradient_method.upper() == "FDIFF":
        dphi_tot = tools.gradient_by_fdiff(cell, phi_tot, mesh)
    else:
        raise NotImplementedError

    if rho_pol is None:
        log_eps1 = _get_log_eps_gradient(cell, eps, Gv, mesh, gradient_method)
        log_eps1 = lib.multiply(.25/numpy.pi, log_eps1, out=log_eps1)

        rho_iter=None
        for x in range(3):
            if x == 0:
                rho_iter = lib.multiply(dphi_tot[x], log_eps1[x], out=rho_iter)
            else:
                tmp = lib.multiply(dphi_tot[x], log_eps1[x])
                rho_iter = lib.add(rho_iter, tmp, out=rho_iter)
        sccs.rho_pol = rho_pol = rho_iter - rho_solute + rho_solute / eps
        phi_pol = tools.solve_poisson(cell, rho_pol, coulG=coulG, Gv=Gv, mesh=mesh)[0]

    weight = cell.vol / numpy.prod(mesh)
    e_pol = .5 * lib.sum(lib.multiply(phi_pol, rho_solute)) * weight
    logger.info(sccs, 'Polarization energy = %.8g', e_pol)

    dphi_tot_square = None
    for x in range(3):
        if x == 0:
            dphi_tot_square = lib.multiply(dphi_tot[x], dphi_tot[x])
        else:
            tmp = lib.multiply(dphi_tot[x], dphi_tot[x])
            dphi_tot_square = lib.add(dphi_tot_square, tmp, out=dphi_tot_square)
    dphi_tot = None

    deps_drho = lib.multiply(-0.125/numpy.pi, deps_drho, out=deps_drho)
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
        self.max_cycle = 100
        self.conv_tol = 1e-5
        self.rho_pol = None
        self.phi_eps = None
        self.phi_tot = None
        self.rho_core = None
        self.gradient_method = 'fft'

    def kernel(self, rho, rho_core=None):
        return kernel(self, rho, rho_core=rho_core, method=self.method,
                      rho_min=self.rho_min, rho_max=self.rho_max,
                      conv_tol=self.conv_tol, max_cycle=self.max_cycle)
