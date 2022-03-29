'''
Self-consistente continuum solvation model

Reference:
1. J. Chem. Phys. 136, 064102 (2012); https://doi.org/10.1063/1.3676407
2. J. Chem. Phys. 144, 014103 (2016); https://doi.org/10.1063/1.4939125
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.pbc import tools

libpbc = lib.load_library('libpbc')

def _get_eps(rhoR, rho_core, rho_min, rho_max, eps0):
    if rho_core is not None:
        rhoR += rho_core
    print("rhoR.min() = ", rhoR.min())
    ng = rhoR.size
    out = numpy.empty_like(rhoR, order='C')
    fun = getattr(libpbc, 'get_eps')
    fun(out.ctypes.data_as(ctypes.c_void_p),
        rhoR.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(rho_min), ctypes.c_double(rho_max),
        ctypes.c_double(eps0), ctypes.c_size_t(ng))
    return out

def _get_eps1_intermediate(rhoR, rho_core, rho_min, rho_max, eps0):
    if rho_core is not None:
        rhoR += rho_core
    ng = rhoR.size
    out = numpy.empty_like(rhoR, order='C')
    fun = getattr(libpbc, 'get_eps1_intermediate')
    fun(out.ctypes.data_as(ctypes.c_void_p),
        rhoR.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(rho_min), ctypes.c_double(rho_max),
        ctypes.c_double(eps0), ctypes.c_size_t(ng))
    return out

def _get_log_eps1(cell, rhoR, rho_core, Gv, mesh, eps1_inter, rhoR1=None):
    if rho_core is not None:
        rhoR += rho_core
    if rhoR1 is None:
        rhoG = tools.fft(rhoR, mesh)
        rhoG1 = cell.contract_rhoG_Gv(rhoG, Gv).reshape(-1, rhoR.size)
        rhoR1 = numpy.asarray(tools.ifft(rhoG1, mesh).real, order='C')
        rhoG = rhoG1 = None

    out = numpy.empty_like(rhoR1, order='C')
    for x in range(3):
        out[x] = lib.multiply(rhoR1[x], eps1_inter, out=out[x])
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

def _get_deps_drho(eps, eps1_inter):
    return lib.multiply(eps, eps1_inter)

from pyscf.tools import write_cube
def kernel(sccs, rhoR, rho_core=None, rho_min=1e-4, rho_max=1.5e-3, conv_tol=1e-5, max_cycle=20):
    cell = sccs.cell
    mesh = sccs.mesh
    eps0 = sccs.eps
    Gv = cell.get_Gv(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)

    rhoR[rhoR<0] = 0
    rhoR = rhoR.reshape(-1, numpy.prod(mesh))
    if len(rhoR) == 1:
        rhoR = rhoR[0]
        rhoR1 = None
    elif len(rhoR) == 4:
        rhoR1 = rhoR[1:4]
        rhoR = rhoR[0]
    else:
        raise RuntimeError

    rhoR = numpy.asarray(rhoR, order='C', dtype=float)
    write_cube.write_cube(cell, rhoR.reshape(*mesh), mesh, "rhoR.cube")
    if rho_core is not None:
        write_cube.write_cube(cell, rho_core.reshape(*mesh), mesh, "rho_core.cube")
    if rhoR1 is not None:
        rhoR1 = numpy.asarray(rhoR1, order='C', dtype=float)

    rho_solute = rhoR + rho_core
    write_cube.write_cube(cell, rho_solute.reshape(*mesh), mesh, "rho_tot.cube")
    eps = _get_eps(rhoR, None, rho_min, rho_max, eps0)
    write_cube.write_cube(cell, eps.reshape(*mesh), mesh, "eps.cube")
    eps_inter = _get_eps1_intermediate(rhoR, None, rho_min, rho_max, eps0)
    #log_eps1 = _get_log_eps1(cell, rhoR, rho_core, Gv, mesh, eps_inter, rhoR1=rhoR1)
    log_eps1 = _get_log_eps1_fdiff(cell, eps, rhoR, None, rho_min, rho_max, mesh)
    #print('log_eps1 max:' , abs(log_eps1).max())
    deps_drho = _get_deps_drho(eps, eps_inter)
    rho_iter = numpy.zeros_like(rhoR)
    for i in range(max_cycle):
        rho_iter_old = lib.copy(rho_iter)
        rho_tot = lib.add(rho_solute / eps, rho_iter)
        rho_pol = rho_tot - rhoR
        write_cube.write_cube(cell, rho_pol.reshape(*mesh), mesh, f"rho_pol_{i}.cube")
        rhoG = tools.fft(rho_tot, mesh)
        rhoG1 = cell.contract_rhoG_Gv(rhoG, Gv).reshape(3,-1)
        for x in range(3):
            vG1 = lib.multiply(rhoG1[x], coulG)
            vR1 = tools.ifft(vG1, mesh).real  #Gamma point
            #write_cube.write_cube(cell, vR1.reshape(*mesh), mesh, f"vR1_{x}.cube")
            #write_cube.write_cube(cell, log_eps1[x].reshape(*mesh), mesh, f"log_eps1_{x}.cube")
            print('vR1 max:',  abs(vR1).max())
            print('log_eps1 max:', abs(log_eps1[x]).max())
            if x == 0:
                rho_iter = lib.multiply(vR1, log_eps1[x], out=rho_iter)
            else:
                tmp = lib.multiply(vR1, log_eps1[x])
                rho_iter = lib.add(rho_iter, tmp, out=rho_iter)
        res = abs(rho_iter - rho_iter_old).max()
        rho_iter = 0.6 * rho_iter + 0.4 * rho_iter_old
        print(f'res = {res}')
        if res < conv_tol:
            break

    rho_tot = lib.add(rho_solute / eps, rho_iter)
    rhoG = tools.fft(rho_tot, mesh)
    rhoG1 = cell.contract_rhoG_Gv(rhoG, Gv).reshape(3,-1)
    vR1_square = numpy.empty_like(rhoR)
    for x in range(3):
        vG1 = lib.multiply(rhoG1[x], coulG)
        vR1 = tools.ifft(vG1, mesh).real
        if x == 0:
            vR1_square = lib.multiply(vR1, vR1, out=vR1_square)
        else:
            tmp = lib.multiply(vR1, vR1)
            vR1_square = lib.add(vR1_square, tmp, out=vR1_square)

    vG = lib.multiply(rhoG, coulG)
    vR = tools.ifft(vG, mesh).real
    vR += lib.multiply(deps_drho, vR1_square) * (-0.125/numpy.pi)
    return vR

class SCCS(lib.StreamObject):
    def __init__(self, cell, mesh, eps=78.3553, rho_min=1e-4, rho_max=1.5e-3):
        self.cell = cell
        self.mesh = mesh
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.eps = eps
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.max_cycle = 5
        self.conv_tol = 1e-5

    def kernel(self, rho, rho_core=None):
        return kernel(self, rho, rho_core=rho_core,
                      rho_min=self.rho_min, rho_max=self.rho_max,
                      conv_tol=self.conv_tol, max_cycle=self.max_cycle)
