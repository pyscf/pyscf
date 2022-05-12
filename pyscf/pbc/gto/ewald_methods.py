import ctypes
import numpy as np
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import mole
from pyscf.pbc import tools

libpbc = lib.load_library('libpbc')

def _cut_mesh_for_ewald(cell, mesh):
    mesh = np.copy(mesh)
    mesh_max = np.asarray(np.linalg.norm(cell.lattice_vectors(), axis=1) * 2,
                          dtype=int)

    if (cell.dimension < 2 or
        (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
        mesh_max[cell.dimension:] = mesh[cell.dimension:]

    mesh_max[mesh_max<80] = 80
    mesh[mesh>mesh_max] = mesh_max[mesh>mesh_max]
    return mesh

def _bspline(u, n=4):
    fac = 1. / scipy.special.factorial(n-1)
    M = 0
    for k in range(n+1):
        fac1 = ((-1)**k) * scipy.special.binom(n, k)
        M += fac1 * ((np.maximum(u-k, 0)) ** (n-1))
    M *= fac
    return M

def bspline(u, ng, n=4):
    u = np.asarray(u).ravel()
    u_floor = np.floor(u)
    delta = u - u_floor
    idx = []
    val = []
    for i in range(n):
        idx.append(np.rint((u_floor - i) % ng).astype(int))
        val.append(delta + i)

    M = np.zeros((u.size, ng))
    for i in range(n):
        M[np.arange(u.size),idx[i]] += _bspline(val[i], n)

    m = np.arange(ng)
    b = np.exp(2*np.pi*1j*(n-1)*m/ng)
    tmp = 0
    for k in range(n-1):
        tmp += _bspline(k+1, n) * np.exp(2*np.pi*1j*m*k/ng) 
    b /= tmp
    return M, b, idx

def _get_ewald_direct(cell, ew_eta=None, ew_cut=None):
    if ew_eta is None:
        ew_eta = cell.get_ewald_params()[0]
    if ew_cut is None:
        ew_cut = cell.get_ewald_params()[1]

    chargs = np.asarray(cell.atom_charges(), order='C', dtype=float)
    coords = np.asarray(cell.atom_coords(), order='C')
    Lall = np.asarray(cell.get_lattice_Ls(rcut=ew_cut), order='C')

    natm = len(chargs)
    nL = len(Lall)
    ewovrl = np.zeros([1])
    fun = getattr(libpbc, "get_ewald_direct")
    fun(ewovrl.ctypes.data_as(ctypes.c_void_p),
        chargs.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        Lall.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(ew_eta), ctypes.c_double(ew_cut),
        ctypes.c_int(natm), ctypes.c_int(nL))
    return ewovrl[0] 

#XXX The default interpolation order may be too high
def smooth_particle_mesh_ewald(cell, ew_eta=None, ew_cut=None, order=10):
    if cell.a is None:
        return mole.energy_nuc(cell)

    if cell.natm == 0:
        return 0

    if cell.dimension != 3:
        raise NotImplementedError("Particle mesh ewald only works for 3D.")

    if ew_eta is None:
        ew_eta = cell.get_ewald_params()[0]
    if ew_cut is None:
        ew_cut = cell.get_ewald_params()[1]

    chargs = cell.atom_charges()
    coords = cell.atom_coords()

    ewovrl = _get_ewald_direct(cell, ew_eta, ew_cut)
    ewself  = -.5 * np.dot(chargs,chargs) * 2 * ew_eta / np.sqrt(np.pi)
    if cell.dimension == 3:
        ewself += -.5 * np.sum(chargs)**2 * np.pi/(ew_eta**2 * cell.vol)

    mesh = _cut_mesh_for_ewald(cell, cell.mesh)

    b = cell.reciprocal_vectors(norm_to=1)
    u = np.dot(coords, b.T) * mesh[None,:]

    Mx, bx, idx = bspline(u[:,0], mesh[0], order)
    My, by, idy = bspline(u[:,1], mesh[1], order)
    Mz, bz, idz = bspline(u[:,2], mesh[2], order)

    idx = np.asarray(idx).T
    idy = np.asarray(idy).T
    idz = np.asarray(idz).T
    Mx_s = Mx[np.indices(idx.shape)[0], idx]
    My_s = My[np.indices(idy.shape)[0], idy]
    Mz_s = Mz[np.indices(idz.shape)[0], idz]

    #Q = np.einsum('i,ix,iy,iz->xyz', chargs, Mx, My, Mz)
    Q = np.zeros([*mesh])
    for ia in range(len(chargs)):
        Q_s = np.einsum('x,y,z->xyz', Mx_s[ia], My_s[ia], Mz_s[ia])
        Q[np.ix_(idx[ia], idy[ia], idz[ia])] += chargs[ia] * Q_s

    B = np.einsum('x,y,z->xyz', bx*bx.conj(), by*by.conj(), bz*bz.conj())

    Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    absG2 = lib.multiply_sum(Gv, Gv, axis=1)
    absG2[absG2==0] = 1e200
    coulG = 4*np.pi / absG2
    C = weights * coulG * np.exp(-absG2/(4*ew_eta**2))
    C = C.reshape(*mesh)

    Q_ifft = tools.ifft(Q, mesh).reshape(*mesh)
    tmp = tools.fft(B * C * Q_ifft, mesh).reshape(*mesh)
    ewg = 0.5 * np.prod(mesh) * np.einsum('xyz,xyz->', Q, tmp).real

    logger.debug(cell, 'Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg

if __name__ == "__main__":
    from pyscf.pbc import gto as pbcgto
    cell = pbcgto.Cell()
    cell.a = np.diag([12.,]*3)
    cell.atom = '''
        O          5.84560        5.21649        5.10372
        H          6.30941        5.30070        5.92953
        H          4.91429        5.26674        5.28886
    '''
    cell.ke_cutoff = 200
    cell.pseudo = 'gth-pade'
    cell.verbose = 5
    cell.build()
