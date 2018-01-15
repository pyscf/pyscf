import sys
import copy
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv3

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    fftn_wrapper = pyfftw.interfaces.numpy_fft.fftn
    ifftn_wrapper = pyfftw.interfaces.numpy_fft.ifftn
    nproc = lib.num_threads()
except ImportError:
    def fftn_wrapper(a, s=None, axes=None, norm=None, **kwargs):
        return np.fft.fftn(a, s, axes)
    def ifftn_wrapper(a, s=None, axes=None, norm=None, **kwargs):
        return np.fft.ifftn(a, s, axes)
    nproc = 1

def fft(f, gs):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    Re: MH (3.25), we assume Ns := ngs = 2*gs+1

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of :func:`cartesian_prod`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    if f.size == 0:
        return np.zeros_like(f)

    f3d = f.reshape([-1] + [2*x+1 for x in gs])
    assert(f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = fftn_wrapper(f3d, axes=(1,2,3), threads=nproc)
    if f.ndim == 1:
        return g3d.ravel()
    else:
        return g3d.reshape(f.shape[0], -1)

def ifft(g, gs):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `span3`.
        gs : (3,) ndarray of ints
            The number of *positive* G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    if g.size == 0:
        return np.zeros_like(g)

    g3d = g.reshape([-1] + [2*x+1 for x in gs])
    assert(g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = ifftn_wrapper(g3d, axes=(1,2,3), threads=nproc)
    if g.ndim == 1:
        return f3d.ravel()
    else:
        return f3d.reshape(g.shape[0], -1)


def fftk(f, gs, expmikr):
    '''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*expmikr, gs)


def ifftk(g, gs, expikr):
    '''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, gs) * expikr


def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, gs=None, Gv=None,
              wrap_around=True):
    '''Calculate the Coulomb kernel for all G-vectors, handling G=0 and exchange.

    Args:
        k : (3,) ndarray
            k-point
        exx : bool or str
            Whether this is an exchange matrix element.
        mf : instance of :class:`SCF`

    Returns:
        coulG : (ngs,) ndarray
            The Coulomb kernel.

    '''
    exxdiv = exx
    if isinstance(exx, str):
        exxdiv = exx
    elif exx and mf is not None:
# sys.stderr.write('pass exxdiv directly')
        exxdiv = mf.exxdiv

    if gs is None:
        gs = cell.gs
    if Gv is None:
        Gv = cell.get_Gv(gs)

    kG = k + Gv
    equal2boundary = np.zeros(Gv.shape[0], dtype=bool)
    if wrap_around and abs(k).sum() > 1e-9:
        # Here we 'wrap around' the high frequency k+G vectors into their lower
        # frequency counterparts.  Important if you want the gamma point and k-point
        # answers to agree
        b = cell.reciprocal_vectors()
        box_edge = np.einsum('i,ij->ij', np.asarray(gs)+0.5, b)
        assert(all(np.linalg.solve(box_edge.T, k).round(9).astype(int)==0))
        reduced_coords = np.linalg.solve(box_edge.T, kG.T).T.round(9)
        on_edge = reduced_coords.astype(int)
        if cell.dimension >= 1:
            equal2boundary |= reduced_coords[:,0] == 1
            equal2boundary |= reduced_coords[:,0] ==-1
            kG[on_edge[:,0]== 1] -= 2 * box_edge[0]
            kG[on_edge[:,0]==-1] += 2 * box_edge[0]
        if cell.dimension >= 2:
            equal2boundary |= reduced_coords[:,1] == 1
            equal2boundary |= reduced_coords[:,1] ==-1
            kG[on_edge[:,1]== 1] -= 2 * box_edge[1]
            kG[on_edge[:,1]==-1] += 2 * box_edge[1]
        if cell.dimension == 3:
            equal2boundary |= reduced_coords[:,2] == 1
            equal2boundary |= reduced_coords[:,2] ==-1
            kG[on_edge[:,2]== 1] -= 2 * box_edge[2]
            kG[on_edge[:,2]==-1] += 2 * box_edge[2]

    absG2 = np.einsum('gi,gi->g', kG, kG)

    if mf is not None and hasattr(mf, 'kpts'):
        kpts = mf.kpts
    else:
        kpts = k.reshape(1,3)
    Nk = len(kpts)

    if not exxdiv:
        with np.errstate(divide='ignore'):
            coulG = 4*np.pi/absG2
        coulG[absG2==0] = 0
    elif exxdiv == 'vcut_sph':  # PRB 77 193110
        Rc = (3*Nk*cell.vol/(4*np.pi))**(1./3)
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.cos(np.sqrt(absG2)*Rc))
        coulG[absG2==0] = 4*np.pi*0.5*Rc**2
    elif exxdiv == 'ewald':
        with np.errstate(divide='ignore'):
            coulG = 4*np.pi/absG2
        G0_idx = np.where(absG2==0)[0]
        if len(G0_idx) > 0:
            coulG[G0_idx] = Nk*cell.vol*madelung(cell, kpts)
    elif exxdiv == 'vcut_ws':  # PRB 87, 165122
        assert(cell.dimension == 3)
        if not hasattr(mf, '_ws_exx'):
            mf._ws_exx = precompute_exx(cell, kpts)
        exx_alpha = mf._ws_exx['alpha']
        exx_kcell = mf._ws_exx['kcell']
        exx_q = mf._ws_exx['q']
        exx_vq = mf._ws_exx['vq']

        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.exp(-absG2/(4*exx_alpha**2)))
        coulG[absG2==0] = np.pi / exx_alpha**2
        # Index k+Gv into the precomputed vq and add on
        gxyz = np.dot(kG, exx_kcell.lattice_vectors().T)/(2*np.pi)
        gxyz = gxyz.round(decimals=6).astype(int)
        ngs = 2*np.asarray(exx_kcell.gs)+1
        gxyz = (gxyz + ngs)%(ngs)
        qidx = (gxyz[:,0]*ngs[1] + gxyz[:,1])*ngs[2] + gxyz[:,2]
        #qidx = [np.linalg.norm(exx_q-kGi,axis=1).argmin() for kGi in kG]
        maxqv = abs(exx_q).max(axis=0)
        is_lt_maxqv = (abs(kG) <= maxqv).all(axis=1)
        coulG = coulG.astype(np.complex128)
        coulG[is_lt_maxqv] += exx_vq[qidx[is_lt_maxqv]]

    coulG[equal2boundary] = 0

    return coulG

def precompute_exx(cell, kpts):
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc.dft import gen_grid
    log = lib.logger.Logger(cell.stdout, cell.verbose)
    log.debug("# Precomputing Wigner-Seitz EXX kernel")
    Nk = get_monkhorst_pack_size(cell, kpts)
    log.debug("# Nk = %s", Nk)

    kcell = pbcgto.Cell()
    kcell.atom = 'H 0. 0. 0.'
    kcell.spin = 1
    kcell.unit = 'B'
    kcell.verbose = 0
    kcell.a = cell.lattice_vectors() * Nk
    Lc = 1.0/lib.norm(np.linalg.inv(kcell.a), axis=0)
    log.debug("# Lc = %s", Lc)
    Rin = Lc.min() / 2.0
    log.debug("# Rin = %s", Rin)
    # ASE:
    alpha = 5./Rin # sqrt(-ln eps) / Rc, eps ~ 10^{-11}
    log.info("WS alpha = %s", alpha)
    kcell.gs = np.array([2*int(L*alpha*3.0) for L in Lc])  # ~ [60,60,60]
    # QE:
    #alpha = 3./Rin * np.sqrt(0.5)
    #kcell.gs = (4*alpha*np.linalg.norm(kcell.a,axis=1)).astype(int)
    log.debug("# kcell.gs FFT = %s", kcell.gs)
    kcell.build(False,False)
    rs = gen_grid.gen_uniform_grids(kcell)
    kngs = len(rs)
    log.debug("# kcell kngs = %d", kngs)
    corners = np.dot(np.indices((2,2,2)).reshape((3,8)).T, kcell.a)
    #vR = np.empty(kngs)
    #for i, rv in enumerate(rs):
    #    # Minimum image convention to corners of kcell parallelepiped
    #    r = lib.norm(rv-corners, axis=1).min()
    #    if np.isclose(r, 0.):
    #        vR[i] = 2*alpha / np.sqrt(np.pi)
    #    else:
    #        vR[i] = scipy.special.erf(alpha*r) / r
    r = np.min([lib.norm(rs-c, axis=1) for c in corners], axis=0)
    vR = scipy.special.erf(alpha*r) / (r+1e-200)
    vR[r<1e-9] = 2*alpha / np.sqrt(np.pi)
    vG = (kcell.vol/kngs) * fft(vR, kcell.gs)
    ws_exx = {'alpha': alpha,
              'kcell': kcell,
              'q'    : kcell.Gv,
              'vq'   : vG}
    log.debug("# Finished precomputing")
    return ws_exx


def madelung(cell, kpts):
    Nk = get_monkhorst_pack_size(cell, kpts)
    ecell = copy.copy(cell)
    ecell._atm = np.array([[1, 0, 0, 0, 0, 0]])
    ecell._env = np.array([0., 0., 0.])
    ecell.unit = 'B'
    #ecell.verbose = 0
    ecell.a = np.einsum('xi,x->xi', cell.lattice_vectors(), Nk)
    ecell.gs = np.asarray(cell.gs) * Nk
    ew_eta, ew_cut = ecell.get_ewald_params(cell.precision, ecell.gs)
    lib.logger.debug1(cell, 'Monkhorst pack size %s ew_eta %s ew_cut %s',
                      Nk, ew_eta, ew_cut)
    return -2*ecell.ewald(ew_eta, ew_cut)


def get_monkhorst_pack_size(cell, kpts):
    skpts = cell.get_scaled_kpts(kpts).round(decimals=6)
    Nk = np.array([len(np.unique(ki)) for ki in skpts.T])
    return Nk


def get_lattice_Ls(cell, nimgs=None, rcut=None, dimension=None):
    '''Get the (Cartesian, unitful) lattice translation vectors for nearby images.
    The translation vectors can be used for the lattice summation.'''
    a = cell.lattice_vectors()
    b = cell.reciprocal_vectors(norm_to=1)
    heights_inv = lib.norm(b, axis=1)

    if nimgs is None:
        if rcut is None:
            rcut = cell.rcut
# plus 1 image in rcut to handle the case atoms within the adjacent cells are
# close to each other
        nimgs = np.ceil(rcut*heights_inv + 1.1).astype(int)
    else:
        rcut = max((np.asarray(nimgs))/heights_inv)

    if dimension is None:
        dimension = cell.dimension
    if dimension == 0:
        nimgs = [0, 0, 0]
    elif dimension == 1:
        nimgs = [nimgs[0], 0, 0]
    elif dimension == 2:
        nimgs = [nimgs[0], nimgs[1], 0]

    Ts = lib.cartesian_prod((np.arange(-nimgs[0],nimgs[0]+1),
                             np.arange(-nimgs[1],nimgs[1]+1),
                             np.arange(-nimgs[2],nimgs[2]+1)))
    Ls = np.dot(Ts, a)
    idx = np.zeros(len(Ls), dtype=bool)
    for ax in (-a[0], 0, a[0]):
        for ay in (-a[1], 0, a[1]):
            for az in (-a[2], 0, a[2]):
                idx |= lib.norm(Ls+(ax+ay+az), axis=1) < rcut
    Ls = Ls[idx]
    return np.asarray(Ls, order='C')


def super_cell(cell, ncopy):
    '''Create an ncopy[0] x ncopy[1] x ncopy[2] supercell of the input cell
    Note this function differs from :fun:`cell_plus_imgs` that cell_plus_imgs
    creates images in both +/- direction.

    Args:
        cell : instance of :class:`Cell`
        ncopy : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    supcell = cell.copy()
    a = cell.lattice_vectors()
    #:supcell.atom = []
    #:for Lx in range(ncopy[0]):
    #:    for Ly in range(ncopy[1]):
    #:        for Lz in range(ncopy[2]):
    #:            # Using cell._atom guarantees coord is in Bohr
    #:            for atom, coord in cell._atom:
    #:                L = np.dot([Lx, Ly, Lz], a)
    #:                supcell.atom.append([atom, coord + L])
    Ts = lib.cartesian_prod((np.arange(ncopy[0]),
                             np.arange(ncopy[1]),
                             np.arange(ncopy[2])))
    Ls = np.dot(Ts, a)
    symbs = [atom[0] for atom in cell._atom] * len(Ls)
    coords = Ls.reshape(-1,1,3) + cell.atom_coords()
    supcell.atom = list(zip(symbs, coords.reshape(-1,3)))
    supcell.unit = 'B'
    supcell.a = np.einsum('i,ij->ij', ncopy, a)
    supcell.gs = np.array([ncopy[0]*cell.gs[0] + (ncopy[0]-1)//2,
                           ncopy[1]*cell.gs[1] + (ncopy[1]-1)//2,
                           ncopy[2]*cell.gs[2] + (ncopy[2]-1)//2])
    supcell.build(False, False, verbose=0)
    supcell.verbose = cell.verbose
    return supcell


def cell_plus_imgs(cell, nimgs):
    '''Create a supercell via nimgs[i] in each +/- direction, as in get_lattice_Ls().
    Note this function differs from :fun:`super_cell` that super_cell only
    stacks the images in + direction.

    Args:
        cell : instance of :class:`Cell`
        nimgs : (3,) array

    Returns:
        supcell : instance of :class:`Cell`
    '''
    supcell = cell.copy()
    a = cell.lattice_vectors()
    Ts = lib.cartesian_prod((np.arange(-nimgs[0],nimgs[0]+1),
                             np.arange(-nimgs[1],nimgs[1]+1),
                             np.arange(-nimgs[2],nimgs[2]+1)))
    Ls = np.dot(Ts, a)
    symbs = [atom[0] for atom in cell._atom] * len(Ls)
    coords = Ls.reshape(-1,1,3) + cell.atom_coords()
    supcell.atom = list(zip(symbs, coords.reshape(-1,3)))
    supcell.unit = 'B'
    supcell.a = np.einsum('i,ij->ij', nimgs, a)
    supcell.build(False, False, verbose=0)
    supcell.verbose = cell.verbose
    return supcell


def cutoff_to_gs(a, cutoff):
    '''
    Convert KE cutoff to #grid points (gs variable) for FFT-mesh

        uses KE = k^2 / 2, where k_max ~ \pi / grid_spacing

    Args:
        a : (3,3) ndarray
            The real-space unit cell lattice vectors. Each row represents a
            lattice vector.
        cutoff : float
            KE energy cutoff in a.u.

    Returns:
        gs : (3,) array
    '''
#    grid_spacing = np.pi / np.sqrt(2 * cutoff)
#
#    # number of grid points is 2gs+1 (~ 2 gs) along each direction
#    gs = np.ceil(lib.norm(a, axis=1) / (2*grid_spacing)).astype(int)

    b = 2 * np.pi * np.linalg.inv(a.T)
    gs = np.ceil(np.sqrt(2*cutoff)/lib.norm(b, axis=1)).astype(int)
    return gs

def gs_to_cutoff(a, gs):
    '''
    Convert #grid points to KE cutoff
    '''
    b = 2 * np.pi * np.linalg.inv(a.T)
    Gmax = lib.norm(b, axis=1) * np.asarray(gs)
    return Gmax**2/2
