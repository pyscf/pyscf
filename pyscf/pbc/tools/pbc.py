#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import get_kconserv, get_kconserv3
from pyscf import __config__

FFT_ENGINE = getattr(__config__, 'pbc_tools_pbc_fft_engine', 'BLAS')

def _fftn_blas(f, mesh):
    Gx = np.fft.fftfreq(mesh[0])
    Gy = np.fft.fftfreq(mesh[1])
    Gz = np.fft.fftfreq(mesh[2])
    expRGx = np.exp(np.einsum('x,k->xk', -2j*np.pi*np.arange(mesh[0]), Gx))
    expRGy = np.exp(np.einsum('x,k->xk', -2j*np.pi*np.arange(mesh[1]), Gy))
    expRGz = np.exp(np.einsum('x,k->xk', -2j*np.pi*np.arange(mesh[2]), Gz))
    g0 = g = lib.transpose(f.reshape(-1, mesh[1]*mesh[2])).astype(np.complex128)
    g1 = np.empty_like(g0)
    g = lib.dot(g.reshape(-1,mesh[0])  , expRGx, c=g1.reshape(-1,mesh[0]))
    g = lib.dot(g.reshape(mesh[1],-1).T, expRGy, c=g0.reshape(-1,mesh[1]))
    g = lib.dot(g.reshape(mesh[2],-1).T, expRGz, c=g1.reshape(-1,mesh[2]))
    return g.reshape(-1, *mesh)

def _ifftn_blas(g, mesh):
    Gx = np.fft.fftfreq(mesh[0])
    Gy = np.fft.fftfreq(mesh[1])
    Gz = np.fft.fftfreq(mesh[2])
    expRGx = np.exp(np.einsum('x,k->xk', 2j*np.pi*np.arange(mesh[0]), Gx))
    expRGy = np.exp(np.einsum('x,k->xk', 2j*np.pi*np.arange(mesh[1]), Gy))
    expRGz = np.exp(np.einsum('x,k->xk', 2j*np.pi*np.arange(mesh[2]), Gz))
    f0 = f = lib.transpose(g.reshape(-1, mesh[1]*mesh[2])).astype(np.complex128)
    f1 = np.empty_like(f0)
    f = lib.dot(f.reshape(-1,mesh[0])  , expRGx, 1./mesh[0], c=f1.reshape(-1,mesh[0]))
    f = lib.dot(f.reshape(mesh[1],-1).T, expRGy, 1./mesh[1], c=f0.reshape(-1,mesh[1]))
    f = lib.dot(f.reshape(mesh[2],-1).T, expRGz, 1./mesh[2], c=f1.reshape(-1,mesh[2]))
    return f.reshape(-1, *mesh)

if FFT_ENGINE == 'FFTW':
    # pyfftw is slower than np.fft in most cases
    try:
        import pyfftw
        pyfftw.interfaces.cache.enable()
        nproc = lib.num_threads()
        def _fftn_wrapper(a):
            return pyfftw.interfaces.numpy_fft.fftn(a, axes=(1,2,3), threads=nproc)
        def _ifftn_wrapper(a):
            return pyfftw.interfaces.numpy_fft.ifftn(a, axes=(1,2,3), threads=nproc)
    except ImportError:
        def _fftn_wrapper(a):
            return np.fft.fftn(a, axes=(1,2,3))
        def _ifftn_wrapper(a):
            return np.fft.ifftn(a, axes=(1,2,3))

elif FFT_ENGINE == 'NUMPY':
    def _fftn_wrapper(a):
        return np.fft.fftn(a, axes=(1,2,3))
    def _ifftn_wrapper(a):
        return np.fft.ifftn(a, axes=(1,2,3))

elif FFT_ENGINE == 'NUMPY+BLAS':
    _EXCLUDE = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97,101,103,107,109,113,127,131,137,139,149,151,157,163,
                167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,
                257,263,269,271,277,281,283,293]
    _EXCLUDE = set(_EXCLUDE + [n*2 for n in _EXCLUDE] + [n*3 for n in _EXCLUDE])
    def _fftn_wrapper(a):
        mesh = a.shape[1:]
        if mesh[0] in _EXCLUDE and mesh[1] in _EXCLUDE and mesh[2] in _EXCLUDE:
            return _fftn_blas(a, mesh)
        else:
            return np.fft.fftn(a, axes=(1,2,3))
    def _ifftn_wrapper(a):
        mesh = a.shape[1:]
        if mesh[0] in _EXCLUDE and mesh[1] in _EXCLUDE and mesh[2] in _EXCLUDE:
            return _ifftn_blas(a, mesh)
        else:
            return np.fft.ifftn(a, axes=(1,2,3))

#?elif:  # 'FFTW+BLAS'
else:  # 'BLAS'
    def _fftn_wrapper(a):
        mesh = a.shape[1:]
        return _fftn_blas(a, mesh)
    def _ifftn_wrapper(a):
        mesh = a.shape[1:]
        return _ifftn_blas(a, mesh)


def fft(f, mesh):
    '''Perform the 3D FFT from real (R) to reciprocal (G) space.

    After FFT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv.

    FFT normalization factor is 1., as in MH and in `numpy.fft`.

    Args:
        f : (nx*ny*nz,) ndarray
            The function to be FFT'd, flattened to a 1D array corresponding
            to the index order of :func:`cartesian_prod`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The FFT 1D array in same index order as Gv (natural order of
            numpy.fft).

    '''
    if f.size == 0:
        return np.zeros_like(f)

    f3d = f.reshape(-1, *mesh)
    assert(f3d.shape[0] == 1 or f[0].size == f3d[0].size)
    g3d = _fftn_wrapper(f3d)
    ngrids = np.prod(mesh)
    if f.ndim == 1 or (f.ndim == 3 and f.size == ngrids):
        return g3d.ravel()
    else:
        return g3d.reshape(-1, ngrids)

def ifft(g, mesh):
    '''Perform the 3D inverse FFT from reciprocal (G) space to real (R) space.

    Inverse FFT normalization factor is 1./N, same as in `numpy.fft` but
    **different** from MH (they use 1.).

    Args:
        g : (nx*ny*nz,) ndarray
            The function to be inverse FFT'd, flattened to a 1D array
            corresponding to the index order of `span3`.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    Returns:
        (nx*ny*nz,) ndarray
            The inverse FFT 1D array in same index order as Gv (natural order
            of numpy.fft).

    '''
    if g.size == 0:
        return np.zeros_like(g)

    g3d = g.reshape(-1, *mesh)
    assert(g3d.shape[0] == 1 or g[0].size == g3d[0].size)
    f3d = _ifftn_wrapper(g3d)
    ngrids = np.prod(mesh)
    if g.ndim == 1 or (g.ndim == 3 and g.size == ngrids):
        return f3d.ravel()
    else:
        return f3d.reshape(-1, ngrids)


def fftk(f, mesh, expmikr):
    '''Perform the 3D FFT of a real-space function which is (periodic*e^{ikr}).

    fk(k+G) = \sum_r fk(r) e^{-i(k+G)r} = \sum_r [f(k)e^{-ikr}] e^{-iGr}
    '''
    return fft(f*expmikr, mesh)


def ifftk(g, mesh, expikr):
    '''Perform the 3D inverse FFT of f(k+G) into a function which is (periodic*e^{ikr}).

    fk(r) = (1/Ng) \sum_G fk(k+G) e^{i(k+G)r} = (1/Ng) \sum_G [fk(k+G)e^{iGr}] e^{ikr}
    '''
    return ifft(g, mesh) * expikr


def get_coulG(cell, k=np.zeros(3), exx=False, mf=None, mesh=None, Gv=None,
              wrap_around=True, low_dim_ft_type=None, **kwargs):
    '''Calculate the Coulomb kernel for all G-vectors, handling G=0 and exchange.

    Args:
        k : (3,) ndarray
            k-point
        exx : bool or str
            Whether this is an exchange matrix element.
        mf : instance of :class:`SCF`

    Returns:
        coulG : (ngrids,) ndarray
            The Coulomb kernel.
        mesh : (3,) ndarray of ints (= nx,ny,nz)
            The number G-vectors along each direction.

    '''
    exxdiv = exx
    if isinstance(exx, str):
        exxdiv = exx
    elif exx and mf is not None:
# sys.stderr.write('pass exxdiv directly')
        exxdiv = mf.exxdiv

    if mesh is None:
        mesh = cell.mesh
    if 'gs' in kwargs:
        warnings.warn('cell.gs is deprecated.  It is replaced by cell.mesh,'
                      'the number of PWs (=2*gs+1) along each direction.')
        mesh = [2*n+1 for n in kwargs['gs']]
    if Gv is None:
        Gv = cell.get_Gv(mesh)

    kG = k + Gv
    equal2boundary = np.zeros(Gv.shape[0], dtype=bool)
    if wrap_around and abs(k).sum() > 1e-9:
        # Here we 'wrap around' the high frequency k+G vectors into their lower
        # frequency counterparts.  Important if you want the gamma point and k-point
        # answers to agree
        b = cell.reciprocal_vectors()
        box_edge = np.einsum('i,ij->ij', np.asarray(mesh)//2+0.5, b)
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
        if cell.dimension == 3 or low_dim_ft_type is None:
            with np.errstate(divide='ignore'):
                coulG = 4*np.pi/absG2
            coulG[absG2==0] = 0
        elif cell.dimension == 2 and low_dim_ft_type == 'analytic_2d_1':
            # The following 2D analytical fourier transform is taken from:
            # R. Sundararaman and T. Arias PRB 87, 2013
            b = cell.reciprocal_vectors()
            Ld2 = np.pi/np.linalg.norm(b[2])
            Gz = kG[:,2]
            Gp = np.linalg.norm(kG[:,:2], axis=1)
            weights = 1. - np.cos(Gz*Ld2) * np.exp(-Gp*Ld2)
            with np.errstate(divide='ignore', invalid='ignore'):
                coulG = weights*4*np.pi/absG2
            coulG[absG2==0] = -2*np.pi*(np.pi/np.linalg.norm(b[2]))**2 #-pi*L_z^2/2
        else:
            raise NotImplementedError('no method for PBC dimension %s, '
                'dim-type %s and exxdiv = %s' %
                (cell.dimension, low_dim_ft_type, exxdiv))
    elif exxdiv == 'vcut_sph':  # PRB 77 193110
        Rc = (3*Nk*cell.vol/(4*np.pi))**(1./3)
        with np.errstate(divide='ignore',invalid='ignore'):
            coulG = 4*np.pi/absG2*(1.0 - np.cos(np.sqrt(absG2)*Rc))
        coulG[absG2==0] = 4*np.pi*0.5*Rc**2
    elif exxdiv == 'ewald':
        G0_idx = np.where(absG2==0)[0]
        if cell.dimension == 3 or low_dim_ft_type is None:
            with np.errstate(divide='ignore'):
                coulG = 4*np.pi/absG2
            if len(G0_idx) > 0:
                coulG[G0_idx] = Nk*cell.vol*madelung(cell, kpts)
        elif cell.dimension == 2 and low_dim_ft_type == 'analytic_2d_1':
            b = cell.reciprocal_vectors()
            Ld2 = np.pi/np.linalg.norm(b[2])
            Gz = kG[:,2]
            Gp = np.linalg.norm(kG[:,:2], axis=1)
            weights = 1. - np.cos(Gz*Ld2) * np.exp(-Gp*Ld2)
            with np.errstate(divide='ignore', invalid='ignore'):
                coulG = weights*4*np.pi/absG2
            if len(G0_idx) > 0:
                coulG[G0_idx] = -2*np.pi*(np.pi/
                                np.linalg.norm(b[2]))**2 #-pi*L_z^2/2
                coulG[G0_idx] += Nk*cell.vol*madelung(cell, kpts)
        else:
            raise NotImplementedError('no method for PBC dimension %s, '
                'dim-type %s and exxdiv = %s' %
                (cell.dimension, low_dim_ft_type, exxdiv))
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
        mesh = np.asarray(exx_kcell.mesh)
        gxyz = (gxyz + mesh)%mesh
        qidx = (gxyz[:,0]*mesh[1] + gxyz[:,1])*mesh[2] + gxyz[:,2]
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
    kcell.mesh = np.array([4*int(L*alpha*3.0) for L in Lc])  # ~ [60,60,60]
    # QE:
    #alpha = 3./Rin * np.sqrt(0.5)
    #kcell.mesh = (4*alpha*np.linalg.norm(kcell.a,axis=1)).astype(int)
    log.debug("# kcell.mesh FFT = %s", kcell.mesh)
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
    vG = (kcell.vol/kngs) * fft(vR, kcell.mesh)
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
    ecell.mesh = np.asarray(cell.mesh) * Nk
    ew_eta, ew_cut = ecell.get_ewald_params(cell.precision, ecell.mesh)
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
    supcell.mesh = np.array([ncopy[0]*cell.mesh[0],
                             ncopy[1]*cell.mesh[1],
                             ncopy[2]*cell.mesh[2]])
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


def cutoff_to_mesh(a, cutoff):
    '''
    Convert KE cutoff to FFT-mesh

        uses KE = k^2 / 2, where k_max ~ \pi / grid_spacing

    Args:
        a : (3,3) ndarray
            The real-space unit cell lattice vectors. Each row represents a
            lattice vector.
        cutoff : float
            KE energy cutoff in a.u.

    Returns:
        mesh : (3,) array
    '''
    b = 2 * np.pi * np.linalg.inv(a.T)
    mesh = np.ceil(np.sqrt(2*cutoff)/lib.norm(b, axis=1) * 2).astype(int)
    return mesh

def mesh_to_cutoff(a, mesh):
    '''
    Convert #grid points to KE cutoff
    '''
    b = 2 * np.pi * np.linalg.inv(a.T)
    Gmax = lib.norm(b, axis=1) * np.asarray(mesh) * .5
    return Gmax**2/2

def cutoff_to_gs(a, cutoff):
    '''Deprecated.  Replaced by function cutoff_to_mesh.'''
    return [n//2 for n in cutoff_to_mesh(a, cutoff)]

def gs_to_cutoff(a, gs):
    '''Deprecated.  Replaced by function mesh_to_cutoff.'''
    return mesh_to_cutoff(a, [2*n+1 for n in gs])
