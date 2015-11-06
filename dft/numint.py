#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import _ctypes
import time
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.dft.vxc

libdft = pyscf.lib.load_library('libdft')
OCCDROP = 1e-12
BLKSIZE = 224

def eval_ao(mol, coords, isgga=False, relativity=0, bastart=0, bascount=None,
            non0tab=None, verbose=None):
    '''Evaluate AO function value on the given grids, for LDA and GGA functional.

    Args:
        mol : an instance of :class:`Mole`

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        isgga : bool
            Whether to evalute the AO gradients for GGA functional.  It affects
            the shape of the return array.  If isgga=False,  the returned AO
            values are stored in a (N,nao) array.  Otherwise the AO values are
            stored in an array of shape (4,N,nao).  Here N is the number of
            grids, nao is the number of AO functions.
        relativity : bool
            No effects.
        bastart, bascount : int
            If given, only part of AOs (bastart <= shell_id < bastart+bascount) are evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        2D array of shape (N,nao) for AO values if isgga is False.
        Or 3D array of shape (4,N,nao) for AO values and AO gradients if isgga is True.
        In the 3D array, the first (N,nao) elements are the AO values.  The
        following (3,N,nao) are the AO gradients for x,y,z compoents.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords)
    >>> print(ao_value.shape)
    (100, 24)
    >>> ao_value = eval_ao(mol, coords, isgga=True, bastart=1, bascount=3)
    >>> print(ao_value.shape)
    (4, 100, 7)
    '''
    assert(coords.flags.c_contiguous)
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    ngrids = len(coords)
    if bascount is None:
        bascount = mol.nbas - bastart
        nao = mol.nao_nr()
    else:
        nao_bound = mol.nao_nr_range(bastart, bastart+bascount)
        nao = nao_bound[1] - nao_bound[0]
    if isgga:
        ao = numpy.empty((4, ngrids,nao)) # plain, dx, dy, dz
        feval = _ctypes.dlsym(libdft._handle, 'VXCeval_nr_gto_grad')
    else:
        ao = numpy.empty((ngrids,nao))
        feval = _ctypes.dlsym(libdft._handle, 'VXCeval_nr_gto')

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)

    libdft.VXCeval_ao_drv(ctypes.c_void_p(feval),
                          ctypes.c_int(nao), ctypes.c_int(ngrids),
                          ctypes.c_int(bastart), ctypes.c_int(bascount),
                          ctypes.c_int(BLKSIZE),
                          ao.ctypes.data_as(ctypes.c_void_p),
                          coords.ctypes.data_as(ctypes.c_void_p),
                          non0tab.ctypes.data_as(ctypes.c_void_p),
                          mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                          mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                          mol._env.ctypes.data_as(ctypes.c_void_p))
    return ao

def make_mask(mol, coords, relativity=0, bastart=0, bascount=None,
              verbose=None):
    '''Mask to indicate whether a shell is zero on particular grid

    Args:
        mol : an instance of :class:`Mole`

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        relativity : bool
            No effects.
        bastart, bascount : int
            If given, only part of AOs (bastart <= shell_id < bastart+bascount) are evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        2D bool array of shape (N,nbas), where N is the number of grids, nbas
        is the number of shells
    '''
    assert(coords.flags.c_contiguous)
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    ngrids = len(coords)
    if bascount is None:
        bascount = mol.nbas - bastart

    non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,bascount),
                          dtype=numpy.int8)
    libdft.VXCnr_ao_screen(non0tab.ctypes.data_as(ctypes.c_void_p),
                           coords.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                           mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                           mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                           mol._env.ctypes.data_as(ctypes.c_void_p))
    return non0tab

def eval_rho(mol, ao, dm, non0tab=None, isgga=False, verbose=None):
    '''Calculate the electron density for LDA functional, and the density
    derivatives for GGA functional.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA or 3D array of shape (4,N,nao) for GGA
            N is the number of grids, nao is the number of AO functions.
            AO values on a set of grids.  If isgga is True, the AO values
            need to be 3D array in which ao[0] is the AO values and ao[1:3]
            are the AO gradients.
        dm : 2D array
            Density matrix

    Kwargs:
        isgga : bool
            Whether to evalute the AO gradients for GGA functional.  It affects
            the shape of the argument `ao` and the returned density.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if isgga is False.  2D
        array of (4,N) to store density and "density derivatives" for x,y,z
        components if isgga is True.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords, isgga=True)
    >>> dm = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> dm = dm + dm.T
    >>> rho, dx_rho, dy_rho, dz_rho = eval_rho(mol, ao, dm, isgga=True)
    '''
    assert(ao.flags.c_contiguous)
    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    if isgga:
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, nao, ngrids, non0tab)
        rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], dm, nao, ngrids, non0tab)
            rho[i] = numpy.einsum('pi,pi->p', ao[0], c1) * 2 # *2 for +c.c.
    else:
        c0 = _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab)
        rho = numpy.einsum('pi,pi->p', ao, c0)
    return rho

def eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab=None, isgga=False,
              verbose=None):
    '''Calculate the electron density for LDA functional, and the density
    derivatives for GGA functional.  This function has the same functionality
    as :func:`eval_rho` except that the density are evaluated based on orbital
    coefficients and orbital occupancy.  It is more efficient than
    :func:`eval_rho` in most scenario.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA or 3D array of shape (4,N,nao) for GGA
            N is the number of grids, nao is the number of AO functions.
            AO values on a set of grids.  If isgga is True, the AO values
            need to be 3D array in which ao[0] is the AO values and ao[1:3]
            are the AO gradients.
        mo_coeff : 2D array
            Orbital coefficients
        mo_occ : 2D array
            Orbital occupancy

    Kwargs:
        isgga : bool
            Whether to evalute the AO gradients for GGA functional.  It affects
            the shape of the argument `ao` and the returned density.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if isgga is False.  2D
        array of (4,N) to store density and "density derivatives" for x,y,z
        components if isgga is True.
    '''
    assert(ao.flags.c_contiguous)
    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    pos = mo_occ > OCCDROP
    cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
    if pos.sum() > 0:
        if isgga:
            rho = numpy.empty((4,ngrids))
            c0 = _dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
        else:
            c0 = _dot_ao_dm(mol, ao, cpos, nao, ngrids, non0tab)
            rho = numpy.einsum('pi,pi->p', c0, c0)
    else:
        if isgga:
            rho = numpy.zeros((4,ngrids))
        else:
            rho = numpy.zeros(ngrids)

    neg = mo_occ < -OCCDROP
    if neg.sum() > 0:
        cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
        if isgga:
            c0 = _dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
        else:
            c0 = _dot_ao_dm(mol, ao, cneg, nao, ngrids, non0tab)
            rho -= numpy.einsum('pi,pi->p', c0, c0)
    return rho

def eval_mat(mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
             isgga=False, verbose=None):
    '''Calculate XC potential matrix.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA or 3D array of shape (4,N,nao) for GGA
            N is the number of grids, nao is the number of AO functions.
            AO values on a set of grids.  If isgga is True, the AO values
            need to be 3D array in which ao[0] is the AO values and ao[1:3]
            are the AO gradients.
        weight : 1D array
            Integral weights on grids.
        rho : 1D array of size N for LDA or 2D array of shape (4,N) for GGA
            electron density on each grid.  If isgga is True, it also stores
            the density derivatives.
        vrho : 1D array of size N
            XC potential value on each grid.

    Kwargs:
        vsigma : 2D array of shape (3,N)
            GGA potential value on each grid
        isgga : bool
            Whether to evalute the AO gradients for GGA functional.  It affects
            the shape of the argument `ao` and `rho`.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        XC potential matrix in 2D array of shape (nao,nao) where nao is the
        number of AO functions.
    '''
    assert(ao.flags.c_contiguous)
    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    if isgga:
        assert(vsigma is not None and rho.ndim==2)
        #wv = weight * vsigma * 2
        #aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
        wv = numpy.empty_like(rho)
        wv[0]  = weight * vrho * .5
        wv[1:] = rho[1:] * (weight * vsigma * 2)
        aow = numpy.einsum('npi,np->pi', ao, wv)
        #mat = pyscf.lib.dot(ao[0].T, aow)
        mat = _dot_ao_ao(mol, ao[0], aow, nao, ngrids, non0tab)
    else:
        # *.5 because return mat + mat.T
        #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = ao * (.5*weight*vrho).reshape(-1,1)
        #mat = pyscf.lib.dot(ao.T, aow)
        mat = _dot_ao_ao(mol, ao, aow, nao, ngrids, non0tab)
    return mat + mat.T

def eval_x(x_id, rho, sigma, spin=0, relativity=0, verbose=None):
    '''Interface to call libxc library to evaluate exchange functional and potential.

    Args:
        x_id : int
            Exchange functional ID used by libxc library.  See pyscf/dft/vxc.py for more details.
        rho : 1D array or 2D array
            Shape of (N) for electron density if spin = 0;
            Shape of (N,2) for alpha electron density and beta density if spin = 1
            where N is number of grids
        sigma : 1D array or 2D array
            (Density derivatives)^2.
            Shape of (N) if spin = 0;
            Shape of (N,3) for alpha*alpha, alpha*beta, beta*beta components if spin = 1
            where N is the number of grids

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        3 1D arrays ex, vrho, vsigma  where exc is the exchange functional
        value on each grid, vrho is exchange potential on each grid, vsigma is
        the derivative potential for GGA
    '''
    rho = numpy.asarray(rho, order='C')
    sigma = numpy.asarray(sigma, order='C')
    ngrids = len(rho)
    if spin == 0:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty(ngrids)
        vsigma = numpy.empty(ngrids)
        nspin = 1
    else:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty((ngrids,2))
        vsigma = numpy.empty((ngrids,3))
        nspin = 2
    libdft.VXCnr_eval_x(ctypes.c_int(x_id),
                        ctypes.c_int(nspin), ctypes.c_int(relativity),
                        ctypes.c_int(ngrids),
                        rho.ctypes.data_as(ctypes.c_void_p),
                        sigma.ctypes.data_as(ctypes.c_void_p),
                        exc.ctypes.data_as(ctypes.c_void_p),
                        vrho.ctypes.data_as(ctypes.c_void_p),
                        vsigma.ctypes.data_as(ctypes.c_void_p))
    return exc, vrho, vsigma

def eval_c(c_id, rho, sigma, spin=0, relativity=0, verbose=None):
    '''Interface to call libxc library to evaluate correlation functional and potential.
    For hybrid functional, the returned ec, vcrho, vcsigma are all zero.

    Args:
        c_id : int
            Correlation functional ID used by libxc library.  See pyscf/dft/vxc.py for more details.
        rho : 1D array or 2D array
            Shape of (N) for electron density if spin = 0;
            Shape of (N,2) for alpha electron density and beta density if spin = 1
            where N is number of grids
        sigma : 1D array or 2D array
            (Density derivatives)^2.
            Shape of (N) if spin = 0;
            Shape of (N,3) for alpha*alpha, alpha*beta, beta*beta components if spin = 1
            where N is the number of grids

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        3 1D arrays ec, vrho, vsigma  where ec is the correlation functional
        value on each grid, vrho is correlation potential on each grid, vsigma
        is the derivative potential for GGA
    '''
    rho = numpy.asarray(rho, order='C')
    sigma = numpy.asarray(sigma, order='C')
    ngrids = len(rho)
    if spin == 0:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty(ngrids)
        vsigma = numpy.empty(ngrids)
        nspin = 1
    else:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty((ngrids,2))
        vsigma = numpy.empty((ngrids,3))
        nspin = 2
    libdft.VXCnr_eval_c(ctypes.c_int(c_id),
                        ctypes.c_int(nspin), ctypes.c_int(relativity),
                        ctypes.c_int(ngrids),
                        rho.ctypes.data_as(ctypes.c_void_p),
                        sigma.ctypes.data_as(ctypes.c_void_p),
                        exc.ctypes.data_as(ctypes.c_void_p),
                        vrho.ctypes.data_as(ctypes.c_void_p),
                        vsigma.ctypes.data_as(ctypes.c_void_p))
    return exc, vrho, vsigma

def eval_xc(x_id, c_id, rho, sigma, spin=0, relativity=0, verbose=None):
    '''Interface to call libxc library to evaluate XC functional and potential.

    Args:
        x_id, c_id : int
            Exchange/Correlation functional ID used by libxc library.
            See pyscf/dft/vxc.py for more details.
        rho : 1D array or 2D array
            Shape of (N) for electron density if spin = 0;
            Shape of (N,2) for alpha electron density and beta density if spin = 1
            where N is number of grids
        sigma : 1D array or 2D array
            (Density derivatives)^2.
            Shape of (N) if spin = 0;
            Shape of (N,3) for alpha*alpha, alpha*beta, beta*beta components if spin = 1
            where N is the number of grids

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        3 1D arrays ec, vrho, vsigma  where ec is the correlation functional
        value on each grid, vrho is correlation potential on each grid, vsigma
        is the derivative potential for GGA
    '''
    rho = numpy.asarray(rho, order='C')
    sigma = numpy.asarray(sigma, order='C')
    ngrids = len(rho)
    if spin == 0:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty(ngrids)
        vsigma = numpy.empty(ngrids)
        nspin = 1
    else:
        exc = numpy.empty(ngrids)
        vrho = numpy.empty((ngrids,2))
        vsigma = numpy.empty((ngrids,3))
        nspin = 2
    libdft.VXCnr_eval_xc(ctypes.c_int(x_id), ctypes.c_int(c_id),
                         ctypes.c_int(nspin), ctypes.c_int(relativity),
                         ctypes.c_int(ngrids),
                         rho.ctypes.data_as(ctypes.c_void_p),
                         sigma.ctypes.data_as(ctypes.c_void_p),
                         exc.ctypes.data_as(ctypes.c_void_p),
                         vrho.ctypes.data_as(ctypes.c_void_p),
                         vsigma.ctypes.data_as(ctypes.c_void_p))
    return exc, vrho, vsigma


def _dot_ao_ao(mol, ao1, ao2, nao, ngrids, non0tab):
    '''return pyscf.lib.dot(ao1.T, ao2)'''
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    ao1 = numpy.asarray(ao1, order='C')
    ao2 = numpy.asarray(ao2, order='C')
    vv = numpy.empty((nao,nao))
    libdft.VXCdot_ao_ao(vv.ctypes.data_as(ctypes.c_void_p),
                        ao1.ctypes.data_as(ctypes.c_void_p),
                        ao2.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(ngrids),
                        ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vv

def _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab):
    '''return pyscf.lib.dot(ao, dm)'''
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    vm = numpy.empty((ngrids,dm.shape[1]))
    ao = numpy.asarray(ao, order='C')
    dm = numpy.asarray(dm, order='C')
    libdft.VXCdot_ao_dm(vm.ctypes.data_as(ctypes.c_void_p),
                        ao.ctypes.data_as(ctypes.c_void_p),
                        dm.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
                        ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vm

def nr_vxc(mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix for given meshgrids and density matrix

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        x_id, c_id : int
            Exchange/Correlation functional ID used by libxc library.
            See pyscf/dft/vxc.py for more details.
        dm : 2D array
            Density matrix

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        hermi : int
            No effects
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> grids.coords = numpy.random.random((100,3))  # 100 random points
    >>> grids.weights = numpy.random.random(100)
    >>> x_id, c_id = dft.vxc.parse_xc_name('lda,vwn')
    >>> dm = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> nelec, exc, vxc = dft.numint.nr_vxc(mol, grids, x_id, c_id, dm)
    '''
    nao = dm.shape[0]
    ngrids = len(grids.weights)
    blksize = min(int(max_memory/6*1e6/8/nao), ngrids)
    nelec = 0
    excsum = 0
    vmat = numpy.zeros_like(dm)
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ngrids, ip0+blksize)
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        if pyscf.dft.vxc._is_lda(x_id) and pyscf.dft.vxc._is_lda(c_id):
            isgga = False
            ao = eval_ao(mol, coords, isgga=isgga)
            rho = eval_rho(mol, ao, dm, isgga=isgga)
            exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                        spin, relativity, verbose)
            den = rho*weight
            nelec += den.sum()
            excsum += (den*exc).sum()
        else:
            isgga = True
            ao = eval_ao(mol, coords, isgga=isgga)
            rho = eval_rho(mol, ao, dm, isgga=isgga)
            sigma = numpy.einsum('ip,ip->p', rho[1:], rho[1:])
            exc, vrho, vsigma = eval_xc(x_id, c_id, rho[0], sigma,
                                        spin, relativity, verbose)
            den = rho[0]*weight
            nelec += den.sum()
            excsum += (den*exc).sum()
        vmat += eval_mat(mol, ao, weight, rho, vrho, vsigma, isgga=isgga,
                         verbose=verbose)
    return nelec, excsum, vmat


class _NumInt(object):
    def __init__(self):
        self.non0tab = None

    def nr_vxc(self, mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix matrix for given meshgrids
        and a set of density matrices.  See :func:`nr_rks` and :func:`nr_uks`
        for more details.
        '''
        if spin == 0:
            return self.nr_rks(mol, grids, x_id, c_id, dm, hermi=hermi,
                               max_memory=max_memory, verbose=verbose)
        else:
            return self.nr_uks(mol, grids, x_id, c_id, dm, hermi=hermi,
                               max_memory=max_memory, verbose=verbose)

    def nr_rks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        '''Calculate RKS XC functional and potential matrix matrix for given meshgrids
        and a set of density matrices

        Args:
            mol : an instance of :class:`Mole`

            grids : an instance of :class:`Grids`
                grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
            x_id, c_id : int
                Exchange/Correlation functional ID used by libxc library.
                See pyscf/dft/vxc.py for more details.
            dm : 2D array a list of 2D arrays
                Density matrix or multiple density matrices

        Kwargs:
            hermi : int
                No effects
            max_memory : int or float
                The maximum size of cache to use (in MB).
            verbose : int or object of :class:`Logger`

        Returns:
            nelec, excsum, vmat.
            nelec is the number of electrons generated by numerical integration.
            excsum is the XC functional value.  vmat is the XC potential matrix in
            2D array of shape (nao,nao) where nao is the number of AO functions.
        '''
        if self.non0tab is None:
            self.non0tab = make_mask(mol, grids.coords)
        nao = mol.nao_nr()
        ngrids = len(grids.weights)
# NOTE to index self.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
        if pyscf.dft.vxc._is_lda(x_id) and pyscf.dft.vxc._is_lda(c_id):
            isgga = False
        else:
            isgga = True

        natocc = []
        natorb = []
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            e, c = scipy.linalg.eigh(dms)
            natocc.append(e)
            natorb.append(c)
        else:
            for dm in dms:
                e, c = scipy.linalg.eigh(dm)
                natocc.append(e)
                natorb.append(c)
        nset = len(natocc)
        nelec = numpy.zeros(nset)
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((nset,nao,nao))
        for ip0, ip1 in prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = self.non0tab[ip0//BLKSIZE:]
            ao = eval_ao(mol, coords, isgga=isgga, non0tab=non0tab)
            for idm in range(nset):
                rho = eval_rho2(mol, ao, natorb[idm], natocc[idm],
                                non0tab=non0tab, isgga=isgga)
                if isgga:
                    sigma = numpy.einsum('ip,ip->p', rho[1:], rho[1:])
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho[0], sigma,
                                                spin=0, verbose=verbose)
                    den = rho[0]*weight
                    nelec[idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
# ref eval_mat function
                    wv = numpy.empty_like(rho)
                    wv[0]  = weight * vrho * .5
                    wv[1:] = rho[1:] * (weight * vsigma * 2)
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0,
                                            non0tab)
                else:
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                                spin=0, verbose=verbose)
                    den = rho*weight
                    nelec[idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
                    aow = ao * (.5*weight*vrho).reshape(-1,1)
                    vmat[idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0,
                                            non0tab)
            wv = aow = None
        for i in range(nset):
            vmat[i] = vmat[i] + vmat[i].T
        if nset == 1:
            nelec = nelec[0]
            excsum = excsum[0]
            vmat = vmat.reshape(nao,nao)
        return nelec, excsum, vmat

    def nr_uks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        '''Calculate UKS XC functional and potential matrix matrix for given meshgrids
        and a set of density matrices

        Args:
            mol : an instance of :class:`Mole`

            grids : an instance of :class:`Grids`
                grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
            x_id, c_id : int
                Exchange/Correlation functional ID used by libxc library.
                See pyscf/dft/vxc.py for more details.
            dm : 2D array a list of 2D arrays
                Density matrix or multiple density matrices

        Kwargs:
            hermi : int
                No effects
            max_memory : int or float
                The maximum size of cache to use (in MB).
            verbose : int or object of :class:`Logger`

        Returns:
            nelec, excsum, vmat.
            nelec is the number of (alpha,beta) electrons generated by numerical integration.
            excsum is the XC functional value.
            vmat is the XC potential matrix for (alpha,beta) spin.
        '''
        if self.non0tab is None:
            self.non0tab = make_mask(mol, grids.coords)
        nao = mol.nao_nr()
        ngrids = len(grids.weights)
# NOTE to index self.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
        if pyscf.dft.vxc._is_lda(x_id) and pyscf.dft.vxc._is_lda(c_id):
            isgga = False
        else:
            isgga = True

        natocc = []
        natorb = []
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            e, c = scipy.linalg.eigh(dms)
            natocc.append((e*.5,e*.5))
            natorb.append((c,c))
            nset = 1
        else:
            nset = len(dms) // 2
            for idm in range(nset):
                e_a, c_a = scipy.linalg.eigh(dms[idm])
                e_b, c_b = scipy.linalg.eigh(dms[nset+idm])
                natocc.append((e_a,e_b))
                natorb.append((c_a,c_b))
        nelec = numpy.zeros((2,nset))
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((2,nset,nao,nao))
        for ip0, ip1 in prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = self.non0tab[ip0//BLKSIZE:]
            ao = eval_ao(mol, coords, isgga=isgga, non0tab=non0tab)
            for idm in range(nset):
                c_a, c_b = natorb[idm]
                e_a, e_b = natocc[idm]
                rho_a = eval_rho2(mol, ao, c_a, e_a, non0tab=non0tab, isgga=isgga)
                rho_b = eval_rho2(mol, ao, c_b, e_b, non0tab=non0tab, isgga=isgga)
                if isgga:
                    rho = numpy.hstack((rho_a[0].reshape(-1,1),
                                        rho_b[0].reshape(-1,1)))
                    sigma = numpy.empty((ip1-ip0,3))
                    sigma[:,0] = numpy.einsum('ip,ip->p', rho_a[1:], rho_a[1:])
                    sigma[:,1] = numpy.einsum('ip,ip->p', rho_a[1:], rho_b[1:])
                    sigma[:,2] = numpy.einsum('ip,ip->p', rho_b[1:], rho_b[1:])
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho, sigma,
                                                spin=1, verbose=verbose)
                    den = rho[:,0]*weight
                    nelec[0,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
                    den = rho[:,1]*weight
                    nelec[1,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()

                    wv = numpy.empty_like(rho_a)
                    wv[0]  = weight * vrho[:,0] * .5
                    wv[1:] = rho_a[1:] * (weight * vsigma[:,0] * 2)  # sigma_uu
                    wv[1:]+= rho_b[1:] * (weight * vsigma[:,1])      # sigma_ud
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0,
                                              non0tab)
                    wv[0]  = weight * vrho[:,1] * .5
                    wv[1:] = rho_b[1:] * (weight * vsigma[:,2] * 2)  # sigma_dd
                    wv[1:]+= rho_a[1:] * (weight * vsigma[:,1])      # sigma_ud
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0,
                                              non0tab)

                else:
                    rho = numpy.hstack((rho_a[:,None],rho_b[:,None]))
                    exc, vrho, vsigma = eval_xc(x_id, c_id, rho, rho,
                                                spin=1, verbose=verbose)
                    den = rho[:,0]*weight
                    nelec[0,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
                    den = rho[:,1]*weight
                    nelec[1,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()

                    aow = ao * (.5*weight*vrho[:,0]).reshape(-1,1)
                    vmat[0,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0,
                                              non0tab)
                    aow = ao * (.5*weight*vrho[:,1]).reshape(-1,1)
                    vmat[1,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0,
                                              non0tab)
            wv = aow = None
        for i in range(nset):
            vmat[0,i] = vmat[0,i] + vmat[0,i].T
            vmat[1,i] = vmat[1,i] + vmat[1,i].T
        if nset == 1:
            nelec = nelec.reshape(2)
            excsum = excsum[0]
            vmat = vmat.reshape(2,nao,nao)
        return nelec, excsum, vmat

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        grids = {"H": (100, 194),
                 "O": (100, 194),},
        basis = '6311g*',)
    mf = dft.RKS(mol)
    mf.grids.setup_grids()
    dm = mf.get_init_guess(key='minao')

    x_code, c_code = pyscf.dft.vxc.parse_xc_name(mf.xc)
#res = vxc.nr_vxc(mol, mf.grids, x_code, c_code, dm, spin=1, relativity=0)
    print(time.clock())
    res = nr_vxc(mol, mf.grids, x_code, c_code, dm, spin=mol.spin, relativity=0)
    print(time.clock())

