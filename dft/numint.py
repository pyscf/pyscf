#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
try:
    from pyscf.dft import libxc
except (ImportError, OSError):
    from pyscf.dft import xcfun as libxc

libdft = pyscf.lib.load_library('libdft')
OCCDROP = 1e-12
BLKSIZE = 96

def eval_ao(mol, coords, deriv=0, relativity=0, bastart=0, bascount=None,
            non0tab=None, out=None, verbose=None):
    '''Evaluate AO function value on the given grids.

    Args:
        mol : an instance of :class:`Mole`

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        deriv : int
            AO derivative order.  It affects the shape of the return array.
            If deriv=0, the returned AO values are stored in a (N,nao) array.
            Otherwise the AO values are stored in an array of shape (M,N,nao).
            Here N is the number of grids, nao is the number of AO functions,
            M is the size associated to the derivative deriv.
        relativity : bool
            No effects.
        bastart, bascount : int
            If given, only part of AOs (bastart <= shell_id < bastart+bascount) are evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        out : ndarray
            If provided, results are written into this array.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        2D array of shape (N,nao) for AO values if deriv = 0.
        Or 3D array of shape (:,N,nao) for AO values and AO derivatives if deriv > 0.
        In the 3D array, the first (N,nao) elements are the AO values,
        followed by (3,N,nao) for x,y,z compoents;
        Then 2nd derivatives (6,N,nao) for xx, xy, xz, yy, yz, zz;
        Then 3rd derivatives (10,N,nao) for xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
        ...

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords)
    >>> print(ao_value.shape)
    (100, 24)
    >>> ao_value = eval_ao(mol, coords, deriv=1, bastart=1, bascount=3)
    >>> print(ao_value.shape)
    (4, 100, 7)
    >>> ao_value = eval_ao(mol, coords, deriv=2, bastart=1, bascount=3)
    >>> print(ao_value.shape)
    (10, 100, 7)
    '''
    assert(coords.flags.c_contiguous)
    if isinstance(deriv, bool):
        logger.warn(mol, '''
You see this error message because of the API updates in pyscf v1.1.
Argument "isgga" is replaced by argument "deriv", to support high order AO derivatives''')

    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_sph_deriv%d' % deriv
    return mol.eval_gto(feval, coords, comp, bastart, bascount, non0tab, out)

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

#TODO: \nabla^2 rho and tau = 1/2 (\nabla f)^2
def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', verbose=None):
    r'''Calculate the electron density for LDA functional, and the density
    derivatives for GGA functional.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or (5,N,nao) for meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA, ao[0] is AO value
            and ao[1:3] are the AO gradients.  If xctype is meta-GGA, ao[4:10]
            are second derivatives of ao values.
        dm : 2D array
            Density matrix

    Kwargs:
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA;  (6,N) array for meta-GGA, where last two rows are
        \nabla^2 rho and tau = 1/2(\nabla f)^2

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords, deriv=0)
    >>> dm = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> dm = dm + dm.T
    >>> rho, dx_rho, dy_rho, dz_rho = eval_rho(mol, ao, dm, xctype='LDA')
    '''
    assert(ao.flags.c_contiguous)
    xctype = xctype.upper()
    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    if xctype == 'LDA':
        c0 = _dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab)
        rho = numpy.einsum('pi,pi->p', ao, c0)
    elif xctype == 'GGA':
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, nao, ngrids, non0tab)
        rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], dm, nao, ngrids, non0tab)
            rho[i] = numpy.einsum('pi,pi->p', ao[0], c1)
            rho[i] *= 2 # *2 for +c.c. in the next two lines
            #c1 = _dot_ao_dm(mol, ao[i], dm.T, nao, ngrids, non0tab)
            #rho[i] += numpy.einsum('pi,pi->p', c1, ao[0])
    else: # meta-GGA
        # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
        rho = numpy.empty((6,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, nao, ngrids, non0tab)
        rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        rho[5] = 0
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], dm, nao, ngrids, non0tab)
            rho[i] = numpy.einsum('pi,pi->p', ao[0], c1) * 2 # *2 for +c.c.
            rho[5] += numpy.einsum('pi,pi->p', c1, c1)
        XX, YY, ZZ = 4, 7, 9
        ao2 = ao[XX] + ao[YY] + ao[ZZ]
        rho[4] = numpy.einsum('pi,pi->p', ao[0], ao2)
        rho[4] += rho[5]
        rho[4] *= 2

        rho[5] *= .5
    return rho

def eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              verbose=None):
    r'''Calculate the electron density for LDA functional, and the density
    derivatives for GGA functional.  This function has the same functionality
    as :func:`eval_rho` except that the density are evaluated based on orbital
    coefficients and orbital occupancy.  It is more efficient than
    :func:`eval_rho` in most scenario.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or (5,N,nao) for meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA, ao[0] is AO value
            and ao[1:3] are the AO gradients.  If xctype is meta-GGA, ao[4:10]
            are second derivatives of ao values.
        dm : 2D array
            Density matrix

    Kwargs:
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA;  (6,N) array for meta-GGA, where last two rows are
        \nabla^2 rho and tau = 1/2(\nabla f)^2
    '''
    assert(ao.flags.c_contiguous)
    xctype = xctype.upper()
    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    pos = mo_occ > OCCDROP
    cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
    if pos.sum() > 0:
        if xctype == 'LDA':
            c0 = _dot_ao_dm(mol, ao, cpos, nao, ngrids, non0tab)
            rho = numpy.einsum('pi,pi->p', c0, c0)
        elif xctype == 'GGA':
            rho = numpy.empty((4,ngrids))
            c0 = _dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
        else: # meta-GGA
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            c0 = _dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            rho[5] = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                rho[5] += numpy.einsum('pi,pi->p', c1, c1)
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = _dot_ao_dm(mol, ao2, cpos, nao, ngrids, non0tab)
            rho[4] = numpy.einsum('pi,pi->p', c0, c1)
            rho[4] += rho[5]
            rho[4] *= 2

            rho[5] *= .5
    else:
        if xctype == 'LDA':
            rho = numpy.zeros(ngrids)
        elif xctype == 'GGA':
            rho = numpy.zeros((4,ngrids))
        else:
            rho = numpy.zeros((6,ngrids))

    neg = mo_occ < -OCCDROP
    if neg.sum() > 0:
        cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
        if xctype == 'LDA':
            c0 = _dot_ao_dm(mol, ao, cneg, nao, ngrids, non0tab)
            rho -= numpy.einsum('pi,pi->p', c0, c0)
        elif xctype == 'GGA':
            c0 = _dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
        else:
            c0 = _dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            rho5 = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                rho5 -= numpy.einsum('pi,pi->p', c1, c1)
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = _dot_ao_dm(mol, ao2, cneg, nao, ngrids, non0tab)
            rho[4] -= numpy.einsum('pi,pi->p', c0, c1) * 2
            rho[4] -= rho5 * 2

            rho[5] -= rho5 * .5
    return rho

def eval_mat(mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
             xctype='LDA', verbose=None):
    '''Calculate XC potential matrix.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or (5,N,nao) for meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA, ao[0] is AO value
            and ao[1:3] are the AO gradients.  If xctype is meta-GGA, ao[4:10]
            are second derivatives of ao values.
        weight : 1D array
            Integral weights on grids.
        rho : 1D array of size N for LDA or 2D array for GGA/meta-GGA,
            electron density (derivatives) on each grid.
        vrho : 1D array of size N
            XC potential value on each grid.

    Kwargs:
        vsigma : 2D array of shape (3,N)
            GGA potential value on each grid
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of `ao` and `rho`
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`

    Returns:
        XC potential matrix in 2D array of shape (nao,nao) where nao is the
        number of AO functions.
    '''
    assert(ao.flags.c_contiguous)
    xctype = xctype.upper()
    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    if xctype == 'LDA':
        # *.5 because return mat + mat.T
        #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = ao * (.5*weight*vrho).reshape(-1,1)
        #mat = pyscf.lib.dot(ao.T, aow)
        mat = _dot_ao_ao(mol, ao, aow, nao, ngrids, non0tab)
    elif xctype == 'GGA':
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
        raise NotImplementedError('meta-GGA')
    return mat + mat.T


def _dot_ao_ao(mol, ao1, ao2, nao, ngrids, non0tab):
    '''return numpy.dot(ao1.T, ao2)'''
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
    '''return numpy.dot(ao, dm)'''
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

def nr_vxc(mol, grids, xc_code, dm, spin=0, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    if isinstance(spin, (list, tuple, numpy.ndarray)):
# shift the old args (..., x_id, c_id, dm, spin, ..)
        import warnings
        xc_code = '%s, %s' % (xc_code, dm)
        dm, spin = spin, relativity
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn('API updates: the 4th argument c_id is depercated '
                          'and will be removed in future release.\n')
    ni = _NumInt()
    ni.non0tab = ni.make_mask(mol, grids.coords)
    if spin == 0:
        return nr_rks(ni, mol, grids, xc_code, dm, relativity,
                      hermi, max_memory, verbose)
    else:
        return nr_uks(ni, mol, grids, xc_code, dm, relativity,
                      hermi, max_memory, verbose)

def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix on given meshgrids
    for a set of density matrices

    Args:
        ni : an instance of :class:`_NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array a list of 2D arrays
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).

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
    >>> dm = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> nelec, exc, vxc = dft.numint.nr_vxc(mol, grids, 'lda,vwn', dm)
    '''
    if isinstance(relativity, (list, tuple, numpy.ndarray)):
        import warnings
        xc_code = '%s, %s' % (xc_code, dms)
        dms = relativity
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn('API updates: the 5th argument c_id is depercated '
                          'and will be removed in future release.\n')

    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)

    ngrids = len(grids.weights)
    if ni.non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    else:
        non0tab = ni.non0tab

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'LDA')
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1, verbose)[:2]
                vrho = vxc[0]
                den = rho * weight
                nelec[idm] += den.sum()
                excsum[idm] += (den * exc).sum()
                # *.5 because vmat + vmat.T
                aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
                vmat[idm] += _dot_ao_ao(mol, ao, aow, nao, weight.size, mask)
                rho = exc = vxc = vrho = aow = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'GGA')
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1, verbose)[:2]
                vrho, vsigma = vxc[:2]
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += (den * exc).sum()
# ref eval_mat function
                wv = numpy.empty_like(rho)
                wv[0]  = weight * vrho * .5
                wv[1:] = rho[1:] * (weight * vsigma * 2)
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, nao, weight.size, mask)
                rho = exc = vxc = vrho = vsigma = wv = aow = None
    else:
        raise NotImplementedError('meta-GGA')

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat.reshape(nao,nao)
    return nelec, excsum, vmat

def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix on given meshgrids
    for a set of density matrices

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm : a list of 2D arrays
            A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of (alpha,beta) electrons generated by numerical integration.
        excsum is the XC functional value.
        vmat is the XC potential matrix for (alpha,beta) spin.
    '''
    if isinstance(relativity, (list, tuple, numpy.ndarray)):
        import warnings
        xc_code = '%s, %s' % (xc_code, dms)
        dms = relativity
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn('API updates: the 5th argument c_id is depercated '
                          'and will be removed in future release.\n')

    xctype = ni._xc_type(xc_code)
    ngrids = len(grids.weights)
    if ni.non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    else:
        non0tab = ni.non0tab

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        make_rhoa, nset, nao = ni._gen_rho_evaluator(mol, dms*.5, hermi)
        make_rhob = make_rhoa
    else:
        nset = len(dms) // 2
        make_rhoa, nset, nao = ni._gen_rho_evaluator(mol, dms[:nset], hermi)
        make_rhob, nset, nao = ni._gen_rho_evaluator(mol, dms[nset:], hermi)

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((2,nset,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,idm] += den.sum()
                excsum[idm] += (den*exc).sum()
                den = rho_b * weight
                nelec[1,idm] += den.sum()
                excsum[idm] += (den*exc).sum()

                aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,0])
                vmat[0,idm] += _dot_ao_ao(mol, ao, aow, nao, weight.size, mask)
                aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1])
                vmat[1,idm] += _dot_ao_ao(mol, ao, aow, nao, weight.size, mask)
                rho_a = rho_b = exc = vxc = vrho = aow = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose)[:2]
                vrho, vsigma = vxc[:2]
                den = rho_a[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += (den*exc).sum()
                den = rho_b[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += (den*exc).sum()

                wv = numpy.empty_like(rho_a)
                wv[0]  = weight * vrho[:,0] * .5
                wv[1:] = rho_a[1:] * (weight * vsigma[:,0] * 2)  # sigma_uu
                wv[1:]+= rho_b[1:] * (weight * vsigma[:,1])      # sigma_ud
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, nao, weight.size, mask)
                wv[0]  = weight * vrho[:,1] * .5
                wv[1:] = rho_b[1:] * (weight * vsigma[:,2] * 2)  # sigma_dd
                wv[1:]+= rho_a[1:] * (weight * vsigma[:,1])      # sigma_ud
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, nao, weight.size, mask)
                rho_a = rho_b = exc = vxc = vrho = vsigma = wv = aow = None
    else:
        raise NotImplementedError('meta-GGA')

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].T
        vmat[1,i] = vmat[1,i] + vmat[1,i].T
    if nset == 1:
        nelec = nelec.reshape(2)
        excsum = excsum[0]
        vmat = vmat.reshape(2,nao,nao)
    return nelec, excsum, vmat

nr_rks_vxc = nr_rks
nr_uks_vxc = nr_uks

def nr_rks_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=1,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Contract RKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`_NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array a list of 2D arrays
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        rho0 : float array
            Zero-order density (and density derivative for GGA).  Giving kwargs rho0,
            vxc and fxc to improve better performance.
        vxc : float array
            First order XC derivatives
        fxc : float array
            Second order XC derivatives

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    '''
    xctype = ni._xc_type(xc_code)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        make_rho0 = ni._gen_rho_evaluator(mol, dm0, 1)[0]

    ngrids = len(grids.weights)
    if ni.non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    else:
        non0tab = ni.non0tab

    vmat = numpy.zeros((nset,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            ngrid = weight.size
            if fxc is None:
                rho = make_rho0(0, ao, mask, 'LDA')
                fxc0 = ni.eval_xc(xc_code, rho, 0, relativity, 2, verbose)[2]
                frr = fxc0[0]
            else:
                frr = fxc[0][ip:ip+ngrid]
                ip += ngrid

            for i in range(nset):
                rho1 = make_rho(i, ao, mask, 'LDA')
                aow = numpy.einsum('pi,p->pi', ao, weight*frr*rho1)
                vmat[i] += _dot_ao_ao(mol, aow, ao, nao, weight.size, mask)
                rho1 = aow = None

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            ngrid = weight.size
            if rho0 is None:
                rho = make_rho0(0, ao, mask, 'GGA')
            else:
                rho = numpy.asarray(rho0[:,ip:ip+ngrid], order='C')
            if vxc is None or fxc is None:
                vxc0, fxc0 = ni.eval_xc(xc_code, rho, 0, relativity, 2, verbose)[1:3]
                vgamma = vxc0[1]
                frr, frg, fgg = fxc0[:3]
            else:
                vgamma = vxc[1][ip:ip+ngrid]
                frr = fxc[0][ip:ip+ngrid]
                frg = fxc[1][ip:ip+ngrid]
                fgg = fxc[2][ip:ip+ngrid]
                ip += ngrid

            wv = numpy.empty((4,ngrid))
            for i in range(nset):
                rho1 = make_rho(i, ao, mask, 'GGA')
                sigma1 = numpy.einsum('xi,xi->i', rho[1:], rho1[1:])
                wv[0]  = frr * rho1[0]
                wv[0] += frg * sigma1 * 2
                wv[1:] = (fgg * sigma1 * 4 + frg * rho1[0] * 2) * rho[1:]
                wv[1:]+= vgamma * rho1[1:] * 2
                wv[1:]*= 2  # for (\nabla\mu) \nu + \mu (\nabla\nu)
                wv *= weight
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[i] += _dot_ao_ao(mol, aow, ao[0], nao, ngrid, mask)
                rho1 = sigma1 = aow = None
    else:
        raise NotImplementedError('meta-GGA')

    for i in range(nset):
        vmat[i] = (vmat[i] + vmat[i].T) * .5
    if nset == 1:
        vmat = vmat.reshape(nao,nao)
    return vmat

def nr_uks_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=1,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Contract UKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`_NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array a list of 2D arrays
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        rho0 : float array
            Zero-order density (and density derivative for GGA).  Giving kwargs rho0,
            vxc and fxc to improve better performance.
        vxc : float array
            First order XC derivatives
        fxc : float array
            Second order XC derivatives

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    '''
    xctype = ni._xc_type(xc_code)

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        make_rhoa, nset, nao = ni._gen_rho_evaluator(mol, dms*.5, hermi)
        make_rhob = make_rhoa
    else:
        nset = len(dms) // 2
        make_rhoa, nset, nao = ni._gen_rho_evaluator(mol, dms[:nset], hermi)
        make_rhob, nset, nao = ni._gen_rho_evaluator(mol, dms[nset:], hermi)

    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2:
            make_rho0a = make_rho0b = ni._gen_rho_evaluator(mol, dm0*.5, 1)[0]
        else:
            make_rho0a = ni._gen_rho_evaluator(mol, dm0[0], 1)[0]
            make_rho0b = ni._gen_rho_evaluator(mol, dm0[1], 1)[0]

    ngrids = len(grids.weights)
    if ni.non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    else:
        non0tab = ni.non0tab

    vmat = numpy.zeros((2,nset,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            ngrid = weight.size
            if fxc is None:
                rho0a = make_rho0a(0, ao, mask, xctype)
                rho0b = make_rho0b(0, ao, mask, xctype)
                fxc0 = ni.eval_xc(xc_code, (rho0a,rho0b), 1, relativity, 2, verbose)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid*3].T
                ip += ngrid

            for i in range(nset):
                rho1a = make_rhoa(i, ao, mask, xctype)
                rho1b = make_rhob(i, ao, mask, xctype)
                wv = u_u * rho1a + u_d * rho1b
                wv *= weight
                aow = numpy.einsum('pi,p->pi', ao, wv)
                vmat[0,i] += _dot_ao_ao(mol, aow, ao, nao, weight.size, mask)
                wv = u_d * rho1a + d_d * rho1b
                wv *= weight
                aow = numpy.einsum('pi,p->pi', ao, wv)
                vmat[1,i] += _dot_ao_ao(mol, aow, ao, nao, weight.size, mask)
                rho1 = aow = None

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, non0tab):
            ngrid = weight.size
            if rho0 is None:
                rho0a = make_rho0a(0, ao, mask, xctype)
                rho0b = make_rho0b(0, ao, mask, xctype)
            else:
                rho0a = rho0[0][:,ip:ip+ngrid]
                rho0b = rho0[1][:,ip:ip+ngrid]
            if vxc is None or fxc is None:
                vxc0, fxc0 = ni.eval_xc(xc_code, (rho0a,rho0b), 1, relativity, 2, verbose)[1:3]
                uu, ud, dd = vxc0[1].T
                u_u, u_d, d_d = fxc0[0].T
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc0[1].T
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc0[2].T
            else:
                uu, ud, dd = vxc[1][ip:ip+ngrid].T
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1][ip:ip+ngrid].T
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2][ip:ip+ngrid].T
                ip += ngrid

            wv = numpy.empty((4,ngrid))
            for i in range(nset):
                rho1a = make_rhoa(i, ao, mask, xctype)
                rho1b = make_rhob(i, ao, mask, xctype)
                a0a1 = numpy.einsum('xi,xi->i', rho0a[1:], rho1a[1:])
                a0b1 = numpy.einsum('xi,xi->i', rho0a[1:], rho1b[1:])
                b0a1 = numpy.einsum('xi,xi->i', rho0b[1:], rho1a[1:])
                b0b1 = numpy.einsum('xi,xi->i', rho0b[1:], rho1b[1:])

                # alpha = alpha-alpha * alpha
                wv[0]  = u_u * rho1a[0]
                wv[0] += u_uu * a0a1 * 2
                wv[0] += u_ud * b0a1
                wv[1:] = uu * rho1a[1:] * 2
                wv[1:]+= u_uu * rho1a[0] * rho0a[1:] * 2
                wv[1:]+= u_ud * rho1a[0] * rho0b[1:]
                wv[1:]+= uu_uu * a0a1 * rho0a[1:] * 4
                wv[1:]+= uu_ud * a0a1 * rho0b[1:] * 2
                wv[1:]+= uu_ud * b0a1 * rho0a[1:] * 2
                wv[1:]+= ud_ud * b0a1 * rho0b[1:]

                # alpha = alpha-beta  * beta
                wv[0] += u_d * rho1b[0]
                wv[0] += u_ud * a0b1
                wv[0] += u_dd * b0b1 * 2
                wv[1:]+= ud * rho1b[1:]
                wv[1:]+= d_uu * rho1b[0] * rho0a[1:] * 2
                wv[1:]+= d_ud * rho1b[0] * rho0b[1:]
                wv[1:]+= uu_ud * a0b1 * rho0a[1:] * 2
                wv[1:]+= ud_ud * a0b1 * rho0b[1:]
                wv[1:]+= uu_dd * b0b1 * rho0a[1:] * 4
                wv[1:]+= ud_dd * b0b1 * rho0b[1:] * 2

                wv[1:] *= 2  # for (\nabla\mu) \nu + \mu (\nabla\nu)
                wv *= weight
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[0,i] += _dot_ao_ao(mol, aow, ao[0], nao, ngrid, mask)
                aow = None

                # beta = beta-alpha * alpha
                wv[0]  = u_d * rho1a[0]
                wv[0] += d_ud * b0a1
                wv[0] += d_uu * a0a1 * 2
                wv[1:] = ud * rho1a[1:]
                wv[1:]+= u_dd * rho1a[0] * rho0b[1:] * 2
                wv[1:]+= u_ud * rho1a[0] * rho0a[1:]
                wv[1:]+= ud_dd * b0a1 * rho0b[1:] * 2
                wv[1:]+= ud_ud * b0a1 * rho0a[1:]
                wv[1:]+= uu_dd * a0a1 * rho0b[1:] * 4
                wv[1:]+= uu_ud * a0a1 * rho0a[1:] * 2

                # beta = beta-beta  * beta
                wv[0] += d_d * rho1b[0]
                wv[0] += d_dd * b0b1 * 2
                wv[0] += d_ud * a0b1
                wv[1:]+= dd * rho1b[1:] * 2
                wv[1:]+= d_dd * rho1b[0] * rho0b[1:] * 2
                wv[1:]+= d_ud * rho1b[0] * rho0a[1:]
                wv[1:]+= dd_dd * b0b1 * rho0b[1:] * 4
                wv[1:]+= ud_dd * b0b1 * rho0a[1:] * 2
                wv[1:]+= ud_dd * a0b1 * rho0b[1:] * 2
                wv[1:]+= ud_ud * a0b1 * rho0a[1:]

                wv[1:] *= 2  # for (\nabla\mu) \nu + \mu (\nabla\nu)
                wv *= weight
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[1,i] += _dot_ao_ao(mol, aow, ao[0], nao, ngrid, mask)
                aow = None
    else:
        raise NotImplementedError('meta-GGA')

    for i in range(nset):
        vmat[0,i] = (vmat[0,i] + vmat[0,i].T) * .5
        vmat[1,i] = (vmat[1,i] + vmat[1,i].T) * .5
    if nset == 1:
        vmat = vmat.reshape(2,nao,nao)
    return vmat

def nr_fxc(mol, grids, xc_code, dm0, dms, spin0, relativity=0, hermi=1,
           rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    r'''Contract XC kernel matrix with given density matrices

    ... math::

            a_{pq} = f_{pq,rs} * x_{rs}

    '''
    ni = _NumInt()
    ni.non0tab = ni.make_mask(mol, grids.coords)
    if spin == 0:
        return nr_rks_fxc(ni, mol, grids, xc_code, dm, dms, relativity,
                          hermi, rho0, vxc, fxc, max_memory, verbose)
    else:
        return nr_uks_fxc(ni, mol, grids, xc_code, dm, dms, relativity,
                          hermi, rho0, vxc, fxc, max_memory, verbose)


def cache_xc_kernel_(ni, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                     max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    if ni.non0tab is None:
        ni.non0tab = ni.make_mask(mol, grids.coords)

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    else:
        raise NotImplementedError('meta-GGA')

    if spin == 0:
        nao = mo_coeff.shape[0]
        rho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, ni.non0tab):
            rho.append(ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, xctype))
        rho = numpy.hstack(rho)
    else:
        nao = mo_coeff[0].shape[0]
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, ni.non0tab):
            rhoa.append(ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype))
            rhob.append(ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc(xc_code, rho, spin, 0, 2, 0)[1:3]
    return rho, vxc, fxc


def large_rho_indices(ni, mol, dm, grids, cutoff=1e-10, max_memory=2000):
    '''Indices of density which are larger than given cutoff
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1)
    idx = []
    cutoff = cutoff / grids.weights.size
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory, ni.non0tab):
        rho = make_rho(0, ao, mask, 'LDA')
        idx.append(abs(rho*weight) > cutoff)
    return numpy.hstack(idx)


class _NumInt(object):
    '''libxc is the default xc functional evaluator.  Change the default one
    by setting
    _NumInt.libxc = dft.xcfun
    '''
    libxc = libxc

    def __init__(self):
        self.non0tab = None

    def nr_vxc_(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix on given meshgrids
        for a set of density matrices.  See :func:`nr_rks_` and :func:`nr_uks_`
        for more details.
        '''
        if spin == 0:
            return self.nr_rks_(mol, grids, xc_code, dms, relativity, hermi,
                                max_memory, verbose)
        else:
            return self.nr_uks_(mol, grids, xc_code, dms, relativity, hermi,
                                max_memory, verbose)
    nr_vxc = nr_vxc_

    def nr_rks_(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        if self.non0tab is None:
            self.non0tab = self.make_mask(mol, grids.coords)
        return nr_rks(self, mol, grids, xc_code, dms, relativity, hermi,
                      max_memory, verbose)
    nr_rks_.__doc__ = nr_rks_vxc.__doc__
    nr_rks = nr_rks_

    def nr_uks_(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        if self.non0tab is None:
            self.non0tab = self.make_mask(mol, grids.coords)
        return nr_uks(self, mol, grids, xc_code, dms, relativity, hermi,
                      max_memory, verbose)
    nr_uks_.__doc__ = nr_uks_vxc.__doc__
    nr_uks = nr_uks_

    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_fxc = nr_fxc
    cache_xc_kernel_ = cache_xc_kernel_
    cache_xc_kernel  = cache_xc_kernel_

    large_rho_indices

    @pyscf.lib.with_doc(eval_ao.__doc__)
    def eval_ao(self, mol, coords, deriv=0, relativity=0, bastart=0,
                bascount=None, non0tab=None, out=None, verbose=None):
        return eval_ao(mol, coords, deriv, relativity, bastart, bascount,
                       non0tab, out, verbose)

    @pyscf.lib.with_doc(make_mask.__doc__)
    def make_mask(self, mol, coords, relativity=0, bastart=0, bascount=None,
                  verbose=None):
        return make_mask(mol, coords, relativity, bastart, bascount, verbose)

    @pyscf.lib.with_doc(eval_rho2.__doc__)
    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        return eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab, xctype, verbose)

    @pyscf.lib.with_doc(eval_rho.__doc__)
    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', verbose=None):
        return eval_rho(mol, ao, dm, non0tab, xctype, verbose)

    def block_loop(self, mol, grids, nao, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        '''Define this macro to loop over grids by blocks.
        '''
        ngrids = grids.weights.size
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index ni.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = min(int(max_memory*1e6/(comp*2*nao*8*BLKSIZE))*BLKSIZE, ngrids)
            blksize = max(blksize, BLKSIZE)
        if non0tab is None:
            non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                 dtype=numpy.int8)
        if buf is None:
            buf = numpy.empty((comp,blksize,nao))
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = self.eval_ao(mol, coords, deriv, non0tab=non0, out=buf)
            yield ao, non0, weight, coords

    def _gen_rho_evaluator(self, mol, dms, hermi=1):
        if hermi == 1:
            natocc = []
            natorb = []
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                e, c = scipy.linalg.eigh(dms)
                natocc.append(e)
                natorb.append(c)
                nao = dms.shape[0]
            else:
                for dm in dms:
                    e, c = scipy.linalg.eigh(dm)
                    natocc.append(e)
                    natorb.append(c)
                nao = dms[0].shape[0]
            ndms = len(natocc)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(mol, ao, natorb[idm], natocc[idm], non0tab, xctype)
        else:
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                nao = dms.shape[0]
                dms = [dms]
            else:
                nao = dms[0].shape[0]
            ndms = len(dms)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(mol, ao, dms[idm], non0tab, xctype)
        return make_rho, ndms, nao

####################

    def hybrid_coeff(self, xc_code, spin=1):
        return self.libxc.hybrid_coeff(xc_code, spin)

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        return self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv, verbose)
    eval_xc.__doc__ = libxc.eval_xc.__doc__

    def _xc_type(self, xc_code):
        libxc = self.libxc
        if libxc.is_lda(xc_code):
            xctype = 'LDA'
        elif libxc.is_meta_gga(xc_code):
            xctype = 'MGGA'
            raise NotImplementedError('meta-GGA')
        else:
            xctype = 'GGA'
        return xctype


if __name__ == '__main__':
    import time
    from pyscf import gto
    from pyscf import dft

    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = '6311g*',)
    mf = dft.RKS(mol)
    mf.grids.atom_grid = {"H": (30, 194), "O": (30, 194),},
    mf.grids.prune = None
    mf.grids.build_()
    dm = mf.get_init_guess(key='minao')

    numpy.random.seed(1)
    dm1 = numpy.random.random((dm.shape))
    print(time.clock())
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm1, spin=0)
    print(res[1] - -37.08079395351452)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm1,dm1), spin=1)
    print(res[1] - -92.42828116583394)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm, spin=0)
    print(res[1] - -8.631329952248500)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm,dm), spin=1)
    print(res[1] - -21.52029393479996)
    print(time.clock())
