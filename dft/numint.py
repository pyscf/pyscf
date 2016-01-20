#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import time
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.dft.vxc
from pyscf.lib import logger

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
        Or 3D array of shape (*,N,nao) for AO values and AO derivatives if deriv > 0.
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
            rho[i] = numpy.einsum('pi,pi->p', ao[0], c1) * 2 # *2 for +c.c.
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
        verbose : int or object of :class:`Logger`
            No effects.

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

def eval_x(x_id, rho, spin=0, relativity=0, deriv=1, verbose=None):
    r'''Interface to call libxc library to evaluate exchange functional,
    potential and functional derivatives.

    Args:
        x_id : int
            Exchange functional ID used by libxc library.  See pyscf/dft/vxc.py for more details.
        rho : ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))

    Kwargs:
        spin : int
            spin polarized if spin > 0
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        ex, (vrho(*,2), vsigma(*,3), vlapl(*,2), vtau(*,2)), fxc, kxc

        where

        * fxc(N*45) for unrestricted case:
        | v2rho2[*,3]     = (u_u, u_d, d_d)
        | v2rhosigma[*,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
        | v2sigma2[*,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
        | v2lapl2[*,3]
        | vtau2[*,3]
        | v2rholapl[*,4]
        | v2rhotau[*,4]
        | v2lapltau[*,4]
        | v2sigmalapl[*,6]
        | v2sigmatau[*,6]

        * fxc(N*10) for restricted case:
        (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)

        * kxc(N*35) for unrestricted case:
        | v3rho3[*,4]       = (u_u_u, u_u_d, u_d_d, d_d_d)
        | v3rho2sigma[*,9]  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
        | v3rhosigma2[*,12] = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
        | v3sigma[*,10]     = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)

        * kxc(N*4) for restricted case:
        (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma)

        see also libxc_itrf.c
    '''
    if spin == 0:
        rho_u = numpy.asarray(rho, order='C')
        prho_u = rho_u.ctypes.data_as(ctypes.c_void_p)
        prho_d = pyscf.lib.c_null_ptr()
    else:
        rho_u = numpy.asarray(rho[0], order='C')
        rho_d = numpy.asarray(rho[1], order='C')
        prho_u = rho_u.ctypes.data_as(ctypes.c_void_p)
        prho_d = rho_d.ctypes.data_as(ctypes.c_void_p)

    if rho_u.ndim == 2:
        ngrids = rho_u.shape[1]
    else:
        ngrids = len(rho_u)

    if spin == 0:
        nspin = 1
        exc = numpy.empty(ngrids)
        if deriv > 0:
            vxc = numpy.zeros(4*ngrids)
            vrho   = vxc[        :ngrids  ]
            vsigma = vxc[ngrids  :ngrids*2]
            vlapl  = vxc[ngrids*2:ngrids*3]
            vtau   = vxc[ngrids*3:ngrids*4]
            pvxc = vxc.ctypes.data_as(ctypes.c_void_p)
        else:
            vxc = vrho = vsigma = vlapl = vtau = None
            pvxc = pyscf.lib.c_null_ptr()
        if deriv > 1:
            fxc = numpy.zeros(10*ngrids)
            pfxc = fxc.ctypes.data_as(ctypes.c_void_p)
        else:
            fxc = None
            pfxc = pyscf.lib.c_null_ptr()
        if deriv > 2:
            kxc = numpy.zeros(4*ngrids)
            pkxc = kxc.ctypes.data_as(ctypes.c_void_p)
        else:
            kxc = None
            pkxc = pyscf.lib.c_null_ptr()
    else:
        nspin = 2
        exc = numpy.zeros(ngrids)
        if deriv > 0:
            vxc = numpy.zeros(9*ngrids)
            vrho   = vxc[        :ngrids*2].reshape(ngrids,2)
            vsigma = vxc[ngrids*2:ngrids*5].reshape(ngrids,3)
            vlapl  = vxc[ngrids*5:ngrids*7].reshape(ngrids,2)
            vtau   = vxc[ngrids*7:ngrids*9].reshape(ngrids,2)
            pvxc = vxc.ctypes.data_as(ctypes.c_void_p)
        else:
            vxc = vrho = vsigma = vlapl = vtau = None
            pvxc = pyscf.lib.c_null_ptr()
        if deriv > 1:
            fxc = numpy.zeros(45*ngrids)
            pfxc = fxc.ctypes.data_as(ctypes.c_void_p)
        else:
            fxc = None
            pfxc = pyscf.lib.c_null_ptr()
        if deriv > 2:
            kxc = numpy.zeros(35*ngrids)
            pkxc = kxc.ctypes.data_as(ctypes.c_void_p)
        else:
            kxc = None
            pkxc = pyscf.lib.c_null_ptr()
    libdft.VXCnr_eval_x(ctypes.c_int(x_id), ctypes.c_int(nspin),
                        ctypes.c_int(relativity), ctypes.c_int(ngrids),
                        prho_u, prho_d,
                        exc.ctypes.data_as(ctypes.c_void_p), pvxc, pfxc, pkxc)
    return exc, (vrho, vsigma, vlapl, vtau), fxc, kxc

def eval_c(x_id, rho, spin=0, relativity=0, deriv=1, verbose=None):
    r'''Interface to call libxc library to evaluate correlation functional,
    potential and functional derivatives.

    Args:
        x_id : int
            Correlation functional ID used by libxc library.  See pyscf/dft/vxc.py for more details.
        rho : ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))

    Kwargs:
        spin : int
            spin polarized if spin > 0
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        ex, (vrho(*,2), vsigma(*,3), vlapl(*,2), vtau(*,2)), fxc, kxc

        where

        * fxc(N*45) for unrestricted case:
        | v2rho2[*,3]     = (u_u, u_d, d_d)
        | v2rhosigma[*,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
        | v2sigma2[*,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
        | v2lapl2[*,3]
        | vtau2[*,3]
        | v2rholapl[*,4]
        | v2rhotau[*,4]
        | v2lapltau[*,4]
        | v2sigmalapl[*,6]
        | v2sigmatau[*,6]

        * fxc(N*10) for restricted case:
        (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)

        * kxc(N*35) for unrestricted case:
        | v3rho3[*,4]       = (u_u_u, u_u_d, u_d_d, d_d_d)
        | v3rho2sigma[*,9]  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
        | v3rhosigma2[*,12] = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
        | v3sigma[*,10]     = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)

        * kxc(N*4) for restricted case:
        (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma)

        see also libxc_itrf.c
    '''
    if spin == 0:
        rho_u = numpy.asarray(rho, order='C')
        prho_u = rho_u.ctypes.data_as(ctypes.c_void_p)
        prho_d = pyscf.lib.c_null_ptr()
    else:
        rho_u = numpy.asarray(rho[0], order='C')
        rho_d = numpy.asarray(rho[1], order='C')
        prho_u = rho_u.ctypes.data_as(ctypes.c_void_p)
        prho_d = rho_d.ctypes.data_as(ctypes.c_void_p)

    if rho_u.ndim == 2:
        ngrids = rho_u.shape[1]
    else:
        ngrids = len(rho_u)

    if spin == 0:
        nspin = 1
        exc = numpy.zeros(ngrids)
        if deriv > 0:
            vxc = numpy.zeros(4*ngrids)
            vrho   = vxc[        :ngrids  ]
            vsigma = vxc[ngrids  :ngrids*2]
            vlapl  = vxc[ngrids*2:ngrids*3]
            vtau   = vxc[ngrids*3:ngrids*4]
            pvxc = vxc.ctypes.data_as(ctypes.c_void_p)
        else:
            vxc = vrho = vsigma = vlapl = vtau = None
            pvxc = pyscf.lib.c_null_ptr()
        if deriv > 1:
            fxc = numpy.zeros(10*ngrids)
            pfxc = fxc.ctypes.data_as(ctypes.c_void_p)
        else:
            fxc = None
            pfxc = pyscf.lib.c_null_ptr()
        if deriv > 2:
            kxc = numpy.zeros(4*ngrids)
            pkxc = kxc.ctypes.data_as(ctypes.c_void_p)
        else:
            kxc = None
            pkxc = pyscf.lib.c_null_ptr()
    else:
        nspin = 2
        exc = numpy.zeros(ngrids)
        if deriv > 0:
            vxc = numpy.zeros(9*ngrids)
            vrho   = vxc[        :ngrids*2].reshape(ngrids,2)
            vsigma = vxc[ngrids*2:ngrids*5].reshape(ngrids,3)
            vlapl  = vxc[ngrids*5:ngrids*7].reshape(ngrids,2)
            vtau   = vxc[ngrids*7:ngrids*9].reshape(ngrids,2)
            pvxc = vxc.ctypes.data_as(ctypes.c_void_p)
        else:
            vxc = vrho = vsigma = vlapl = vtau = None
            pvxc = pyscf.lib.c_null_ptr()
        if deriv > 1:
            fxc = numpy.zeros(45*ngrids)
            pfxc = fxc.ctypes.data_as(ctypes.c_void_p)
        else:
            fxc = None
            pfxc = pyscf.lib.c_null_ptr()
        if deriv > 2:
            kxc = numpy.zeros(35*ngrids)
            pkxc = kxc.ctypes.data_as(ctypes.c_void_p)
        else:
            kxc = None
            pkxc = pyscf.lib.c_null_ptr()
    libdft.VXCnr_eval_c(ctypes.c_int(x_id), ctypes.c_int(nspin),
                        ctypes.c_int(relativity), ctypes.c_int(ngrids),
                        prho_u, prho_d,
                        exc.ctypes.data_as(ctypes.c_void_p), pvxc, pfxc, pkxc)
    return exc, (vrho, vsigma, vlapl, vtau), fxc, kxc

def eval_xc(x_id, c_id, rho, spin=0, relativity=0, deriv=1, verbose=None):
    r'''Interface to call libxc library to evaluate XC functional, potential
    and functional derivatives.

    Args:
        x_id, c_id : int
            Exchange/Correlation functional ID used by libxc library.
            See pyscf/dft/vxc.py for more details.
        rho : ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))

    Kwargs:
        spin : int
            spin polarized if spin > 0
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        ex, (vrho(*,2), vsigma(*,3), vlapl(*,2), vtau(*,2)), fxc, kxc

        where

        * fxc(N*45) for unrestricted case:
        | v2rho2[*,3]     = (u_u, u_d, d_d)
        | v2rhosigma[*,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
        | v2sigma2[*,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
        | v2lapl2[*,3]
        | vtau2[*,3]
        | v2rholapl[*,4]
        | v2rhotau[*,4]
        | v2lapltau[*,4]
        | v2sigmalapl[*,6]
        | v2sigmatau[*,6]

        * fxc(N*10) for restricted case:
        (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)

        * kxc(N*35) for unrestricted case:
        | v3rho3[*,4]       = (u_u_u, u_u_d, u_d_d, d_d_d)
        | v3rho2sigma[*,9]  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
        | v3rhosigma2[*,12] = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
        | v3sigma[*,10]     = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)

        * kxc(N*4) for restricted case:
        (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma)

        see also libxc_itrf.c
    '''
    exc, (vrho, vsigma, vlapl, vtau), fxc, kxc = \
            eval_x(x_id, rho, spin, relativity, deriv, verbose)
    if c_id > 0 and not pyscf.dft.vxc.is_hybrid_xc(x_id):
        ec, (vrhoc, vsigmac, vlaplc, vtauc), fc, kc = \
                eval_c(c_id, rho, spin, relativity, deriv, verbose)
        exc += ec
        if vrho is not None:
            vrho   += vrhoc
            vsigma += vsigmac
            vlapl  += vlaplc
            vtau   += vtauc
        if fxc is not None:
            fxc += fc
        if kxc is not None:
            kxc += kc
    return exc, (vrho, vsigma, vlapl, vtau), fxc, kxc


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

def nr_vxc(mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    if spin == 0:
        return nr_rks_vxc(_NumInt(), mol, grids, x_id, c_id, dm, spin, relativity,
                          hermi, max_memory, verbose)
    else:
        return nr_uks_vxc(_NumInt(), mol, grids, x_id, c_id, dm, spin, relativity,
                          hermi, max_memory, verbose)

def nr_rks_vxc(ni, mol, grids, x_id, c_id, dms, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix for given meshgrids and density matrix

    Args:
        ni : an instance of :class:`_NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        x_id, c_id : int
            Exchange/Correlation functional ID used by libxc library.
            See pyscf/dft/vxc.py for more details.
        dms : 2D array a list of 2D arrays
            Density matrix or multiple density matrices

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
    assert(hermi == 1)

#TEST ME

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        nao = dms.shape[0]
        dms = [dms]
    else:
        nao = dms[0].shape[0]

    xctype = _xc_type(x_id, c_id)
    ngrids = len(grids.weights)
    blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
    if ni.non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    else:
        non0tab = ni.non0tab

    nset = len(dms)
    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros_like(dms)
    if xctype == 'LDA':
        buf = numpy.empty((blksize,nao))
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=0, non0tab=non0, out=buf)
            for idm, dm in enumerate(dms):
                rho = ni.eval_rho(mol, ao, dm, non0, xctype)
                exc, vxc = ni.eval_xc(x_id, c_id, rho,
                                      spin, relativity, 1, verbose)[:2]
                vrho = vxc[0]
                den = rho * weight
                nelec[idm] += den.sum()
                excsum[idm] += (den*exc).sum()
                aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
                vmat[idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0, non0)
                rho = exc = vxc = vrho = aow = None
    elif xctype == 'GGA':
        buf = numpy.empty((4,blksize,nao))
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=1, non0tab=non0, out=buf)
            for idm, dm in enumerate(dms):
                rho = ni.eval_rho(mol, ao, dm, non0, xctype)
                exc, vxc = ni.eval_xc(x_id, c_id, rho,
                                      spin, relativity, 1, verbose)[:2]
                vrho, vsigma = vxc[:2]
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += (den*exc).sum()
# ref eval_mat function
                wv = numpy.empty_like(rho)
                wv[0]  = weight * vrho * .5
                wv[1:] = rho[1:] * (weight * vsigma * 2)
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0)
                rho = exc = vxc = vrho = vsigma = wv = aow = None
    else:
        buf = numpy.empty((6,blksize,nao))
        raise NotImplementedError('meta-GGA')

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat.reshape(nao,nao)
    return nelec, excsum, vmat

def nr_uks_vxc(ni, mol, grids, x_id, c_id, dms, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix for given meshgrids
    and a set of density matrices

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        x_id, c_id : int
            Exchange/Correlation functional ID used by libxc library.
            See pyscf/dft/vxc.py for more details.
        dms : a list of 2D arrays
            A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

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
    assert(hermi == 1)

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        nao = dms.shape[0]
        nset = 1
        dms = [dms,dms]
    else:
        nao = dms[0].shape[0]
        nset = len(dms) // 2

    xctype = _xc_type(x_id, c_id)
    ngrids = len(grids.weights)
# NOTE to index ni.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
    blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)
    if ni.non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)
    else:
        non0tab = ni.non0tab

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((2,nset,nao,nao))
    if xctype == 'LDA':
        buf = numpy.empty((blksize,nao))
        for ip0, ip1 in prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=0, non0tab=non0, out=buf)
            for idm in range(nset):
                dm_a = dms[idm]
                dm_b = dms[nset+idm]
                rho_a = ni.eval_rho(mol, ao, dm_a, non0, xctype)
                rho_b = ni.eval_rho(mol, ao, dm_b, non0, xctype)
                exc, vxc = ni.eval_xc(x_id, c_id, (rho_a, rho_b),
                                      1, relativity, 1, verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,idm] += den.sum()
                excsum[idm] += (den*exc).sum()
                den = rho_b * weight
                nelec[1,idm] += den.sum()
                excsum[idm] += (den*exc).sum()

                aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,0])
                vmat[0,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0, non0)
                aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1])
                vmat[1,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0, non0)
                rho_a = rho_b = exc = vxc = vrho = aow = None
    elif xctype == 'GGA':
        buf = numpy.empty((4,blksize,nao))
        for ip0, ip1 in prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=1, non0tab=non0, out=buf)
            for idm in range(nset):
                dm_a = dms[idm]
                dm_b = dms[nset+idm]
                rho_a = ni.eval_rho(mol, ao, dm_a, non0, xctype)
                rho_b = ni.eval_rho(mol, ao, dm_b, non0, xctype)
                exc, vxc = ni.eval_xc(x_id, c_id, (rho_a, rho_b),
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
                vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0)
                wv[0]  = weight * vrho[:,1] * .5
                wv[1:] = rho_b[1:] * (weight * vsigma[:,2] * 2)  # sigma_dd
                wv[1:]+= rho_a[1:] * (weight * vsigma[:,1])      # sigma_ud
                aow = numpy.einsum('npi,np->pi', ao, wv)
                vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0)
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


class _NumInt(object):
    def __init__(self):
        self.non0tab = None

    def nr_vxc(self, mol, grids, x_id, c_id, dm, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix for given meshgrids
        and a set of density matrices.  See :func:`nr_rks` and :func:`nr_uks`
        for more details.
        '''
        if spin == 0:
            return self.nr_rks(mol, grids, x_id, c_id, dm, relativity, hermi,
                               max_memory, verbose)
        else:
            return self.nr_uks(mol, grids, x_id, c_id, dm, relativity, hermi,
                               max_memory, verbose)

    def nr_rks(self, mol, grids, x_id, c_id, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        '''Calculate RKS XC functional and potential matrix for given meshgrids
        and a set of density matrices

        Args:
            mol : an instance of :class:`Mole`

            grids : an instance of :class:`Grids`
                grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
            x_id, c_id : int
                Exchange/Correlation functional ID used by libxc library.
                See pyscf/dft/vxc.py for more details.
            dms : 2D array a list of 2D arrays
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
            self.non0tab = self.make_mask(mol, grids.coords)

        if hermi != 1:
            return nr_rks_vxc(self, mol, grids, x_id, c_id, dms,
                              0, relativity, hermi, max_memory, verbose)

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

        xctype = _xc_type(x_id, c_id)
        ngrids = len(grids.weights)
# NOTE to index self.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)

        nset = len(natocc)
        nelec = numpy.zeros(nset)
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((nset,nao,nao))
        if xctype == 'LDA':
            buf = numpy.empty((blksize,nao))
            for ip0, ip1 in prange(0, ngrids, blksize):
                coords = grids.coords[ip0:ip1]
                weight = grids.weights[ip0:ip1]
                non0tab = self.non0tab[ip0//BLKSIZE:]
                ao = self.eval_ao(mol, coords, deriv=0, non0tab=non0tab,
                                  out=buf)
                for idm in range(nset):
                    rho = self.eval_rho2(mol, ao, natorb[idm], natocc[idm],
                                         non0tab, xctype)
                    exc, vxc = self.eval_xc(x_id, c_id, rho,
                                            0, relativity, 1, verbose)[:2]
                    vrho = vxc[0]
                    den = rho * weight
                    nelec[idm] += den.sum()
                    excsum[idm] += (den * exc).sum()
                    # *.5 because vmat + vmat.T
                    aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
                    vmat[idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0, non0tab)
                    rho = exc = vxc = vrho = aow = None
        elif xctype == 'GGA':
            buf = numpy.empty((4,blksize,nao))
            for ip0, ip1 in prange(0, ngrids, blksize):
                coords = grids.coords[ip0:ip1]
                weight = grids.weights[ip0:ip1]
                non0tab = self.non0tab[ip0//BLKSIZE:]
                ao = self.eval_ao(mol, coords, deriv=1, non0tab=non0tab,
                                  out=buf)
                for idm in range(nset):
                    rho = self.eval_rho2(mol, ao, natorb[idm], natocc[idm],
                                         non0tab, xctype)
                    exc, vxc = self.eval_xc(x_id, c_id, rho,
                                            0, relativity, 1, verbose)[:2]
                    vrho, vsigma = vxc[:2]
                    den = rho[0] * weight
                    nelec[idm] += den.sum()
                    excsum[idm] += (den * exc).sum()
# ref eval_mat function
                    wv = numpy.empty_like(rho)
                    wv[0]  = weight * vrho * .5
                    wv[1:] = rho[1:] * (weight * vsigma * 2)
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0tab)
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

    def nr_uks(self, mol, grids, x_id, c_id, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        '''Calculate UKS XC functional and potential matrix for given meshgrids
        and a set of density matrices

        Args:
            mol : an instance of :class:`Mole`

            grids : an instance of :class:`Grids`
                grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
            x_id, c_id : int
                Exchange/Correlation functional ID used by libxc library.
                See pyscf/dft/vxc.py for more details.
            dm : a list of 2D arrays
                A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

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
            self.non0tab = self.make_mask(mol, grids.coords)

        if hermi != 1:
            return nr_uks_vxc(self, mol, grids, x_id, c_id, dms,
                              mol.spin, relativity, hermi, max_memory, verbose)

        natocc = []
        natorb = []
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            e, c = scipy.linalg.eigh(dms)
            natocc.append((e*.5,e*.5))
            natorb.append((c,c))
            nset = 1
            nao = dms.shape[0]
        else:
            nset = len(dms) // 2
            for idm in range(nset):
                e_a, c_a = scipy.linalg.eigh(dms[idm])
                e_b, c_b = scipy.linalg.eigh(dms[nset+idm])
                natocc.append((e_a,e_b))
                natorb.append((c_a,c_b))
            nao = dms[0].shape[0]

        xctype = _xc_type(x_id, c_id)
        ngrids = len(grids.weights)
# NOTE to index self.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        blksize = min(int(max_memory/6*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)

        nelec = numpy.zeros((2,nset))
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((2,nset,nao,nao))
        if xctype == 'LDA':
            buf = numpy.empty((blksize,nao))
            for ip0, ip1 in prange(0, ngrids, blksize):
                coords = grids.coords[ip0:ip1]
                weight = grids.weights[ip0:ip1]
                non0tab = self.non0tab[ip0//BLKSIZE:]
                ao = self.eval_ao(mol, coords, deriv=0, non0tab=non0tab, out=buf)
                for idm in range(nset):
                    c_a, c_b = natorb[idm]
                    e_a, e_b = natocc[idm]
                    rho_a = self.eval_rho2(mol, ao, c_a, e_a, non0tab, xctype)
                    rho_b = self.eval_rho2(mol, ao, c_b, e_b, non0tab, xctype)
                    exc, vxc = self.eval_xc(x_id, c_id, (rho_a, rho_b),
                                            1, relativity, 1, verbose)[:2]
                    vrho = vxc[0]
                    den = rho_a * weight
                    nelec[0,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()
                    den = rho_b * weight
                    nelec[1,idm] += den.sum()
                    excsum[idm] += (den*exc).sum()

                    aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,0])
                    vmat[0,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0, non0tab)
                    aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1])
                    vmat[1,idm] += _dot_ao_ao(mol, ao, aow, nao, ip1-ip0, non0tab)
                    rho_a = rho_b = exc = vxc = vrho = aow = None
        elif xctype == 'GGA':
            buf = numpy.empty((4,blksize,nao))
            for ip0, ip1 in prange(0, ngrids, blksize):
                coords = grids.coords[ip0:ip1]
                weight = grids.weights[ip0:ip1]
                non0tab = self.non0tab[ip0//BLKSIZE:]
                ao = self.eval_ao(mol, coords, deriv=1, non0tab=non0tab, out=buf)
                for idm in range(nset):
                    c_a, c_b = natorb[idm]
                    e_a, e_b = natocc[idm]
                    rho_a = self.eval_rho2(mol, ao, c_a, e_a, non0tab, xctype)
                    rho_b = self.eval_rho2(mol, ao, c_b, e_b, non0tab, xctype)
                    exc, vxc = self.eval_xc(x_id, c_id, (rho_a, rho_b),
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
                    vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0tab)
                    wv[0]  = weight * vrho[:,1] * .5
                    wv[1:] = rho_b[1:] * (weight * vsigma[:,2] * 2)  # sigma_dd
                    wv[1:]+= rho_a[1:] * (weight * vsigma[:,1])      # sigma_ud
                    aow = numpy.einsum('npi,np->pi', ao, wv)
                    vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0tab)
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

    def eval_ao(self, mol, coords, deriv=0, relativity=0, bastart=0,
                bascount=None, non0tab=None, out=None, verbose=None):
        return eval_ao(mol, coords, deriv, relativity, bastart, bascount,
                       non0tab, out, verbose)

    def make_mask(self, mol, coords, relativity=0, bastart=0, bascount=None,
                  verbose=None):
        return make_mask(mol, coords, relativity, bastart, bascount, verbose)

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        return eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab, xctype, verbose)

    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', verbose=None):
        return eval_rho(mol, ao, dm, non0tab, xctype, verbose)

    def eval_xc(self, x_id, c_id, rho, spin=0, relativity=0, deriv=1, verbose=None):
        return eval_xc(x_id, c_id, rho, spin, relativity, deriv, verbose)

    def eval_x(self, x_id, rho, spin=0, relativity=0, deriv=1, verbose=None):
        return eval_x(x_id, rho, spin, relativity, deriv, verbose)

    def eval_c(self, c_id, rho, spin=0, relativity=0, deriv=1, verbose=None):
        return eval_c(c_id, rho, spin, relativity, deriv, verbose)

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _xc_type(x_id, c_id):
    if pyscf.dft.vxc.is_lda(x_id) and pyscf.dft.vxc.is_lda(c_id):
        xctype = 'LDA'
    elif pyscf.dft.vxc.is_meta_gga(x_id) or pyscf.dft.vxc.is_meta_gga(c_id):
        xctype = 'MGGA'
        raise NotImplementedError('meta-GGA')
    else:
        xctype = 'GGA'
    return xctype


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = '6311g*',)
    mf = dft.RKS(mol)
    mf.grids.atom_grid = {"H": (100, 194), "O": (100, 194),},
    mf.grids.setup_grids()
    dm = mf.get_init_guess(key='minao')

    x_code, c_code = pyscf.dft.vxc.parse_xc_name(mf.xc)
#res = vxc.nr_vxc(mol, mf.grids, x_code, c_code, dm, spin=1, relativity=0)
    print(time.clock())
    res = nr_vxc(mol, mf.grids, x_code, c_code, dm, spin=mol.spin, relativity=0)
    print(time.clock())

