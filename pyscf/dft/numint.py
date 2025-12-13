#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Numerical integration functions for RKS and UKS with real AO basis
'''

import warnings
import ctypes
import numpy
from pyscf import lib
try:
    from pyscf.dft import libxc
except (ImportError, OSError):
    try:
        from pyscf.dft import xcfun
        libxc = xcfun
    except (ImportError, OSError):
        warnings.warn('XC functional libraries (libxc or XCfun) are not available.')
        raise

from pyscf.dft.gen_grid import BLKSIZE, NBINS, CUTOFF, ALIGNMENT_UNIT, make_mask
from pyscf.dft import xc_deriv
from pyscf import __config__

libdft = lib.load_library('libdft')
OCCDROP = getattr(__config__, 'dft_numint_occdrop', 1e-12)
# The system size above which to consider the sparsity of the density matrix.
# If the number of AOs in the system is less than this value, all tensors are
# treated as dense quantities and contracted by dgemm directly.
SWITCH_SIZE = getattr(__config__, 'dft_numint_switch_size', 800)

# Whether to compute density laplacian for meta-GGA functionals
MGGA_DENSITY_LAPL = False

def eval_ao(mol, coords, deriv=0, shls_slice=None,
            non0tab=None, cutoff=None, out=None, verbose=None):
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
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in mol will be evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        cutoff : float
            AO values smaller than cutoff will be set to zero. The default
            cutoff threshold is ~1e-22 (defined in gto/grid_ao_drv.h)
        out : ndarray
            If provided, results are written into this array.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        2D array of shape (N,nao) for AO values if deriv = 0.
        Or 3D array of shape (:,N,nao) for AO values and AO derivatives if deriv > 0.
        In the 3D array, the first (N,nao) elements are the AO values,
        followed by (3,N,nao) for x,y,z components;
        Then 2nd derivatives (6,N,nao) for xx, xy, xz, yy, yz, zz;
        Then 3rd derivatives (10,N,nao) for xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
        ...

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords)
    >>> print(ao_value.shape)
    (100, 24)
    >>> ao_value = eval_ao(mol, coords, deriv=1, shls_slice=(1,4))
    >>> print(ao_value.shape)
    (4, 100, 7)
    >>> ao_value = eval_ao(mol, coords, deriv=2, shls_slice=(1,4))
    >>> print(ao_value.shape)
    (10, 100, 7)
    '''
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    if mol.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    return mol.eval_gto(feval, coords, comp, shls_slice, non0tab,
                        cutoff=cutoff, out=out)

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=True, verbose=None):
    r'''Calculate the electron density for LDA functional, and the density
    derivatives for GGA and MGGA functionals.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA (MGGA), ao[0] is AO value
            and ao[1:3] are the AO gradients. ao[4:10] are second derivatives of
            ao values if applicable.
        dm : 2D array
            Density matrix

    Kwargs:
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        hermi : bool
            dm is hermitian or not
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
        array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        or (5,N) (with_lapl=False) where the last row is tau = 1/2(\nabla f)^2

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords, deriv=0)
    >>> dm = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> dm = dm + dm.T
    >>> rho, dx_rho, dy_rho, dz_rho = eval_rho(mol, ao, dm, xctype='LDA')
    '''
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    if xctype == 'LDA' or xctype == 'HF':
        c0 = _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc)
        #:rho = numpy.einsum('pi,pi->p', ao, c0)
        rho = _contract_rho(ao, c0)
    elif xctype in ('GGA', 'NLC'):
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', c0, ao[0])
        rho[0] = _contract_rho(ao[0], c0)
        for i in range(1, 4):
            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            rho[i] = _contract_rho(ao[i], c0)
        if hermi:
            rho[1:4] *= 2  # *2 for + einsum('pi,ij,pj->p', ao[i], dm, ao[0])
        else:
            c1 = _dot_ao_dm(mol, ao[0], dm.conj().T, non0tab, shls_slice, ao_loc)
            for i in range(1, 4):
                rho[i] += _contract_rho(c1, ao[i])
    else: # meta-GGA
        if with_lapl:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            tau_idx = 5
        else:
            rho = numpy.empty((5,ngrids))
            tau_idx = 4
        c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        rho[0] = _contract_rho(ao[0], c0)

        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], dm, non0tab, shls_slice, ao_loc)
            #:rho[tau_idx] += numpy.einsum('pi,pi->p', c1, ao[i])
            rho[tau_idx] += _contract_rho(ao[i], c1)

            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            rho[i] = _contract_rho(ao[i], c0)
            if hermi:
                rho[i] *= 2
            else:
                rho[i] += _contract_rho(c1, ao[0])

        if with_lapl:
            if ao.shape[0] > 4:
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                # \nabla^2 rho
                #:rho[4] = numpy.einsum('pi,pi->p', c0, ao2)
                rho[4] = _contract_rho(ao2, c0)
                rho[4] += rho[5]
                if hermi:
                    rho[4] *= 2
                else:
                    c2 = _dot_ao_dm(mol, ao2, dm, non0tab, shls_slice, ao_loc)
                    rho[4] += _contract_rho(ao[0], c2)
                    rho[4] += rho[5]
            elif MGGA_DENSITY_LAPL:
                raise ValueError('Not enough derivatives in ao')
        # tau = 1/2 (\nabla f)^2
        rho[tau_idx] *= .5
    return rho

def eval_rho1(mol, ao, dm, screen_index=None, xctype='LDA', hermi=0,
              with_lapl=True, cutoff=None, ao_cutoff=CUTOFF, pair_mask=None,
              verbose=None):
    r'''Calculate the electron density for LDA and the density derivatives for
    GGA and MGGA with sparsity information.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA (MGGA), ao[0] is AO value
            and ao[1:3] are the AO gradients. ao[4:10] are second derivatives of
            ao values if applicable.
        dm : 2D array
            Density matrix

    Kwargs:
        screen_index : 2D uint8 array
            How likely the AO values on grids are negligible. This array can be
            obtained by calling :func:`gen_grid.make_screen_index`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        hermi : bool
            dm is hermitian or not
        cutoff : float
            cutoff for density value
        ao_cutoff : float
            cutoff for AO value. Needs to be the same to the cutoff when
            generating screen_index
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
        array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        or (5,N) (with_lapl=False) where the last row is tau = 1/2(\nabla f)^2
    '''
    if not (dm.dtype == ao.dtype == numpy.double):
        lib.logger.warn(mol, 'eval_rho1 does not support complex density, '
                        'eval_rho is called instead')
        return eval_rho(mol, ao, dm, screen_index, xctype, hermi, with_lapl, verbose)

    xctype = xctype.upper()
    ngrids = ao.shape[-2]

    if cutoff is None:
        cutoff = CUTOFF
    cutoff = min(cutoff, .1)
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(ao_cutoff))

    if pair_mask is None:
        ovlp_cond = mol.get_overlap_cond()
        pair_mask = numpy.asarray(ovlp_cond < -numpy.log(cutoff), dtype=numpy.uint8)

    ao_loc = mol.ao_loc_nr()
    if xctype == 'LDA' or xctype == 'HF':
        c0 = _dot_ao_dm_sparse(ao, dm, nbins, screen_index, pair_mask, ao_loc)
        rho = _contract_rho_sparse(ao, c0, screen_index, ao_loc)
    elif xctype in ('GGA', 'NLC'):
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, screen_index, pair_mask, ao_loc)
        rho[0] = _contract_rho_sparse(ao[0], c0, screen_index, ao_loc)
        for i in range(1, 4):
            rho[i] = _contract_rho_sparse(ao[i], c0, screen_index, ao_loc)
        if hermi:
            rho[1:4] *= 2  # *2 for + einsum('pi,ij,pj->p', ao[i], dm, ao[0])
        else:
            dm = lib.transpose(dm)
            c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, screen_index, pair_mask, ao_loc)
            for i in range(1, 4):
                rho[i] += _contract_rho_sparse(c0, ao[i], screen_index, ao_loc)
    else: # meta-GGA
        if with_lapl:
            if MGGA_DENSITY_LAPL:
                raise NotImplementedError('density laplacian not supported')
            rho = numpy.empty((6,ngrids))
            tau_idx = 5
        else:
            rho = numpy.empty((5,ngrids))
            tau_idx = 4
        c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, screen_index, pair_mask, ao_loc)
        rho[0] = _contract_rho_sparse(ao[0], c0, screen_index, ao_loc)

        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = _dot_ao_dm_sparse(ao[i], dm, nbins, screen_index, pair_mask, ao_loc)
            rho[tau_idx] += _contract_rho_sparse(ao[i], c1, screen_index, ao_loc)

            rho[i] = _contract_rho_sparse(ao[i], c0, screen_index, ao_loc)
            if hermi:
                rho[i] *= 2
            else:
                rho[i] += _contract_rho_sparse(ao[0], c1, screen_index, ao_loc)

        # tau = 1/2 (\nabla f)^2
        rho[tau_idx] *= .5
    return rho

def eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              with_lapl=True, verbose=None):
    r'''Calculate the electron density for LDA functional, and the density
    derivatives for GGA functional.  This function has the same functionality
    as :func:`eval_rho` except that the density are evaluated based on orbital
    coefficients and orbital occupancy.  It is more efficient than
    :func:`eval_rho` in most scenario.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA (MGGA), ao[0] is AO value
            and ao[1:3] are the AO gradients. ao[4:10] are second derivatives of
            ao values if applicable.
        dm : 2D array
            Density matrix

    Kwargs:
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        with_lapl: bool
            Whether to compute laplacian. It affects the shape of returns.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
        array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        or (5,N) (with_lapl=False) where the last row is tau = 1/2(\nabla f)^2
    '''
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    pos = mo_occ > OCCDROP
    if numpy.any(pos):
        cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(mol, ao, cpos, non0tab, shls_slice, ao_loc)
            #:rho = numpy.einsum('pi,pi->p', c0, c0)
            rho = _contract_rho(c0, c0)
        elif xctype in ('GGA', 'NLC'):
            rho = numpy.empty((4,ngrids))
            c0 = _dot_ao_dm(mol, ao[0], cpos, non0tab, shls_slice, ao_loc)
            #:rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            rho[0] = _contract_rho(c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, non0tab, shls_slice, ao_loc)
                #:rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                rho[i] = _contract_rho(c0, c1) * 2
        else: # meta-GGA
            if with_lapl:
                # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
                rho = numpy.empty((6,ngrids))
                tau_idx = 5
            else:
                rho = numpy.empty((5,ngrids))
                tau_idx = 4
            c0 = _dot_ao_dm(mol, ao[0], cpos, non0tab, shls_slice, ao_loc)
            #:rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            rho[0] = _contract_rho(c0, c0)

            rho[tau_idx] = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, non0tab, shls_slice, ao_loc)
                #:rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                #:rho[5] += numpy.einsum('pi,pi->p', c1, c1)
                rho[i] = _contract_rho(c0, c1) * 2
                rho[tau_idx] += _contract_rho(c1, c1)

            if with_lapl:
                if ao.shape[0] > 4:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    c1 = _dot_ao_dm(mol, ao2, cpos, non0tab, shls_slice, ao_loc)
                    #:rho[4] = numpy.einsum('pi,pi->p', c0, c1)
                    rho[4] = _contract_rho(c0, c1)
                    rho[4] += rho[5]
                    rho[4] *= 2
                else:
                    rho[4] = 0
            rho[tau_idx] *= .5
    else:
        if xctype == 'LDA' or xctype == 'HF':
            rho = numpy.zeros(ngrids)
        elif xctype in ('GGA', 'NLC'):
            rho = numpy.zeros((4,ngrids))
        else:  # meta-GGA
            if with_lapl:
                rho = numpy.zeros((6, ngrids))
            else:
                rho = numpy.zeros((5, ngrids))

    neg = mo_occ < -OCCDROP
    if numpy.any(neg):
        cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(mol, ao, cneg, non0tab, shls_slice, ao_loc)
            #:rho -= numpy.einsum('pi,pi->p', c0, c0)
            rho -= _contract_rho(c0, c0)
        elif xctype == 'GGA':
            c0 = _dot_ao_dm(mol, ao[0], cneg, non0tab, shls_slice, ao_loc)
            #:rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            rho[0] -= _contract_rho(c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, non0tab, shls_slice, ao_loc)
                #:rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                rho[i] -= _contract_rho(c0, c1) * 2 # *2 for +c.c.
        else:
            c0 = _dot_ao_dm(mol, ao[0], cneg, non0tab, shls_slice, ao_loc)
            #:rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            rho[0] -= _contract_rho(c0, c0)

            rho5 = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, non0tab, shls_slice, ao_loc)
                #:rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                #:rho5 += numpy.einsum('pi,pi->p', c1, c1)
                rho[i] -= _contract_rho(c0, c1) * 2 # *2 for +c.c.
                rho5 += _contract_rho(c1, c1)

            if with_lapl:
                if ao.shape[0] > 4:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    c1 = _dot_ao_dm(mol, ao2, cneg, non0tab, shls_slice, ao_loc)
                    #:rho[4] -= numpy.einsum('pi,pi->p', c0, c1) * 2
                    rho[4] -= _contract_rho(c0, c1) * 2
                    rho[4] -= rho5 * 2
                else:
                    rho[4] = 0

            rho[tau_idx] -= rho5 * .5
    return rho

def _vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    thresh=1e-8

    #output
    exc=numpy.zeros(rho[0,:].size)
    vxc=numpy.zeros([2,rho[0,:].size])

    #outer grid needs threshing
    threshind=rho[0,:]>=thresh
    coords=coords[threshind]
    R=rho[0,:][threshind]
    Gx=rho[1,:][threshind]
    Gy=rho[2,:][threshind]
    Gz=rho[3,:][threshind]
    G=Gx**2.+Gy**2.+Gz**2.

    #inner grid needs threshing
    innerthreshind=vvrho[0,:]>=thresh
    vvcoords=vvcoords[innerthreshind]
    vvweight=vvweight[innerthreshind]
    Rp=vvrho[0,:][innerthreshind]
    RpW=Rp*vvweight
    Gxp=vvrho[1,:][innerthreshind]
    Gyp=vvrho[2,:][innerthreshind]
    Gzp=vvrho[3,:][innerthreshind]
    Gp=Gxp**2.+Gyp**2.+Gzp**2.

    #constants and parameters
    Pi=numpy.pi
    Pi43=4.*Pi/3.
    Bvv, Cvv = nlc_pars
    Kvv=Bvv*1.5*Pi*((9.*Pi)**(-1./6.))
    Beta=((3./(Bvv*Bvv))**(0.75))/32.

    #inner grid
    W0p=Gp/(Rp*Rp)
    W0p=Cvv*W0p*W0p
    W0p=(W0p+Pi43*Rp)**0.5
    Kp=Kvv*(Rp**(1./6.))

    #outer grid
    W0tmp=G/(R**2)
    W0tmp=Cvv*W0tmp*W0tmp
    W0=(W0tmp+Pi43*R)**0.5
    dW0dR=(0.5*Pi43*R-2.*W0tmp)/W0
    dW0dG=W0tmp*R/(G*W0)
    K=Kvv*(R**(1./6.))
    dKdR=(1./6.)*K

    vvcoords = numpy.asarray(vvcoords, order='C')
    coords = numpy.asarray(coords, order='C')
    F = numpy.empty_like(R)
    U = numpy.empty_like(R)
    W = numpy.empty_like(R)
    #for i in range(R.size):
    #    DX=vvcoords[:,0]-coords[i,0]
    #    DY=vvcoords[:,1]-coords[i,1]
    #    DZ=vvcoords[:,2]-coords[i,2]
    #    R2=DX*DX+DY*DY+DZ*DZ
    #    gp=R2*W0p+Kp
    #    g=R2*W0[i]+K[i]
    #    gt=g+gp
    #    T=RpW/(g*gp*gt)
    #    F=numpy.sum(T)
    #    T*=(1./g+1./gt)
    #    U=numpy.sum(T)
    #    W=numpy.sum(T*R2)
    #    F*=-1.5
    libdft.VXC_vv10nlc(F.ctypes.data_as(ctypes.c_void_p),
                       U.ctypes.data_as(ctypes.c_void_p),
                       W.ctypes.data_as(ctypes.c_void_p),
                       vvcoords.ctypes.data_as(ctypes.c_void_p),
                       coords.ctypes.data_as(ctypes.c_void_p),
                       W0p.ctypes.data_as(ctypes.c_void_p),
                       W0.ctypes.data_as(ctypes.c_void_p),
                       K.ctypes.data_as(ctypes.c_void_p),
                       Kp.ctypes.data_as(ctypes.c_void_p),
                       RpW.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(vvcoords.shape[0]),
                       ctypes.c_int(coords.shape[0]))
    #exc is multiplied by Rho later
    exc[threshind] = Beta+0.5*F
    vxc[0,threshind] = Beta+F+1.5*(U*dKdR+W*dW0dR)
    vxc[1,threshind] = 1.5*W*dW0dG
    return exc,vxc

def eval_mat(mol, ao, weight, rho, vxc,
             non0tab=None, xctype='LDA', spin=0, verbose=None):
    r'''Calculate XC potential matrix.

    Args:
        mol : an instance of :class:`Mole`

        ao : ([4/10,] ngrids, nao) ndarray
            2D array of shape (N,nao) for LDA,
            3D array of shape (4,N,nao) for GGA
            or (10,N,nao) for meta-GGA.
            N is the number of grids, nao is the number of AO functions.
            If xctype is GGA (MGGA), ao[0] is AO value and ao[1:3] are the real space
            gradients. ao[4:10] are second derivatives of ao values if applicable.
        weight : 1D array
            Integral weights on grids.
        rho : ([4/6,] ngrids) ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))
        vxc : ([4,] ngrids) ndarray
            XC potential value on each grid = (vrho, vsigma, vlapl, vtau)
            vsigma is GGA potential value on each grid.
            If the kwarg spin != 0, a list [vsigma_uu,vsigma_ud] is required.

    Kwargs:
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of `ao` and `rho`
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        spin : int
            If not 0, the returned matrix is the Vxc matrix of alpha-spin.  It
            is computed with the spin non-degenerated UKS formula.

    Returns:
        XC potential matrix in 2D array of shape (nao,nao) where nao is the
        number of AO functions.
    '''
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    transpose_for_uks = False
    if xctype == 'LDA' or xctype == 'HF':
        if not isinstance(vxc, numpy.ndarray) or vxc.ndim == 2:
            vrho = vxc[0]
        else:
            vrho = vxc
        # *.5 because return mat + mat.T
        #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = _scale_ao(ao, .5*weight*vrho)
        mat = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)
    else:
        #wv = weight * vsigma * 2
        #aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
        vrho, vsigma = vxc[:2]
        if spin == 0:
            assert (vsigma is not None and rho.ndim==2)
            wv = _rks_gga_wv0(rho, vxc, weight)
        else:
            rho_a, rho_b = rho
            wv = numpy.empty((4,ngrids))
            wv[0] = weight * vrho * .5
            try:
                wv[1:4] = rho_a[1:4] * (weight * vsigma[0] * 2)  # sigma_uu
                wv[1:4]+= rho_b[1:4] * (weight * vsigma[1])      # sigma_ud
            except ValueError:
                warnings.warn('Note the output of libxc.eval_xc cannot be '
                              'directly used in eval_mat.\nvsigma from eval_xc '
                              'should be restructured as '
                              '(vsigma[:,0],vsigma[:,1])\n')
                transpose_for_uks = True
                vsigma = vsigma.T
                wv[1:4] = rho_a[1:4] * (weight * vsigma[0] * 2)  # sigma_uu
                wv[1:4]+= rho_b[1:4] * (weight * vsigma[1])      # sigma_ud
        #:aow = numpy.einsum('npi,np->pi', ao[:4], wv)
        aow = _scale_ao(ao[:4], wv)
        mat = _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)

# JCP 138, 244108 (2013); DOI:10.1063/1.4811270
# JCP 112, 7002 (2000); DOI:10.1063/1.481298
    if xctype == 'MGGA':
        vlapl, vtau = vxc[2:]

        if vlapl is None:
            if spin != 0:
                if transpose_for_uks:
                    vtau = vtau.T
                vtau = vtau[0]
            wv = weight * .25 * vtau
            mat += _tau_dot(mol, ao, ao, wv, non0tab, shls_slice, ao_loc)
        else:
            if spin != 0:
                if transpose_for_uks:
                    vlapl = vlapl.T
                vlapl = vlapl[0]
            if ao.shape[0] > 4:
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                #:aow = numpy.einsum('pi,p->pi', ao2, .5 * weight * vlapl, out=aow)
                aow = _scale_ao(ao2, .5 * weight * vlapl, out=aow)
                mat += _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)
            else:
                raise ValueError('Not enough derivatives in ao')

    return mat + mat.T.conj()

def _dot_ao_ao(mol, ao1, ao2, non0tab, shls_slice, ao_loc, hermi=0):
    '''return numpy.dot(ao1.T, ao2)'''
    ngrids, nao = ao1.shape
    if (nao < SWITCH_SIZE or
        non0tab is None or shls_slice is None or ao_loc is None):
        return lib.dot(ao1.conj().T, ao2)

    if not ao1.flags.f_contiguous:
        ao1 = lib.transpose(ao1)
    if not ao2.flags.f_contiguous:
        ao2 = lib.transpose(ao2)
    if ao1.dtype == ao2.dtype == numpy.double:
        fn = libdft.VXCdot_ao_ao
    else:
        fn = libdft.VXCzdot_ao_ao
        ao1 = numpy.asarray(ao1, numpy.complex128)
        ao2 = numpy.asarray(ao2, numpy.complex128)

    vv = numpy.empty((nao,nao), dtype=ao1.dtype)
    fn(vv.ctypes.data_as(ctypes.c_void_p),
       ao1.ctypes.data_as(ctypes.c_void_p),
       ao2.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(ngrids),
       ctypes.c_int(mol.nbas), ctypes.c_int(hermi),
       non0tab.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p))
    return vv

def _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    '''return numpy.dot(ao, dm)'''
    ngrids, nao = ao.shape
    if (nao < SWITCH_SIZE or
        non0tab is None or shls_slice is None or ao_loc is None):
        return lib.dot(dm.T, ao.T).T

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if ao.dtype == dm.dtype == numpy.double:
        fn = libdft.VXCdot_ao_dm
    else:
        fn = libdft.VXCzdot_ao_dm
        ao = numpy.asarray(ao, numpy.complex128)
        dm = numpy.asarray(dm, numpy.complex128)

    vm = numpy.ndarray((ngrids,dm.shape[1]), dtype=ao.dtype, order='F', buffer=out)
    dm = numpy.asarray(dm, order='C')
    fn(vm.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       dm.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       non0tab.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p))
    return vm

def _scale_ao(ao, wv, out=None):
    #:aow = numpy.einsum('npi,np->pi', ao[:4], wv)
    if wv.ndim == 2:
        ao = ao.transpose(0,2,1)
    else:
        ngrids, nao = ao.shape
        ao = ao.T.reshape(1,nao,ngrids)
        wv = wv.reshape(1,ngrids)

    if not ao.flags.c_contiguous:
        return numpy.einsum('nip,np->pi', ao, wv)

    if ao.dtype == numpy.double:
        if wv.dtype == numpy.double:
            fn = libdft.VXC_dscale_ao
            dtype = numpy.double
        elif wv.dtype == numpy.complex128:
            fn = libdft.VXC_dzscale_ao
            dtype = numpy.complex128
        else:
            return numpy.einsum('nip,np->pi', ao, wv)
    elif ao.dtype == numpy.complex128:
        if wv.dtype == numpy.double:
            fn = libdft.VXC_zscale_ao
            dtype = numpy.complex128
        elif wv.dtype == numpy.complex128:
            fn = libdft.VXC_zzscale_ao
            dtype = numpy.complex128
        else:
            return numpy.einsum('nip,np->pi', ao, wv)
    else:
        return numpy.einsum('nip,np->pi', ao, wv)

    wv = numpy.asarray(wv, order='C')
    comp, nao, ngrids = ao.shape
    assert wv.shape[0] == comp
    aow = numpy.ndarray((nao,ngrids), dtype=dtype, buffer=out).T
    fn(aow.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       wv.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(comp), ctypes.c_int(nao),
       ctypes.c_int(ngrids))
    return aow

def _contract_rho(bra, ket):
    '''Real part of rho for rho=einsum('pi,pi->p', bra.conj(), ket)'''
    bra = bra.T
    ket = ket.T
    nao, ngrids = bra.shape
    rho = numpy.empty(ngrids)

    if not (bra.flags.c_contiguous and ket.flags.c_contiguous):
        rho  = numpy.einsum('ip,ip->p', bra.real, ket.real)
        rho += numpy.einsum('ip,ip->p', bra.imag, ket.imag)
    elif bra.dtype == numpy.double and ket.dtype == numpy.double:
        libdft.VXC_dcontract_rho(rho.ctypes.data_as(ctypes.c_void_p),
                                 bra.ctypes.data_as(ctypes.c_void_p),
                                 ket.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(nao), ctypes.c_int(ngrids))
    elif bra.dtype == numpy.complex128 and ket.dtype == numpy.complex128:
        libdft.VXC_zcontract_rho(rho.ctypes.data_as(ctypes.c_void_p),
                                 bra.ctypes.data_as(ctypes.c_void_p),
                                 ket.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(nao), ctypes.c_int(ngrids))
    else:
        rho  = numpy.einsum('ip,ip->p', bra.real, ket.real)
        rho += numpy.einsum('ip,ip->p', bra.imag, ket.imag)
    return rho

def _tau_dot(mol, bra, ket, wv, mask, shls_slice, ao_loc):
    '''nabla_ao dot nabla_ao
    numpy.einsum('p,xpi,xpj->ij', wv, bra[1:4].conj(), ket[1:4])
    '''
    aow = _scale_ao(ket[1], wv)
    mat = _dot_ao_ao(mol, bra[1], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ket[2], wv, aow)
    mat += _dot_ao_ao(mol, bra[2], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ket[3], wv, aow)
    mat += _dot_ao_ao(mol, bra[3], aow, mask, shls_slice, ao_loc)
    return mat

def _sparse_enough(screen_index, threshold=0.5):
    # TODO: improve the turnover threshold
    return numpy.count_nonzero(screen_index) < screen_index.size * threshold

def _dot_ao_ao_dense(ao1, ao2, wv, out=None):
    '''Returns (bra*wv).T.dot(ket)
    '''
    assert ao1.flags.f_contiguous
    assert ao2.flags.f_contiguous
    assert ao1.dtype == ao2.dtype == numpy.double
    ngrids, nao = ao1.shape
    if out is None:
        out = numpy.zeros((nao, nao), dtype=ao1.dtype)

    if wv is None:
        return lib.ddot(ao1.T, ao2, 1, out, 1)
    else:
        assert wv.dtype == numpy.double
        ao1 = _scale_ao(ao1, wv.ravel())
        return lib.ddot(ao1.T, ao2, 1, out, 1)

def _dot_ao_ao_sparse(ao1, ao2, wv, nbins, screen_index, pair_mask, ao_loc,
                      hermi=0, out=None):
    '''Returns (bra*wv).T.dot(ket) while sparsity is explicitly considered.
    Note the return may have ~1e-13 difference to _dot_ao_ao.
    '''
    ngrids, nao = ao1.shape
    if screen_index is None or pair_mask is None or ngrids % ALIGNMENT_UNIT != 0:
        return _dot_ao_ao_dense(ao1, ao2, wv, out)

    assert ao1.flags.f_contiguous
    assert ao2.flags.f_contiguous
    assert ao1.dtype == ao2.dtype == numpy.double
    nbas = screen_index.shape[1]
    if out is None:
        out = numpy.zeros((nao, nao), dtype=ao1.dtype)

    if wv is None:
        libdft.VXCdot_ao_ao_sparse(
            out.ctypes.data_as(ctypes.c_void_p),
            ao1.ctypes.data_as(ctypes.c_void_p),
            ao2.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(nbas), ctypes.c_int(hermi),
            ctypes.c_int(nbins), screen_index.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
    else:
        assert wv.dtype == numpy.double
        libdft.VXCdot_aow_ao_sparse(
            out.ctypes.data_as(ctypes.c_void_p),
            ao1.ctypes.data_as(ctypes.c_void_p),
            ao2.ctypes.data_as(ctypes.c_void_p),
            wv.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(nbas), ctypes.c_int(hermi),
            ctypes.c_int(nbins), screen_index.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
    return out

def _dot_ao_dm_sparse(ao, dm, nbins, screen_index, pair_mask, ao_loc):
    '''Returns numpy.dot(ao, dm) while sparsity is explicitly considered.
    Note the return may be different to _dot_ao_dm. After contracting to another
    ao matrix, (numpy.dot(ao, dm)*ao).sum(axis=1), their value can be matched up
    to ~1e-13.
    '''
    ngrids, nao = ao.shape
    if screen_index is None or pair_mask is None or ngrids % ALIGNMENT_UNIT != 0:
        return lib.dot(dm.T, ao.T).T

    assert ao.flags.f_contiguous
    assert ao.dtype == dm.dtype == numpy.double
    nbas = screen_index.shape[1]
    dm = numpy.asarray(dm, order='C')
    out = _empty_aligned((nao, ngrids)).T

    fn = libdft.VXCdot_ao_dm_sparse
    fn(out.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       dm.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
       ctypes.c_int(nbins), screen_index.ctypes.data_as(ctypes.c_void_p),
       pair_mask.ctypes.data_as(ctypes.c_void_p),
       ao_loc.ctypes.data_as(ctypes.c_void_p))
    return out

def _scale_ao_sparse(ao, wv, screen_index, ao_loc, out=None):
    '''Returns einsum('xgi,xg->gi', ao, wv) while sparsity is explicitly considered.
    Note the return may be different to _scale_ao. After contracting to another
    ao matrix, scale_ao.T.dot(ao), their value can be matched up to ~1e-13.
    '''
    if screen_index is None:
        return _scale_ao(ao, wv, out=out)

    assert ao.dtype == wv.dtype == numpy.double
    if ao.ndim == 3:
        assert ao[0].flags.f_contiguous
        ngrids, nao = ao[0].shape
        comp = wv.shape[0]
    else:
        assert ao.flags.f_contiguous
        ngrids, nao = ao.shape
        comp = 1
    nbas = screen_index.shape[1]
    if ngrids % ALIGNMENT_UNIT != 0:
        return _scale_ao(ao, wv, out=out)

    if out is None:
        out = _empty_aligned((nao, ngrids)).T
    else:
        out = numpy.ndarray((ngrids, nao), buffer=out, order='F')
    libdft.VXCdscale_ao_sparse(
        out.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), ctypes.c_int(nao),
        ctypes.c_int(ngrids), ctypes.c_int(nbas),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return out

def _contract_rho_sparse(bra, ket, screen_index, ao_loc):
    '''Returns numpy.einsum('gi,gi->g', bra, ket) while sparsity is explicitly
    considered. Note the return may have ~1e-13 difference to _contract_rho.
    '''
    ngrids, nao = bra.shape
    if screen_index is None or ngrids % ALIGNMENT_UNIT != 0:
        return _contract_rho(bra, ket)

    assert bra.flags.f_contiguous
    assert ket.flags.f_contiguous
    assert bra.dtype == ket.dtype == numpy.double
    nbas = screen_index.shape[1]
    rho = numpy.empty(ngrids)
    libdft.VXCdcontract_rho_sparse(
        rho.ctypes.data_as(ctypes.c_void_p),
        bra.ctypes.data_as(ctypes.c_void_p),
        ket.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return rho

def _tau_dot_sparse(bra, ket, wv, nbins, screen_index, pair_mask, ao_loc, out=None):
    '''Similar to _tau_dot, while sparsity is explicitly considered. Note the
    return may have ~1e-13 difference to _tau_dot.
    '''
    nao = bra.shape[1]
    if out is None:
        out = numpy.zeros((nao, nao), dtype=bra.dtype)
    hermi = 1
    _dot_ao_ao_sparse(bra[1], ket[1], wv, nbins, screen_index, pair_mask,
                      ao_loc, hermi, out)
    _dot_ao_ao_sparse(bra[2], ket[2], wv, nbins, screen_index, pair_mask,
                      ao_loc, hermi, out)
    _dot_ao_ao_sparse(bra[3], ket[3], wv, nbins, screen_index, pair_mask,
                      ao_loc, hermi, out)
    return out

def nr_vxc(mol, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    '''
    Evaluate RKS/UKS XC functional and potential matrix on given meshgrids
    for a set of density matrices.  See :func:`nr_rks` and :func:`nr_uks`
    for more details.

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array or a list of 2D arrays
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
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((2,nao,nao))
    >>> nelec, exc, vxc = dft.numint.nr_vxc(mol, grids, 'lda,vwn', dm, spin=1)
    '''
    ni = NumInt()
    return ni.nr_vxc(mol, grids, xc_code, dms, spin, relativity,
                     hermi, max_memory, verbose)

def nr_sap_vxc(ni, mol, grids, max_memory=2000, verbose=None):
    '''Calculate superposition of atomic potentials matrix on given meshgrids.

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.

    Kwargs:
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        vmat is the XC potential matrix in 2D array of shape (nao,nao)
        where nao is the number of AO functions.

    Examples:
    >>> import numpy
    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> ni = dft.numint.NumInt()
    >>> vsap = ni.nr_sap(mol, grids)
    '''
    from pyscf.dft.sap import sap_effective_charge
    assert not mol.has_ecp(), 'ECP or PP not supported'
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    nao = mol.nao
    vmat = numpy.zeros((nao,nao))
    aow = None
    ao_deriv = 0

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()

    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv,
                                                  max_memory=max_memory):
        vxc = numpy.zeros(weight.size)
        # Form potential
        for ia, z in enumerate(atom_charges):
            rnuc = numpy.linalg.norm(atom_coords[ia] - coords, axis=1)
            Zeff = sap_effective_charge(z, rnuc)
            vxc -= Zeff/rnuc

        aow = _scale_ao(ao, weight*vxc, out=aow)
        vmat += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
        vxc = None

    return vmat

def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix on given meshgrids
    for a set of density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array or a list of 2D arrays
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
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
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((nao,nao))
    >>> ni = dft.numint.NumInt()
    >>> nelec, exc, vxc = ni.nr_rks(mol, grids, 'lda,vwn', dm)
    '''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset,nao,nao))

    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
                if xctype == 'LDA':
                    den = rho * weight
                else:
                    den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    aow = None
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 because vmat + vmat.T at the end
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))

    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            wv[4] *= .5  # *.5 for 1/2 in tau
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))
        vmat += v1

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_rks for functional {xc_code}')

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
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
        dms : a list of 2D arrays
            A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of (alpha,beta) electrons generated by numerical integration.
        excsum is the XC functional value.
        vmat is the XC potential matrix for (alpha,beta) spin.

    Examples:

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> grids.coords = numpy.random.random((100,3))  # 100 random points
    >>> grids.weights = numpy.random.random(100)
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((2,nao,nao))
    >>> ni = dft.numint.NumInt()
    >>> nelec, exc, vxc = ni.nr_uks(mol, grids, 'lda,vwn', dm)
    '''
    xctype = ni._xc_type(xc_code)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((2,nset,nao,nao))

    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao, mask, xctype)
                rho_b = make_rhob(i, ao, mask, xctype)
                rho = (rho_a, rho_b)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[:2]
                if xctype == 'LDA':
                    den_a = rho_a * weight
                    den_b = rho_b * weight
                else:
                    den_a = rho_a[0] * weight
                    den_b = rho_b[0] * weight
                nelec[0,i] += den_a.sum()
                nelec[1,i] += den_b.sum()
                excsum[i] += numpy.dot(den_a, exc)
                excsum[i] += numpy.dot(den_b, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[0,i])
            _dot_ao_ao_sparse(ao, ao, wv[1,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[1,i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wva, wvb = wv
            aow = _scale_ao_sparse(ao, wva, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[0,i])
            aow = _scale_ao_sparse(ao, wvb, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[1,i])
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)

    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wv[:,4] *= .5
            wva, wvb = wv
            aow = _scale_ao_sparse(ao[:4], wva[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[0,i])
            _tau_dot_sparse(ao, ao, wva[4], nbins, mask, pair_mask, ao_loc, out=v1[0,i])
            aow = _scale_ao_sparse(ao[:4], wvb[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[1,i])
            _tau_dot_sparse(ao, ao, wvb[4], nbins, mask, pair_mask, ao_loc, out=v1[1,i])
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)
        vmat += v1
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')

    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = numpy.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat

def _format_uks_dm(dms):
    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:  # RHF DM
        dma = dmb = dms * .5
    else:
        dma, dmb = dms
    if getattr(dms, 'mo_coeff', None) is not None:
        mo_coeff = dms.mo_coeff
        mo_occ = dms.mo_occ
        if mo_coeff[0].ndim < dma.ndim: # handle ROKS
            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
            dma = lib.tag_array(dma, mo_coeff=mo_coeff, mo_occ=mo_occa)
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff, mo_occ=mo_occb)
        else:
            dma = lib.tag_array(dma, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
    return dma, dmb

nr_rks_vxc = nr_rks
nr_uks_vxc = nr_uks

def nr_nlc_vxc(ni, mol, grids, xc_code, dm, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
    '''Calculate NLC functional and potential matrix on given grids

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm : 2D array
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    assert nset == 1
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    ao_deriv = 1
    vvrho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        vvrho.append(make_rho(0, ao, mask, 'GGA'))
    rho = numpy.hstack(vvrho)

    exc = 0
    vxc = 0
    nlc_coefs = ni.nlc_coeff(xc_code)
    for nlc_pars, fac in nlc_coefs:
        e, v = _vv10nlc(rho, grids.coords, rho, grids.weights,
                        grids.coords, nlc_pars)
        exc += e * fac
        vxc += v * fac
    den = rho[0] * grids.weights
    nelec = den.sum()
    excsum = numpy.dot(den, exc)
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    aow = None
    vmat = numpy.zeros((nao,nao))
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5
        aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
        _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                          hermi=0, out=vmat)
    vmat = vmat + vmat.T
    return nelec, excsum, vmat

def nr_rks_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Contract RKS XC (singlet hessian) kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm0 : 2D array
            Zeroth order density matrix
        dms : 2D array a list of 2D arrays
            First order density matrix or density matrices

    Kwargs:
        hermi : int
            First order density matrix symmetric or not. It also indicates
            whether the matrices in return are symmetric or not.
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
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if hermi != 1 and dtype != numpy.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)
    if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=0,
                                  max_memory=max_memory)[2]

    make_rho1, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[:,:,p0:p1]
            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = weight * rho1 * _fxc[0]
                else:
                    wv = numpy.einsum('yg,xyg,g->xg', rho1, _fxc, weight)
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((nset,nao,nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])

        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            wv[4] *= .5  # *.5 for 1/2 in tau
            aow = _scale_ao_sparse(ao, wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))
        vmat += v1

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat

def nr_rks_fxc_st(ni, mol, grids, xc_code, dm0, dms_alpha, hermi=0, singlet=True,
                  rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Associated to singlet or triplet Hessian
    Note the difference to nr_rks_fxc, dms_alpha is the response density
    matrices of alpha spin, alpha+/-beta DM is applied due to singlet/triplet
    coupling

    Ref. CPL, 256, 454
    '''
    if fxc is None:
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=1,
                                  max_memory=max_memory)[2]
    if singlet:
        fxc = fxc[0,:,0] + fxc[0,:,1]
    else:
        fxc = fxc[0,:,0] - fxc[0,:,1]
    return ni.nr_rks_fxc(mol, grids, xc_code, dm0, dms_alpha, hermi=hermi, fxc=fxc,
                         max_memory=max_memory)

def _rks_gga_wv0(rho, vxc, weight):
    vrho, vgamma = vxc[:2]
    ngrid = vrho.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = vrho * .5  # v+v.T should be applied in the caller
    wv[1:] = 2 * vgamma * rho[1:4]
    wv[:] *= weight
    return wv

def _rks_gga_wv1(rho0, rho1, vxc, fxc, weight):
    vgamma = vxc[1]
    frho, frhogamma, fgg = fxc[:3]
    # sigma1 ~ \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    ngrid = sigma1.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = frho * rho1[0]
    wv[0] += frhogamma * sigma1 * 2
    wv[1:] = (fgg * sigma1 * 4 + frhogamma * rho1[0] * 2) * rho0[1:4]
    wv[1:]+= vgamma * rho1[1:4] * 2
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0] *= .5
    return wv

def _rks_gga_wv2(rho0, rho1, fxc, kxc, weight):
    frr, frg, fgg = fxc[:3]
    frrr, frrg, frgg, fggg = kxc[:4]
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    r1r1 = rho1[0]**2
    s1s1 = sigma1**2
    r1s1 = rho1[0] * sigma1
    sigma2 = numpy.einsum('xi,xi->i', rho1[1:4], rho1[1:4])
    ngrid = sigma1.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = frrr * r1r1
    wv[0] += 4 * frrg * r1s1
    wv[0] += 4 * frgg * s1s1
    wv[0] += 2 * frg * sigma2
    wv[1:4]  = 2 * frrg * r1r1 * rho0[1:4]
    wv[1:4] += 8 * frgg * r1s1 * rho0[1:4]
    wv[1:4] += 4 * frg * rho1[0] * rho1[1:4]
    wv[1:4] += 4 * fgg * sigma2 * rho0[1:4]
    wv[1:4] += 8 * fgg * sigma1 * rho1[1:4]
    wv[1:4] += 8 * fggg * s1s1 * rho0[1:4]
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0]*=.5
    return wv

def _rks_mgga_wv0(rho, vxc, weight):
    vrho, vgamma, vlapl, vtau = vxc[:4]
    ngrid = vrho.size
    wv = numpy.zeros((6,ngrid))
    wv[0] = weight * vrho
    wv[1:4] = (weight * vgamma * 2) * rho[1:4]
    # *0.5 is for tau = 1/2 \nabla\phi\dot\nabla\phi
    wv[5] = weight * vtau * .5
    # *0.5 because v+v.T should be applied in the caller
    wv[0] *= .5
    wv[5] *= .5
    return wv

def _rks_mgga_wv1(rho0, rho1, vxc, fxc, weight):
    vsigma = vxc[1]
    frr, frg, fgg, fll, ftt, frl, frt, flt, fgl, fgt = fxc
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    ngrids = sigma1.size
    wv = numpy.zeros((6, ngrids))
    wv[0]  = frr * rho1[0]
    wv[0] += frt * rho1[5]
    wv[0] += frg * sigma1 * 2
    wv[1:4] = (fgg * sigma1 * 4 + frg * rho1[0] * 2 + fgt * rho1[5] * 2) * rho0[1:4]
    wv[1:4]+= vsigma * rho1[1:4] * 2
    wv[5]  = ftt * rho1[5] * .5
    wv[5] += frt * rho1[0] * .5
    wv[5] += fgt * sigma1
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0] *= .5
    wv[5] *= .5
    return wv

def _rks_mgga_wv2(rho0, rho1, fxc, kxc, weight):
    frr, frg, fgg, fll, ftt, frl, frt, flt, fgl, fgt = fxc
    frrr, frrg, frgg, fggg = kxc[:4]
    frrt = kxc[5]
    frgt = kxc[7]
    frtt = kxc[10]
    fggt = kxc[12]
    fgtt = kxc[15]
    fttt = kxc[19]
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    r1r1 = rho1[0]**2
    t1t1 = rho1[5]**2
    r1t1 = rho1[0] * rho1[5]
    s1s1 = sigma1**2
    r1s1 = rho1[0] * sigma1
    s1t1 = sigma1 * rho1[5]
    sigma2 = numpy.einsum('xi,xi->i', rho1[1:4], rho1[1:4])

    ngrid = sigma1.size
    wv = numpy.zeros((6, ngrid))
    wv[0]  = frrr * r1r1
    wv[0] += 4 * frrg * r1s1
    wv[0] += 4 * frgg * s1s1
    wv[0] += 2 * frg * sigma2
    wv[0] += frtt * t1t1
    wv[0] += 2 * frrt * r1t1
    wv[0] += 4 * frgt * s1t1
    wv[1:4] += 2 * frrg * r1r1 * rho0[1:4]
    wv[1:4] += 8 * frgg * r1s1 * rho0[1:4]
    wv[1:4] += 4 * fgg * sigma2 * rho0[1:4]
    wv[1:4] += 8 * fggg * s1s1 * rho0[1:4]
    wv[1:4] += 2 * fgtt * t1t1 * rho0[1:4]
    wv[1:4] += 8 * fggt * s1t1 * rho0[1:4]
    wv[1:4] += 4 * frgt * r1t1 * rho0[1:4]
    wv[1:4] += 8 * fgg * sigma1 * rho1[1:4]
    wv[1:4] += 4 * frg * rho1[0] * rho1[1:4]
    wv[1:4] += 4 * fgt * rho1[5] * rho1[1:4]
    wv[5] += fttt * t1t1 * .5
    wv[5] += frtt * r1t1
    wv[5] += frrt * r1r1 * .5
    wv[5] += fgtt * s1t1 * 2
    wv[5] += fggt * s1s1 * 2
    wv[5] += frgt * r1s1 * 2
    wv[5] += fgt * sigma2
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0] *= .5
    wv[5] *= .5
    return wv

def nr_uks_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Contract UKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm0 : (2, N, N) array
            Zeroth order density matrices
        dms : 2D array a list of 2D arrays
            First order density matrices

    Kwargs:
        hermi : int
            First order density matrix symmetric or not. It also indicates
            whether the matrices in return are symmetric or not.
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
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if hermi != 1 and dtype != numpy.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)
    if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=1,
                                  max_memory=max_memory)[2]

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[:,:,:,:,p0:p1]
            for i in range(nset):
                rho1a = make_rhoa(i, ao, mask, xctype)
                rho1b = make_rhob(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = rho1a * _fxc[0,0] + rho1b * _fxc[1,0]
                else:
                    wv  = numpy.einsum('xg,xbyg->byg', rho1a, _fxc[0])
                    wv += numpy.einsum('xg,xbyg->byg', rho1b, _fxc[1])
                wv *= weight
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((2,nset,nao,nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[0,i])
            _dot_ao_ao_sparse(ao, ao, wv[1,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[1,i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wva, wvb = wv
            aow = _scale_ao_sparse(ao, wva, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[0,i])
            aow = _scale_ao_sparse(ao, wvb, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[1,i])

        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wv[:,4] *= .5
            wva, wvb = wv
            aow = _scale_ao_sparse(ao[:4], wva[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[0,i])
            _tau_dot_sparse(ao, ao, wva[4], nbins, mask, pair_mask, ao_loc, out=v1[0,i])
            aow = _scale_ao_sparse(ao[:4], wvb[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[1,i])
            _tau_dot_sparse(ao, ao, wvb[4], nbins, mask, pair_mask, ao_loc, out=v1[1,i])
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)
        vmat += v1

    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat

def _uks_gga_wv0(rho, vxc, weight):
    rhoa, rhob = rho
    vrho, vsigma = vxc[:2]
    ngrids = vrho.shape[0]
    wva, wvb = numpy.empty((2, 4, ngrids))
    wva[0]  = vrho[:,0] * .5  # v+v.T should be applied in the caller
    wva[1:] = rhoa[1:4] * vsigma[:,0] * 2  # sigma_uu
    wva[1:]+= rhob[1:4] * vsigma[:,1]      # sigma_ud
    wva[:] *= weight
    wvb[0]  = vrho[:,1] * .5  # v+v.T should be applied in the caller
    wvb[1:] = rhob[1:4] * vsigma[:,2] * 2  # sigma_dd
    wvb[1:]+= rhoa[1:4] * vsigma[:,1]      # sigma_ud
    wvb[:] *= weight
    return wva, wvb

def _uks_gga_wv1(rho0, rho1, vxc, fxc, weight):
    uu, ud, dd = vxc[1].T
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    ngrid = uu.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    ab_1 = a0b1 + b0a1

    wva, wvb = numpy.empty((2,4,ngrid))
    # alpha = alpha-alpha * alpha
    wva[0]  = u_u * rho1a[0]
    wva[0] += u_uu * a0a1
    wva[0] += u_ud * ab_1
    wva[1:] = uu * rho1a[1:4] * 2
    wva[1:]+= u_uu * rho1a[0] * rho0a[1:4] * 2
    wva[1:]+= u_ud * rho1a[0] * rho0b[1:4]
    wva[1:]+= uu_uu * a0a1 * rho0a[1:4] * 2
    wva[1:]+= uu_ud * a0a1 * rho0b[1:4]
    wva[1:]+= uu_ud * ab_1 * rho0a[1:4] * 2
    wva[1:]+= ud_ud * ab_1 * rho0b[1:4]
    # alpha = alpha-beta  * beta
    wva[0] += u_d * rho1b[0]
    wva[0] += u_dd * b0b1
    wva[1:]+= ud * rho1b[1:4]
    wva[1:]+= d_uu * rho1b[0] * rho0a[1:4] * 2
    wva[1:]+= d_ud * rho1b[0] * rho0b[1:4]
    wva[1:]+= uu_dd * b0b1 * rho0a[1:4] * 2
    wva[1:]+= ud_dd * b0b1 * rho0b[1:4]
    wva *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1a.dtype == numpy.double
    wva[0] *= .5

    # beta = beta-alpha * alpha
    wvb[0]  = u_d * rho1a[0]
    wvb[0] += d_ud * ab_1
    wvb[0] += d_uu * a0a1
    wvb[1:] = ud * rho1a[1:4]
    wvb[1:]+= u_dd * rho1a[0] * rho0b[1:4] * 2
    wvb[1:]+= u_ud * rho1a[0] * rho0a[1:4]
    wvb[1:]+= ud_dd * ab_1 * rho0b[1:4] * 2
    wvb[1:]+= ud_ud * ab_1 * rho0a[1:4]
    wvb[1:]+= uu_dd * a0a1 * rho0b[1:4] * 2
    wvb[1:]+= uu_ud * a0a1 * rho0a[1:4]
    # beta = beta-beta  * beta
    wvb[0] += d_d * rho1b[0]
    wvb[0] += d_dd * b0b1
    wvb[1:]+= dd * rho1b[1:4] * 2
    wvb[1:]+= d_dd * rho1b[0] * rho0b[1:4] * 2
    wvb[1:]+= d_ud * rho1b[0] * rho0a[1:4]
    wvb[1:]+= dd_dd * b0b1 * rho0b[1:4] * 2
    wvb[1:]+= ud_dd * b0b1 * rho0a[1:4]
    wvb *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1b.dtype == numpy.double
    wvb[0] *= .5
    return wva, wvb

def _uks_gga_wv2(rho0, rho1, fxc, kxc, weight):
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    u_u_u, u_u_d, u_d_d, d_d_d = kxc[0].T
    u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, \
            d_d_ud, d_d_dd = kxc[1].T
    u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, \
            d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd = kxc[2].T
    uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, \
            ud_ud_dd, ud_dd_dd, dd_dd_dd = kxc[3].T
    ngrid = u_u.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    a1a1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1a[1:4]) * 2
    a1b1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1b[1:4]) * 2
    b1b1 = numpy.einsum('xi,xi->i', rho1b[1:4], rho1b[1:4]) * 2
    rara = rho1a[0] * rho1a[0]
    rarb = rho1a[0] * rho1b[0]
    rbrb = rho1b[0] * rho1b[0]
    ab_1 = a0b1 + b0a1

    wva, wvb = numpy.zeros((2, 4, ngrid))
    wva[0] += u_u_u * rho1a[0] * rho1a[0]
    wva[0] += u_u_d * rho1a[0] * rho1b[0] * 2
    wva[0] += u_d_d * rho1b[0] * rho1b[0]
    wva[0] += u_uu * a1a1
    wva[0] += u_ud * a1b1
    wva[0] += u_dd * b1b1
    wva[1:4] += u_uu * rho1a[0] * rho1a[1:4] * 4
    wva[1:4] += u_ud * rho1a[0] * rho1b[1:4] * 2
    wva[1:4] += d_uu * rho1b[0] * rho1a[1:4] * 4
    wva[1:4] += d_ud * rho1b[0] * rho1b[1:4] * 2
    wva[1:4] += uu_uu * a1a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu * a0a1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * ab_1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * a1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_ud * a1a1 * rho0b[1:4]
    wva[1:4] += uu_ud * a0a1 * rho1b[1:4] * 2
    wva[1:4] += uu_dd * b1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_dd * b0b1 * rho1a[1:4] * 4
    wva[1:4] += ud_ud * ab_1 * rho1b[1:4] * 2
    wva[1:4] += ud_ud * a1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b0b1 * rho1b[1:4] * 2
    wva[0] += u_u_uu * rho1a[0] * a0a1 * 2
    wva[0] += u_d_uu * rho1b[0] * a0a1 * 2
    wva[0] += u_u_ud * rho1a[0] * ab_1 * 2
    wva[0] += u_d_ud * rho1b[0] * ab_1 * 2
    wva[0] += u_u_dd * rho1a[0] * b0b1 * 2
    wva[0] += u_d_dd * rho1b[0] * b0b1 * 2
    wva[1:4] += u_u_uu * rara * rho0a[1:4] * 2
    wva[1:4] += u_u_ud * rara * rho0b[1:4]
    wva[1:4] += u_d_uu * rarb * rho0a[1:4] * 4
    wva[1:4] += u_d_ud * rarb * rho0b[1:4] * 2
    wva[1:4] += d_d_uu * rbrb * rho0a[1:4] * 2
    wva[1:4] += d_d_ud * rbrb * rho0b[1:4]
    wva[1:4] += u_uu_uu * rho1a[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_uu * rho1b[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += u_uu_dd * rho1a[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_dd * rho1b[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0b[1:4] * 2
    wva[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0b[1:4] * 2
    wva[0] += u_uu_uu * a0a1 * a0a1
    wva[0] += u_uu_ud * a0a1 * ab_1 * 2
    wva[0] += u_uu_dd * a0a1 * b0b1 * 2
    wva[0] += u_ud_ud * ab_1**2
    wva[0] += u_ud_dd * ab_1 * b0b1 * 2
    wva[0] += u_dd_dd * b0b1 * b0b1
    wva[1:4] += uu_uu_uu * a0a1 * a0a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu_ud * a0a1 * ab_1 * rho0a[1:4] * 4
    wva[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0b[1:4]
    wva[1:4] += uu_uu_dd * a0a1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_ud * ab_1**2 * rho0a[1:4] * 2
    wva[1:4] += uu_ud_ud * a0a1 * ab_1 * rho0b[1:4] * 2
    wva[1:4] += uu_ud_dd * ab_1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_dd * a0a1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += uu_dd_dd * b0b1 * b0b1 * rho0a[1:4] * 2
    wva[1:4] += ud_ud_ud * ab_1**2 * rho0b[1:4]
    wva[1:4] += ud_ud_dd * ab_1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0b[1:4]
    wva *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1a.dtype == numpy.double
    wva[0]*=.5

    wvb[0] += d_d_d * rho1b[0] * rho1b[0]
    wvb[0] += u_d_d * rho1b[0] * rho1a[0] * 2
    wvb[0] += u_u_d * rho1a[0] * rho1a[0]
    wvb[0] += d_dd * b1b1
    wvb[0] += d_ud * a1b1
    wvb[0] += d_uu * a1a1
    wvb[1:4] += u_ud * rho1a[0] * rho1a[1:4] * 2
    wvb[1:4] += u_dd * rho1a[0] * rho1b[1:4] * 4
    wvb[1:4] += d_ud * rho1b[0] * rho1a[1:4] * 2
    wvb[1:4] += d_dd * rho1b[0] * rho1b[1:4] * 4
    wvb[1:4] += dd_dd * b0b1 * rho1b[1:4] * 4
    wvb[1:4] += ud_dd * b0b1 * rho1a[1:4] * 2
    wvb[1:4] += ud_dd * ab_1 * rho1b[1:4] * 4
    wvb[1:4] += ud_ud * ab_1 * rho1a[1:4] * 2
    wvb[1:4] += uu_dd * a0a1 * rho1b[1:4] * 4
    wvb[1:4] += uu_ud * a0a1 * rho1a[1:4] * 2
    wvb[1:4] += dd_dd * b1b1 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd * a1b1 * rho0b[1:4] * 2
    wvb[1:4] += uu_dd * a1a1 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd * b1b1 * rho0a[1:4]
    wvb[1:4] += ud_ud * a1b1 * rho0a[1:4]
    wvb[1:4] += uu_ud * a1a1 * rho0a[1:4]
    wvb[0] += d_d_dd * rho1b[0] * b0b1 * 2
    wvb[0] += u_d_dd * rho1a[0] * b0b1 * 2
    wvb[0] += d_d_ud * rho1b[0] * ab_1 * 2
    wvb[0] += u_d_ud * rho1a[0] * ab_1 * 2
    wvb[0] += d_d_uu * rho1b[0] * a0a1 * 2
    wvb[0] += u_d_uu * rho1a[0] * a0a1 * 2
    wvb[1:4] += u_u_ud * rara * rho0a[1:4]
    wvb[1:4] += u_u_dd * rara * rho0b[1:4] * 2
    wvb[1:4] += u_d_ud * rarb * rho0a[1:4] * 2
    wvb[1:4] += u_d_dd * rarb * rho0b[1:4] * 4
    wvb[1:4] += d_d_ud * rbrb * rho0a[1:4]
    wvb[1:4] += d_d_dd * rbrb * rho0b[1:4] * 2
    wvb[1:4] += d_dd_dd * rho1b[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += u_dd_dd * rho1a[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += d_ud_dd * rho1b[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += u_ud_dd * rho1a[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += d_uu_dd * rho1b[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += u_uu_dd * rho1a[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0a[1:4] * 2
    wvb[0] += d_dd_dd * b0b1 * b0b1
    wvb[0] += d_ud_dd * ab_1 * b0b1 * 2
    wvb[0] += d_ud_ud * ab_1**2
    wvb[0] += d_uu_dd * b0b1 * a0a1 * 2
    wvb[0] += d_uu_ud * ab_1 * a0a1 * 2
    wvb[0] += d_uu_uu * a0a1 * a0a1
    wvb[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0a[1:4]
    wvb[1:4] += uu_uu_dd * a0a1 * a0a1 * rho0b[1:4] * 2
    wvb[1:4] += uu_ud_ud * ab_1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * b0b1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * ab_1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += uu_dd_dd * b0b1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += ud_ud_ud * ab_1**2 * rho0a[1:4]
    wvb[1:4] += ud_ud_dd * ab_1 * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += ud_ud_dd * ab_1**2 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0a[1:4]
    wvb[1:4] += ud_dd_dd * ab_1 * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += dd_dd_dd * b0b1 * b0b1 * rho0b[1:4] * 2
    wvb *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1b.dtype == numpy.double
    wvb[0]*=.5
    return wva, wvb

def _uks_mgga_wv0(rho, vxc, weight):
    rhoa, rhob = rho
    vrho, vsigma, vlapl, vtau = vxc
    ngrid = vrho.shape[0]
    wva, wvb = numpy.zeros((2,6,ngrid))
    wva[0] = vrho[:,0] * .5  # v+v.T should be applied in the caller
    wva[1:4] = rhoa[1:4] * vsigma[:,0] * 2  # sigma_uu
    wva[1:4]+= rhob[1:4] * vsigma[:,1]      # sigma_ud
    wva[5] = vtau[:,0] * .25
    wva *= weight
    wvb[0] = vrho[:,1] * .5  # v+v.T should be applied in the caller
    wvb[1:4] = rhob[1:4] * vsigma[:,2] * 2  # sigma_dd
    wvb[1:4]+= rhoa[1:4] * vsigma[:,1]      # sigma_ud
    wvb[5] = vtau[:,1] * .25
    wvb *= weight
    return wva, wvb

def _uks_mgga_wv1(rho0, rho1, vxc, fxc, weight):
    uu, ud, dd = vxc[1].T
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    ftt = fxc[4].T
    frt = fxc[6].T
    fgt = fxc[9].T
    ngrids = uu.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    ab_1 = a0b1 + b0a1

    wva, wvb = numpy.zeros((2, 6, ngrids))

    # alpha = alpha-alpha * alpha
    wva[0] += u_u * rho1a[0]
    wva[0] += u_uu * a0a1
    wva[0] += u_ud * ab_1
    wva[0] += frt[0] * rho1a[5]
    wva[1:4]+= uu * rho1a[1:4] * 2
    wva[1:4]+= u_uu * rho1a[0] * rho0a[1:4] * 2
    wva[1:4]+= u_ud * rho1a[0] * rho0b[1:4]
    wva[1:4]+= uu_uu * a0a1 * rho0a[1:4] * 2
    wva[1:4]+= uu_ud * ab_1 * rho0a[1:4] * 2
    wva[1:4]+= uu_ud * a0a1 * rho0b[1:4]
    wva[1:4]+= ud_ud * ab_1 * rho0b[1:4]
    wva[1:4]+= fgt[0] * rho1a[5] * rho0a[1:4] * 2
    wva[1:4]+= fgt[2] * rho1a[5] * rho0b[1:4]
    wva[5] += ftt[0] * rho1a[5] * .5
    wva[5] += frt[0] * rho1a[0] * .5
    wva[5] += fgt[0] * a0a1 * .5
    wva[5] += fgt[2] * ab_1 * .5
    # alpha = alpha-beta  * beta
    wva[0] += u_d * rho1b[0]
    wva[0] += u_dd * b0b1
    wva[0] += frt[1] * rho1b[5]
    wva[1:4]+= ud * rho1b[1:4]
    wva[1:4]+= d_uu * rho1b[0] * rho0a[1:4] * 2
    wva[1:4]+= d_ud * rho1b[0] * rho0b[1:4]
    wva[1:4]+= uu_dd * b0b1 * rho0a[1:4] * 2
    wva[1:4]+= ud_dd * b0b1 * rho0b[1:4]
    # uu_d * rho1b[5] * rho0a[1:4]
    wva[1:4]+= fgt[1] * rho1b[5] * rho0a[1:4] * 2
    wva[1:4]+= fgt[3] * rho1b[5] * rho0b[1:4]
    wva[5] += ftt[1] * rho1b[5] * .5
    wva[5] += frt[2] * rho1b[0] * .5
    wva[5] += fgt[4] * b0b1 * .5
    wva *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1a.dtype == numpy.double
    wva[0] *= .5
    wva[5] *= .5

    # beta = beta-alpha * alpha
    wvb[0] += u_d * rho1a[0]
    wvb[0] += d_ud * ab_1
    wvb[0] += d_uu * a0a1
    wvb[0] += frt[2] * rho1a[5]
    wvb[1:4]+= ud * rho1a[1:4]
    wvb[1:4]+= u_dd * rho1a[0] * rho0b[1:4] * 2
    wvb[1:4]+= u_ud * rho1a[0] * rho0a[1:4]
    wvb[1:4]+= ud_dd * ab_1 * rho0b[1:4] * 2
    wvb[1:4]+= ud_ud * ab_1 * rho0a[1:4]
    wvb[1:4]+= uu_dd * a0a1 * rho0b[1:4] * 2
    wvb[1:4]+= uu_ud * a0a1 * rho0a[1:4]
    # dd_u * rho1a[5] * rho0b[1:4]
    wvb[1:4]+= fgt[4] * rho1a[5] * rho0b[1:4] * 2
    wvb[1:4]+= fgt[2] * rho1a[5] * rho0a[1:4]
    wvb[5] += ftt[1] * rho1a[5] * .5
    wvb[5] += frt[1] * rho1a[0] * .5
    wvb[5] += fgt[3] * ab_1 * .5
    wvb[5] += fgt[1] * a0a1 * .5
    # beta = beta-beta  * beta
    wvb[0] += d_d * rho1b[0]
    wvb[0] += d_dd * b0b1
    wvb[0] += frt[3] * rho1b[5]
    wvb[1:4]+= dd * rho1b[1:4] * 2
    wvb[1:4]+= d_dd * rho1b[0] * rho0b[1:4] * 2
    wvb[1:4]+= d_ud * rho1b[0] * rho0a[1:4]
    wvb[1:4]+= dd_dd * b0b1 * rho0b[1:4] * 2
    wvb[1:4]+= ud_dd * b0b1 * rho0a[1:4]
    wvb[1:4]+= fgt[5] * rho1b[5] * rho0b[1:4] * 2
    wvb[1:4]+= fgt[3] * rho1b[5] * rho0a[1:4]
    wvb[5] += ftt[2] * rho1b[5] * .5
    wvb[5] += frt[3] * rho1b[0] * .5
    wvb[5] += fgt[5] * b0b1 * .5
    wvb *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1b.dtype == numpy.double
    wvb[0] *= .5
    wvb[5] *= .5
    return wva, wvb

def _uks_mgga_wv2(rho0, rho1, fxc, kxc, weight):
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    u_u_u, u_u_d, u_d_d, d_d_d = kxc[0].T
    u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, \
            d_d_ud, d_d_dd = kxc[1].T
    u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, \
            d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd = kxc[2].T
    uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, \
            ud_ud_dd, ud_dd_dd, dd_dd_dd = kxc[3].T
    fgt = fxc[9].T
    frrt = kxc[5].T
    frgt = kxc[7].T
    frtt = kxc[10].T
    fggt = kxc[12].T
    fgtt = kxc[15].T
    fttt = kxc[19].T
    ngrid = u_u.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    a1a1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1a[1:4]) * 2
    a1b1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1b[1:4]) * 2
    b1b1 = numpy.einsum('xi,xi->i', rho1b[1:4], rho1b[1:4]) * 2
    ab_1 = a0b1 + b0a1
    rara = rho1a[0] * rho1a[0]
    rarb = rho1a[0] * rho1b[0]
    rbrb = rho1b[0] * rho1b[0]
    rata = rho1a[0] * rho1a[5]
    ratb = rho1a[0] * rho1b[5]
    rbta = rho1b[0] * rho1a[5]
    rbtb = rho1b[0] * rho1b[5]
    tata = rho1a[5] * rho1a[5]
    tatb = rho1a[5] * rho1b[5]
    tbtb = rho1b[5] * rho1b[5]

    wva, wvb = numpy.zeros((2, 6, ngrid))

    wva[0] += u_u_u * rara
    wva[0] += u_u_d * rarb * 2
    wva[0] += u_d_d * rbrb
    wva[0] += u_uu * a1a1
    wva[0] += u_ud * a1b1
    wva[0] += u_dd * b1b1
    wva[0] += u_u_uu * rho1a[0] * a0a1 * 2
    wva[0] += u_u_ud * rho1a[0] * ab_1 * 2
    wva[0] += u_u_dd * rho1a[0] * b0b1 * 2
    wva[0] += u_d_uu * rho1b[0] * a0a1 * 2
    wva[0] += u_d_ud * rho1b[0] * ab_1 * 2
    wva[0] += u_d_dd * rho1b[0] * b0b1 * 2
    wva[0] += u_uu_uu * a0a1 * a0a1
    wva[0] += u_uu_ud * a0a1 * ab_1 * 2
    wva[0] += u_uu_dd * a0a1 * b0b1 * 2
    wva[0] += u_ud_ud * ab_1**2
    wva[0] += u_ud_dd * b0b1 * ab_1 * 2
    wva[0] += u_dd_dd * b0b1 * b0b1
    wva[0] += frgt[0] * rho1a[5] * a0a1 * 2                     # u_uu_u
    wva[0] += frgt[1] * rho1b[5] * a0a1 * 2                     # u_uu_d
    wva[0] += frgt[2] * rho1a[5] * ab_1 * 2                     # u_ud_u
    wva[0] += frgt[3] * rho1b[5] * ab_1 * 2                     # u_ud_d
    wva[0] += frgt[4] * rho1a[5] * b0b1 * 2                     # u_dd_u
    wva[0] += frgt[5] * rho1b[5] * b0b1 * 2                     # u_dd_d
    wva[0] += frrt[0] * rata * 2                                # u_u_u
    wva[0] += frrt[1] * ratb * 2                                # u_u_d
    wva[0] += frrt[2] * rbta * 2                                # u_d_u
    wva[0] += frrt[3] * rbtb * 2                                # u_d_d
    wva[0] += frtt[0] * tata                                    # u_u_u
    wva[0] += frtt[1] * tatb * 2                                # u_u_d
    wva[0] += frtt[2] * tbtb                                    # u_d_d
    wva[1:4] += u_uu * rho1a[0] * rho1a[1:4] * 4
    wva[1:4] += u_ud * rho1a[0] * rho1b[1:4] * 2
    wva[1:4] += d_uu * rho1b[0] * rho1a[1:4] * 4
    wva[1:4] += d_ud * rho1b[0] * rho1b[1:4] * 2
    wva[1:4] += uu_uu * a1a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu * a0a1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * ab_1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * a1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_ud * a1a1 * rho0b[1:4]
    wva[1:4] += uu_ud * a0a1 * rho1b[1:4] * 2
    wva[1:4] += uu_dd * b1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_dd * b0b1 * rho1a[1:4] * 4
    wva[1:4] += ud_ud * ab_1 * rho1b[1:4] * 2
    wva[1:4] += ud_ud * a1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b0b1 * rho1b[1:4] * 2
    wva[1:4] += u_u_uu * rara * rho0a[1:4] * 2
    wva[1:4] += u_u_ud * rara * rho0b[1:4]
    wva[1:4] += u_d_uu * rarb * rho0a[1:4] * 4
    wva[1:4] += u_d_ud * rarb * rho0b[1:4] * 2
    wva[1:4] += d_d_uu * rbrb * rho0a[1:4] * 2
    wva[1:4] += d_d_ud * rbrb * rho0b[1:4]
    wva[1:4] += u_uu_uu * rho1a[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += u_uu_dd * rho1a[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0b[1:4] * 2
    wva[1:4] += d_uu_uu * rho1b[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += d_uu_dd * rho1b[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0b[1:4] * 2
    wva[1:4] += uu_uu_uu * a0a1 * a0a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu_ud * a0a1 * ab_1 * rho0a[1:4] * 4
    wva[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0b[1:4]
    wva[1:4] += uu_uu_dd * a0a1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_ud * ab_1**2 * rho0a[1:4] * 2
    wva[1:4] += uu_ud_ud * a0a1 * ab_1 * rho0b[1:4] * 2
    wva[1:4] += uu_ud_dd * ab_1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_dd * a0a1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += uu_dd_dd * b0b1 * b0b1 * rho0a[1:4] * 2
    wva[1:4] += ud_ud_ud * ab_1**2 * rho0b[1:4]
    wva[1:4] += ud_ud_dd * ab_1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0b[1:4]
    wva[1:4] += frgt[0] * rata * rho0a[1:4] * 4                 # u_uu_u
    wva[1:4] += frgt[1] * ratb * rho0a[1:4] * 4                 # u_uu_d
    wva[1:4] += frgt[2] * rata * rho0b[1:4] * 2                 # u_ud_u
    wva[1:4] += frgt[3] * ratb * rho0b[1:4] * 2                 # u_ud_d
    wva[1:4] += frgt[6] * rbta * rho0a[1:4] * 4                 # d_uu_u
    wva[1:4] += frgt[7] * rbtb * rho0a[1:4] * 4                 # d_uu_d
    wva[1:4] += frgt[8] * rbta * rho0b[1:4] * 2                 # d_ud_u
    wva[1:4] += frgt[9] * rbtb * rho0b[1:4] * 2                 # d_ud_d
    wva[1:4] += fgt[0] * rho1a[5] * rho1a[1:4] * 4              # uu_u
    wva[1:4] += fgt[1] * rho1b[5] * rho1a[1:4] * 4              # uu_d
    wva[1:4] += fgt[2] * rho1a[5] * rho1b[1:4] * 2              # ud_u
    wva[1:4] += fgt[3] * rho1b[5] * rho1b[1:4] * 2              # ud_d
    wva[1:4] += fggt[0] * rho1a[5] * a0a1 * rho0a[1:4] * 4      # uu_uu_u
    wva[1:4] += fggt[1] * rho1b[5] * a0a1 * rho0a[1:4] * 4      # uu_uu_d
    wva[1:4] += fggt[2] * rho1a[5] * a0a1 * rho0b[1:4] * 2      # uu_ud_u
    wva[1:4] += fggt[2] * rho1a[5] * ab_1 * rho0a[1:4] * 4      # uu_ud_u
    wva[1:4] += fggt[3] * rho1b[5] * a0a1 * rho0b[1:4] * 2      # uu_ud_d
    wva[1:4] += fggt[3] * rho1b[5] * ab_1 * rho0a[1:4] * 4      # uu_ud_d
    wva[1:4] += fggt[4] * rho1a[5] * b0b1 * rho0a[1:4] * 4      # uu_dd_u
    wva[1:4] += fggt[5] * rho1b[5] * b0b1 * rho0a[1:4] * 4      # uu_dd_d
    wva[1:4] += fggt[6] * rho1a[5] * ab_1 * rho0b[1:4] * 2      # ud_ud_u
    wva[1:4] += fggt[7] * rho1b[5] * ab_1 * rho0b[1:4] * 2      # ud_ud_d
    wva[1:4] += fggt[8] * rho1a[5] * b0b1 * rho0b[1:4] * 2      # ud_dd_u
    wva[1:4] += fggt[9] * rho1b[5] * b0b1 * rho0b[1:4] * 2      # ud_dd_d
    wva[1:4] += fgtt[0] * tata * rho0a[1:4] * 2                 # uu_u_u
    wva[1:4] += fgtt[1] * tatb * rho0a[1:4] * 4                 # uu_u_d
    wva[1:4] += fgtt[2] * tbtb * rho0a[1:4] * 2                 # uu_d_d
    wva[1:4] += fgtt[3] * tata * rho0b[1:4]                     # ud_u_u
    wva[1:4] += fgtt[4] * tatb * rho0b[1:4] * 2                 # ud_u_d
    wva[1:4] += fgtt[5] * tbtb * rho0b[1:4]                     # ud_d_d
    wva[5] += frgt[0 ] * rho1a[0] * a0a1                        # u_uu_u
    wva[5] += frgt[2 ] * rho1a[0] * ab_1                        # u_ud_u
    wva[5] += frgt[4 ] * rho1a[0] * b0b1                        # u_dd_u
    wva[5] += frgt[6 ] * rho1b[0] * a0a1                        # d_uu_u
    wva[5] += frgt[8 ] * rho1b[0] * ab_1                        # d_ud_u
    wva[5] += frgt[10] * rho1b[0] * b0b1                        # d_dd_u
    wva[5] += fggt[0 ] * a0a1 * a0a1 * .5                       # uu_uu_u
    wva[5] += fggt[2 ] * a0a1 * ab_1                            # uu_ud_u
    wva[5] += fggt[4 ] * a0a1 * b0b1                            # uu_dd_u
    wva[5] += fggt[6 ] * ab_1**2 * .5                           # ud_ud_u
    wva[5] += fggt[8 ] * ab_1 * b0b1                            # ud_dd_u
    wva[5] += fggt[10] * b0b1 * b0b1 * .5                       # dd_dd_u
    wva[5] += fgtt[0] * a0a1 * rho1a[5]                         # uu_u_u
    wva[5] += fgtt[1] * a0a1 * rho1b[5]                         # uu_u_d
    wva[5] += fgtt[3] * ab_1 * rho1a[5]                         # ud_u_u
    wva[5] += fgtt[4] * ab_1 * rho1b[5]                         # ud_u_d
    wva[5] += fgtt[6] * b0b1 * rho1a[5]                         # dd_u_u
    wva[5] += fgtt[7] * b0b1 * rho1b[5]                         # dd_u_d
    wva[5] += fgt[0] * a1a1 * .5                                # uu_u
    wva[5] += fgt[2] * a1b1 * .5                                # ud_u
    wva[5] += fgt[4] * b1b1 * .5                                # dd_u
    wva[5] += frrt[0] * rara * .5                               # u_u_u
    wva[5] += frrt[2] * rarb                                    # u_d_u
    wva[5] += frrt[4] * rbrb * .5                               # d_d_u
    wva[5] += frtt[0] * rata                                    # u_u_u
    wva[5] += frtt[1] * ratb                                    # u_u_d
    wva[5] += frtt[3] * rbta                                    # d_u_u
    wva[5] += frtt[4] * rbtb                                    # d_u_d
    wva[5] += fttt[0] * tata * .5                               # u_u_u
    wva[5] += fttt[1] * tatb                                    # u_u_d
    wva[5] += fttt[2] * tbtb * .5                               # u_d_d
    wva *= weight
    wva[0] *= .5
    wva[5] *= .5

    wvb[0] += u_u_d * rara
    wvb[0] += u_d_d * rarb * 2
    wvb[0] += d_d_d * rbrb
    wvb[0] += d_uu * a1a1
    wvb[0] += d_ud * a1b1
    wvb[0] += d_dd * b1b1
    wvb[0] += u_d_uu * rho1a[0] * a0a1 * 2
    wvb[0] += u_d_ud * rho1a[0] * ab_1 * 2
    wvb[0] += u_d_dd * rho1a[0] * b0b1 * 2
    wvb[0] += d_d_uu * rho1b[0] * a0a1 * 2
    wvb[0] += d_d_ud * rho1b[0] * ab_1 * 2
    wvb[0] += d_d_dd * rho1b[0] * b0b1 * 2
    wvb[0] += d_uu_uu * a0a1 * a0a1
    wvb[0] += d_uu_ud * a0a1 * ab_1 * 2
    wvb[0] += d_uu_dd * b0b1 * a0a1 * 2
    wvb[0] += d_ud_ud * ab_1**2
    wvb[0] += d_ud_dd * b0b1 * ab_1 * 2
    wvb[0] += d_dd_dd * b0b1 * b0b1
    wvb[0] += frgt[6 ] * rho1a[5] * a0a1 * 2                    # d_uu_u
    wvb[0] += frgt[7 ] * rho1b[5] * a0a1 * 2                    # d_uu_d
    wvb[0] += frgt[8 ] * rho1a[5] * ab_1 * 2                    # d_ud_u
    wvb[0] += frgt[9 ] * rho1b[5] * ab_1 * 2                    # d_ud_d
    wvb[0] += frgt[10] * rho1a[5] * b0b1 * 2                    # d_dd_u
    wvb[0] += frgt[11] * rho1b[5] * b0b1 * 2                    # d_dd_d
    wvb[0] += frrt[2] * rata * 2                                # u_d_u
    wvb[0] += frrt[3] * ratb * 2                                # u_d_d
    wvb[0] += frrt[4] * rbta * 2                                # d_d_u
    wvb[0] += frrt[5] * rbtb * 2                                # d_d_d
    wvb[0] += frtt[3] * tata                                    # d_u_u
    wvb[0] += frtt[4] * tatb * 2                                # d_u_d
    wvb[0] += frtt[5] * tbtb                                    # d_d_d
    wvb[1:4] += u_ud * rho1a[0] * rho1a[1:4] * 2
    wvb[1:4] += u_dd * rho1a[0] * rho1b[1:4] * 4
    wvb[1:4] += d_ud * rho1b[0] * rho1a[1:4] * 2
    wvb[1:4] += d_dd * rho1b[0] * rho1b[1:4] * 4
    wvb[1:4] += uu_ud * a1a1 * rho0a[1:4]
    wvb[1:4] += uu_ud * a0a1 * rho1a[1:4] * 2
    wvb[1:4] += uu_dd * a1a1 * rho0b[1:4] * 2
    wvb[1:4] += uu_dd * a0a1 * rho1b[1:4] * 4
    wvb[1:4] += ud_ud * a1b1 * rho0a[1:4]
    wvb[1:4] += ud_ud * ab_1 * rho1a[1:4] * 2
    wvb[1:4] += ud_dd * b1b1 * rho0a[1:4]
    wvb[1:4] += ud_dd * a1b1 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd * b0b1 * rho1a[1:4] * 2
    wvb[1:4] += ud_dd * ab_1 * rho1b[1:4] * 4
    wvb[1:4] += dd_dd * b1b1 * rho0b[1:4] * 2
    wvb[1:4] += dd_dd * b0b1 * rho1b[1:4] * 4
    wvb[1:4] += u_u_ud * rara * rho0a[1:4]
    wvb[1:4] += u_u_dd * rara * rho0b[1:4] * 2
    wvb[1:4] += u_d_ud * rarb * rho0a[1:4] * 2
    wvb[1:4] += u_d_dd * rarb * rho0b[1:4] * 4
    wvb[1:4] += d_d_ud * rbrb * rho0a[1:4]
    wvb[1:4] += d_d_dd * rbrb * rho0b[1:4] * 2
    wvb[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += u_uu_dd * rho1a[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_dd * rho1a[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += u_dd_dd * rho1a[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += d_uu_dd * rho1b[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += d_ud_dd * rho1b[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += d_dd_dd * rho1b[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0a[1:4]
    wvb[1:4] += uu_uu_dd * a0a1 * a0a1 * rho0b[1:4] * 2
    wvb[1:4] += uu_ud_ud * ab_1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * b0b1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * ab_1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += uu_dd_dd * b0b1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += ud_ud_ud * ab_1**2 * rho0a[1:4]
    wvb[1:4] += ud_ud_dd * b0b1 * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += ud_ud_dd * ab_1**2 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0a[1:4]
    wvb[1:4] += ud_dd_dd * b0b1 * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += dd_dd_dd * b0b1 * b0b1 * rho0b[1:4] * 2
    wvb[1:4] += frgt[2 ] * rata * rho0a[1:4] * 2                # u_ud_u
    wvb[1:4] += frgt[3 ] * ratb * rho0a[1:4] * 2                # u_ud_d
    wvb[1:4] += frgt[4 ] * rata * rho0b[1:4] * 4                # u_dd_u
    wvb[1:4] += frgt[5 ] * ratb * rho0b[1:4] * 4                # u_dd_d
    wvb[1:4] += frgt[8 ] * rbta * rho0a[1:4] * 2                # d_ud_u
    wvb[1:4] += frgt[9 ] * rbtb * rho0a[1:4] * 2                # d_ud_d
    wvb[1:4] += frgt[10] * rbta * rho0b[1:4] * 4                # d_dd_u
    wvb[1:4] += frgt[11] * rbtb * rho0b[1:4] * 4                # d_dd_d
    wvb[1:4] += fgt[2] * rho1a[5] * rho1a[1:4] * 2              # ud_u
    wvb[1:4] += fgt[3] * rho1b[5] * rho1a[1:4] * 2              # ud_d
    wvb[1:4] += fgt[4] * rho1a[5] * rho1b[1:4] * 4              # dd_u
    wvb[1:4] += fgt[5] * rho1b[5] * rho1b[1:4] * 4              # dd_d
    wvb[1:4] += fggt[2 ] * rho1a[5] * a0a1 * rho0a[1:4] * 2     # uu_ud_u
    wvb[1:4] += fggt[3 ] * rho1b[5] * a0a1 * rho0a[1:4] * 2     # uu_ud_d
    wvb[1:4] += fggt[4 ] * rho1a[5] * a0a1 * rho0b[1:4] * 4     # uu_dd_u
    wvb[1:4] += fggt[5 ] * rho1b[5] * a0a1 * rho0b[1:4] * 4     # uu_dd_d
    wvb[1:4] += fggt[6 ] * rho1a[5] * ab_1 * rho0a[1:4] * 2     # ud_ud_u
    wvb[1:4] += fggt[7 ] * rho1b[5] * ab_1 * rho0a[1:4] * 2     # ud_ud_d
    wvb[1:4] += fggt[8 ] * rho1a[5] * ab_1 * rho0b[1:4] * 4     # ud_dd_u
    wvb[1:4] += fggt[8 ] * rho1a[5] * b0b1 * rho0a[1:4] * 2     # ud_dd_u
    wvb[1:4] += fggt[9 ] * rho1b[5] * ab_1 * rho0b[1:4] * 4     # ud_dd_d
    wvb[1:4] += fggt[9 ] * rho1b[5] * b0b1 * rho0a[1:4] * 2     # ud_dd_d
    wvb[1:4] += fggt[10] * rho1a[5] * b0b1 * rho0b[1:4] * 4     # dd_dd_u
    wvb[1:4] += fggt[11] * rho1b[5] * b0b1 * rho0b[1:4] * 4     # dd_dd_d
    wvb[1:4] += fgtt[3] * tata * rho0a[1:4]                     # ud_u_u
    wvb[1:4] += fgtt[4] * tatb * rho0a[1:4] * 2                 # ud_u_d
    wvb[1:4] += fgtt[5] * tbtb * rho0a[1:4]                     # ud_d_d
    wvb[1:4] += fgtt[6] * tata * rho0b[1:4] * 2                 # dd_u_u
    wvb[1:4] += fgtt[7] * tatb * rho0b[1:4] * 4                 # dd_u_d
    wvb[1:4] += fgtt[8] * tbtb * rho0b[1:4] * 2                 # dd_d_d
    wvb[5] += frgt[1 ] * rho1a[0] * a0a1                        # u_uu_d
    wvb[5] += frgt[3 ] * rho1a[0] * ab_1                        # u_ud_d
    wvb[5] += frgt[5 ] * rho1a[0] * b0b1                        # u_dd_d
    wvb[5] += frgt[7 ] * rho1b[0] * a0a1                        # d_uu_d
    wvb[5] += frgt[9 ] * rho1b[0] * ab_1                        # d_ud_d
    wvb[5] += frgt[11] * rho1b[0] * b0b1                        # d_dd_d
    wvb[5] += fggt[1 ] * a0a1 * a0a1 * .5                       # uu_uu_d
    wvb[5] += fggt[3 ] * ab_1 * a0a1                            # uu_ud_d
    wvb[5] += fggt[5 ] * b0b1 * a0a1                            # uu_dd_d
    wvb[5] += fggt[7 ] * ab_1**2 * .5                           # ud_ud_d
    wvb[5] += fggt[9 ] * b0b1 * ab_1                            # ud_dd_d
    wvb[5] += fggt[11] * b0b1 * b0b1 * .5                       # dd_dd_d
    wvb[5] += fgt[1] * a1a1 * .5                                # uu_d
    wvb[5] += fgt[3] * a1b1 * .5                                # ud_d
    wvb[5] += fgt[5] * b1b1 * .5                                # dd_d
    wvb[5] += fgtt[1] * a0a1 * rho1a[5]                         # uu_u_d
    wvb[5] += fgtt[2] * a0a1 * rho1b[5]                         # uu_d_d
    wvb[5] += fgtt[4] * ab_1 * rho1a[5]                         # ud_u_d
    wvb[5] += fgtt[5] * ab_1 * rho1b[5]                         # ud_d_d
    wvb[5] += fgtt[7] * b0b1 * rho1a[5]                         # dd_u_d
    wvb[5] += fgtt[8] * b0b1 * rho1b[5]                         # dd_d_d
    wvb[5] += frrt[1] * rara * .5                               # u_u_d
    wvb[5] += frrt[3] * rarb                                    # u_d_d
    wvb[5] += frrt[5] * rbrb * .5                               # d_d_d
    wvb[5] += frtt[1] * rata                                    # u_u_d
    wvb[5] += frtt[2] * ratb                                    # u_d_d
    wvb[5] += frtt[4] * rbta                                    # d_u_d
    wvb[5] += frtt[5] * rbtb                                    # d_d_d
    wvb[5] += fttt[1] * tata * .5                               # u_u_d
    wvb[5] += fttt[2] * tatb                                    # u_d_d
    wvb[5] += fttt[3] * tbtb * .5                               # d_d_d
    wvb *= weight
    wvb[0] *= .5
    wvb[5] *= .5

    return wva, wvb

def _empty_aligned(shape, alignment=8):
    if alignment <= 1:
        return numpy.empty(shape)

    size = numpy.prod(shape)
    buf = numpy.empty(size + alignment - 1)
    align8 = alignment * 8
    offset = buf.ctypes.data % align8
    if offset != 0:
        offset = (align8 - offset) // 8
    return numpy.ndarray(size, buffer=buf[offset:offset+size]).reshape(shape)


def nr_fxc(mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
           rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    r'''Contract XC kernel matrix with given density matrices

    ... math::

            a_{pq} = f_{pq,rs} * x_{rs}

    '''
    ni = NumInt()
    return ni.nr_fxc(mol, grids, xc_code, dm0, dms, spin, relativity,
                     hermi, rho0, vxc, fxc, max_memory, verbose)


def cache_xc_kernel(ni, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                    max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0
    with_lapl = MGGA_DENSITY_LAPL

    if mo_coeff[0].ndim == 1:  # RKS
        nao = mo_coeff.shape[0]
        rho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho.append(ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, xctype, with_lapl))
        rho = numpy.hstack(rho)
        if spin == 1:  # RKS with nr_rks_fxc_st
            rho *= .5
            rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
    else:  # UKS
        assert mo_coeff[0].ndim == 2
        assert spin == 1
        nao = mo_coeff[0].shape[0]
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rhoa.append(ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl))
            rhob.append(ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
    return rho, vxc, fxc

def cache_xc_kernel1(ni, mol, grids, xc_code, dm, spin=0, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc. Note dm the zeroth order density matrix must be a
    hermitian matrix.
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0

    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    if dm[0].ndim == 1:  # RKS
        rho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho.append(make_rho(0, ao, mask, xctype))
        rho = numpy.hstack(rho)
        if spin == 1:  # RKS with nr_rks_fxc_st
            rho *= .5
            rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
    else:  # UKS
        assert dm[0].ndim == 2
        assert spin == 1
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa.append(make_rho(0, ao, mask, xctype))
            rhob.append(make_rho(1, ao, mask, xctype))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
    return rho, vxc, fxc

def get_rho(ni, mol, dm, grids, max_memory=2000):
    '''Density in real space
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)
    assert nset == 1
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao, mask, 'LDA')
    return rho


class LibXCMixin:
    libxc = libxc

    omega = None  # RSH parameter

####################
# Overwrite following functions to use custom XC functional

    def hybrid_coeff(self, xc_code, spin=0):
        return self.libxc.hybrid_coeff(xc_code, spin)

    def nlc_coeff(self, xc_code):
        return self.libxc.nlc_coeff(xc_code)

    def rsh_coeff(self, xc_code):
        return self.libxc.rsh_coeff(xc_code)

    @lib.with_doc(libxc.eval_xc.__doc__)
    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        if omega is None: omega = self.omega
        return self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv,
                                  omega, verbose)

    def eval_xc1(self, xc_code, rho, spin=0, deriv=1, omega=None):
        if omega is None: omega = self.omega
        return self.libxc.eval_xc1(xc_code, rho, spin, deriv, omega)

    def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None, xctype=None,
                    verbose=None, spin=None):
        r'''Returns the derivative tensor against the density parameters

        [density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a]

        or spin-polarized density parameters

        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].

        It differs from the eval_xc method in the derivatives of non-local part.
        The eval_xc method returns the XC functional derivatives to sigma
        (|\nabla \rho|^2)

        Args:
            rho: 2-dimensional or 3-dimensional array
                Total density or (spin-up, spin-down) densities (and their
                derivatives if GGA or MGGA functionals) on grids

        Kwargs:
            deriv: int
                derivative orders
            omega: float
                define the exponent in the attenuated Coulomb for RSH functional
            spin : int
                spin polarized if spin > 0
        '''
        if omega is None: omega = self.omega
        if xctype is None: xctype = self._xc_type(xc_code)

        rho = numpy.asarray(rho, order='C', dtype=numpy.double)
        if xctype == 'MGGA' and rho.shape[-2] == 6:
            rho = numpy.asarray(rho[...,[0,1,2,3,5],:], order='C')

        if spin is None:
            spin_polarized = rho.ndim >= 2 and rho.shape[0] == 2
            if spin_polarized:
                spin = 1
            else:
                spin = 0

        out = self.eval_xc1(xc_code, rho, spin, deriv, omega)
        evfk = [out[0]]
        for order in range(1, deriv+1):
            evfk.append(xc_deriv.transform_xc(rho, out, xctype, spin, order))
        if deriv < 3:
            # Returns at least [e, v, f, k] terms
            evfk.extend([None] * (3 - deriv))
        return evfk

    def _xc_type(self, xc_code):
        return self.libxc.xc_type(xc_code)

    def rsh_and_hybrid_coeff(self, xc_code, spin=0):
        '''Range-separated parameter and HF exchange components: omega, alpha, beta

        Exc_RSH = c_SR * SR_HFX + c_LR * LR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
                = alpha * HFX + beta * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
                = alpha * LR_HFX + hyb * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec

        SR_HFX = < pi | (1-erf(-omega r_{12}))/r_{12} | iq >
        LR_HFX = < pi | erf(-omega r_{12})/r_{12} | iq >
        alpha = c_LR
        beta = c_SR - c_LR
        '''
        omega, alpha, beta = self.rsh_coeff(xc_code)
        if self.omega is not None:
            if omega == 0 and self.omega != 0:
                raise RuntimeError(f'Not support assigning omega={self.omega}. '
                                   f'{xc_code} is not a RSH functional')
            omega = self.omega

        if omega != 0:
            hyb = alpha + beta
        else:
            hyb = self.hybrid_coeff(xc_code, spin)
        return omega, alpha, hyb

# Export the symbol _NumIntMixin for backward compatibility.
# _NumIntMixin should be dropped in the future.
_NumIntMixin = LibXCMixin


class NumInt(lib.StreamObject, LibXCMixin):
    '''Numerical integration methods for non-relativistic RKS and UKS

    Input Attributes:
        omega :
            The Coulomb attenuation parameter for range-separated functionals.
            If specified, this value will replace the default setting in libxc
            when evaluating the libxc RSH functional.
    '''

    cutoff = CUTOFF * 1e2  # cutoff for small AO product

    @lib.with_doc(nr_vxc.__doc__)
    def nr_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        if spin == 0:
            return self.nr_rks(mol, grids, xc_code, dms, relativity, hermi,
                               max_memory, verbose)
        else:
            return self.nr_uks(mol, grids, xc_code, dms, relativity, hermi,
                               max_memory, verbose)
    get_vxc = nr_vxc

    @lib.with_doc(nr_fxc.__doc__)
    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        if spin == 0:
            return self.nr_rks_fxc(mol, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, max_memory, verbose)
        else:
            return self.nr_uks_fxc(mol, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, max_memory, verbose)
    get_fxc = nr_fxc

    nr_rks = nr_rks
    nr_uks = nr_uks
    nr_nlc_vxc = nr_nlc_vxc
    nr_sap = nr_sap_vxc = nr_sap_vxc
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    cache_xc_kernel  = cache_xc_kernel
    cache_xc_kernel1 = cache_xc_kernel1

    make_mask = staticmethod(make_mask)
    eval_ao = staticmethod(eval_ao)
    eval_rho  = staticmethod(eval_rho)
    eval_rho1 = lib.module_method(eval_rho1, absences=['cutoff'])
    eval_rho2 = staticmethod(eval_rho2)
    get_rho = get_rho

    def block_loop(self, mol, grids, nao=None, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = mol.nao
        ngrids = grids.coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/((comp+1)*nao*8*BLKSIZE))
            blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 1200)) * BLKSIZE
        assert blksize % BLKSIZE == 0

        if non0tab is None and mol is grids.mol:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
        screen_index = non0tab

        # the xxx_sparse() functions require ngrids 8-byte aligned
        allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > SWITCH_SIZE

        if buf is None:
            buf = _empty_aligned(comp * blksize * nao)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            mask = screen_index[ip0//BLKSIZE:]
            # TODO: pass grids.cutoff to eval_ao
            ao = self.eval_ao(mol, coords, deriv=deriv, non0tab=mask,
                              cutoff=grids.cutoff, out=buf)
            if not allow_sparse and not _sparse_enough(mask):
                # Unset mask for dense AO tensor. It determines which eval_rho
                # to be called in make_rho
                mask = None
            yield ao, mask, weight, coords

    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        if getattr(dms, 'mo_coeff', None) is not None:
            #TODO: test whether dm.mo_coeff matching dm
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
        else:
            mo_coeff = mo_occ = None

        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            dms = dms[numpy.newaxis]

        if hermi != 1 and dms[0].dtype == numpy.double:
            # (D + D.T)/2 because eval_rho computes 2*(|\nabla i> D_ij <j|) instead of
            # |\nabla i> D_ij <j| + |i> D_ij <\nabla j| for efficiency when dm is real
            dms = lib.hermi_sum(numpy.asarray(dms, order='C'), axes=(0,2,1)) * .5
            hermi = 1

        nao = dms[0].shape[0]
        ndms = len(dms)

        if grids is not None:
            ovlp_cond = mol.get_overlap_cond()
            if dms[0].dtype == numpy.double:
                dm_cond = [mol.condense_to_shell(dm, 'absmax') for dm in dms]
                dm_cond = numpy.max(dm_cond, axis=0)
                pair_mask = numpy.exp(-ovlp_cond) * dm_cond > self.cutoff
            else:
                pair_mask = ovlp_cond < -numpy.log(self.cutoff)
            pair_mask = numpy.asarray(pair_mask, dtype=numpy.uint8)

        if (mo_occ is not None) and (grids is not None):
            # eval_rho2 is more efficient unless we have a very large system
            # for which the pair_mask is significantly sparser than the
            # ratio of occupied to total molecular orbitals. So we use this ratio
            # to switch between eval_rho1 and eval_rho2.
            mo_ao_sparsity = max(0.5 * numpy.sum(mo_occ) / nao, 1e-8)
            wts = mol.ao_loc_nr()
            wts = (wts[1:] - wts[:-1]) / wts[-1]
            rho1_rho2_ratio = numpy.dot(wts, pair_mask).dot(wts) / mo_ao_sparsity
        else:
            rho1_rho2_ratio = 0.0

        def make_rho(idm, ao, sindex, xctype):
            has_screening = sindex is not None and grids is not None
            has_mo = mo_coeff is not None
            if xctype == "GGA":
                # GGA has to do more contractions using rho2 compared to rho1,
                # so the threshold for switching to rho1 is less strict.
                is_sparse = rho1_rho2_ratio < 4
            else:
                is_sparse = rho1_rho2_ratio < 1
            if has_screening and (not has_mo or is_sparse):
                return self.eval_rho1(mol, ao, dms[idm], sindex, xctype, hermi,
                                      with_lapl, cutoff=self.cutoff,
                                      ao_cutoff=grids.cutoff, pair_mask=pair_mask)
            elif has_mo:
                return self.eval_rho2(mol, ao, mo_coeff[idm], mo_occ[idm],
                                      sindex, xctype, with_lapl)
            else:
                return self.eval_rho(mol, ao, dms[idm], sindex, xctype, hermi,
                                     with_lapl)
        return make_rho, ndms, nao

    def to_gpu(self):
        try:
            from gpu4pyscf.dft import numint # type: ignore
            return numint.NumInt()
        except ImportError:
            raise ImportError('Cannot find GPU4PySCF')
_NumInt = NumInt


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.M(atom=[
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis='6311g**')
    mf = dft.RKS(mol)
    mf.grids.atom_grid = {"H": (30, 194), "O": (30, 194),}
    mf.grids.prune = None
    mf.grids.build()
    dm = mf.get_init_guess(key='minao')

    numpy.random.seed(1)
    dm1 = numpy.random.random((dm.shape))
    dm1 = lib.hermi_triu(dm1)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm1, spin=0)
    print(res[1] - -37.084047825971282)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm1,dm1), spin=1)
    print(res[1] - -92.436362308687094)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm, spin=0)
    print(res[1] - -8.6313329288394947)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm,dm), spin=1)
    print(res[1] - -21.520301399504582)
