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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import warnings
import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
try:
    from pyscf.dft import libxc
except (ImportError, OSError):
    try:
        from pyscf.dft import xcfun
        libxc = xcfun
    except (ImportError, OSError):
        import warnings
        warnings.warn('XC functional libraries (libxc or XCfun) are not available.')
        from pyscf.dft import xc
        libxc = xc

from pyscf.dft.gen_grid import make_mask, BLKSIZE
from pyscf import __config__

libdft = lib.load_library('libdft')
OCCDROP = getattr(__config__, 'dft_numint_OCCDROP', 1e-12)
# The system size above which to consider the sparsity of the density matrix.
# If the number of AOs in the system is less than this value, all tensors are
# treated as dense quantities and contracted by dgemm directly.
SWITCH_SIZE = getattr(__config__, 'dft_numint_SWITCH_SIZE', 800)

def eval_ao(mol, coords, deriv=0, shls_slice=None,
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
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in mol will be evaluated.
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
    return mol.eval_gto(feval, coords, comp, shls_slice, non0tab, out=out)

#TODO: \nabla^2 rho and tau = 1/2 (\nabla f)^2
def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
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
        hermi : bool
            dm is hermitian or not
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
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    if not hermi:
        # (D + D.T)/2 because eval_rho computes 2*(|\nabla i> D_ij <j|) instead of
        # |\nabla i> D_ij <j| + |i> D_ij <\nabla j| for efficiency
        dm = (dm + dm.conj().T) * .5

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
        rho[0] = _contract_rho(c0, ao[0])
        for i in range(1, 4):
            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            rho[i] = _contract_rho(c0, ao[i])
            rho[i] *= 2 # *2 for +c.c. in the next two lines
            #c1 = _dot_ao_dm(mol, ao[i], dm, non0tab, shls_slice, ao_loc)
            #rho[i] += numpy.einsum('pi,pi->p', c1, ao[0])
    else: # meta-GGA
        # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
        rho = numpy.empty((6,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        rho[0] = _contract_rho(ao[0], c0)
        rho[5] = 0
        for i in range(1, 4):
            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i]) * 2 # *2 for +c.c.
            rho[i] = _contract_rho(c0, ao[i]) * 2
            c1 = _dot_ao_dm(mol, ao[i], dm.T, non0tab, shls_slice, ao_loc)
            #:rho[5] += numpy.einsum('pi,pi->p', c1, ao[i])
            rho[5] += _contract_rho(c1, ao[i])
        XX, YY, ZZ = 4, 7, 9
        ao2 = ao[XX] + ao[YY] + ao[ZZ]
        #:rho[4] = numpy.einsum('pi,pi->p', c0, ao2)
        rho[4] = _contract_rho(c0, ao2)
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
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    pos = mo_occ > OCCDROP
    if pos.sum() > 0:
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
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            c0 = _dot_ao_dm(mol, ao[0], cpos, non0tab, shls_slice, ao_loc)
            #:rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            rho[0] = _contract_rho(c0, c0)
            rho[5] = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, non0tab, shls_slice, ao_loc)
                #:rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                #:rho[5] += numpy.einsum('pi,pi->p', c1, c1)
                rho[i] = _contract_rho(c0, c1) * 2
                rho[5] += _contract_rho(c1, c1)
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = _dot_ao_dm(mol, ao2, cpos, non0tab, shls_slice, ao_loc)
            #:rho[4] = numpy.einsum('pi,pi->p', c0, c1)
            rho[4] = _contract_rho(c0, c1)
            rho[4] += rho[5]
            rho[4] *= 2

            rho[5] *= .5
    else:
        if xctype == 'LDA' or xctype == 'HF':
            rho = numpy.zeros(ngrids)
        elif xctype in ('GGA', 'NLC'):
            rho = numpy.zeros((4,ngrids))
        else:
            rho = numpy.zeros((6,ngrids))

    neg = mo_occ < -OCCDROP
    if neg.sum() > 0:
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
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = _dot_ao_dm(mol, ao2, cneg, non0tab, shls_slice, ao_loc)
            #:rho[4] -= numpy.einsum('pi,pi->p', c0, c1) * 2
            rho[4] -= _contract_rho(c0, c1) * 2
            rho[4] -= rho5 * 2

            rho[5] -= rho5 * .5
    return rho

def _vv10nlc(rho,coords,vvrho,vvweight,vvcoords,nlc_pars):
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

    #threshed output
    excthresh=numpy.zeros(R.size)
    vxcthresh=numpy.zeros([2,R.size])

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

    for i in range(R.size):
        DX=vvcoords[:,0]-coords[i,0]
        DY=vvcoords[:,1]-coords[i,1]
        DZ=vvcoords[:,2]-coords[i,2]
        R2=DX*DX+DY*DY+DZ*DZ
        gp=R2*W0p+Kp
        g=R2*W0[i]+K[i]
        gt=g+gp
        T=RpW/(g*gp*gt)
        F=numpy.sum(T)
        T*=(1./g+1./gt)
        U=numpy.sum(T)
        W=numpy.sum(T*R2)
        F*=-1.5
        #excthresh is multiplied by Rho later
        excthresh[i]=Beta+0.5*F
        vxcthresh[0,i]=Beta+F+1.5*(U*dKdR[i]+W*dW0dR[i])
        vxcthresh[1,i]=1.5*W*dW0dG[i]
    exc[threshind]=excthresh
    vxc[0,threshind]=vxcthresh[0,:]
    vxc[1,threshind]=vxcthresh[1,:]

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
            If xctype is GGA, ao[0] is AO value and ao[1:3] are the real space
            gradients.  If xctype is meta-GGA, ao[4:10] are second derivatives
            of ao values.
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
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

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
        wv = numpy.empty((4,ngrids))
        if spin == 0:
            assert(vsigma is not None and rho.ndim==2)
            wv[0]  = weight * vrho * .5
            wv[1:4] = rho[1:4] * (weight * vsigma * 2)
        else:
            rho_a, rho_b = rho
            wv[0]  = weight * vrho * .5
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

# JCP, 138, 244108
# JCP, 112, 7002
    if xctype == 'MGGA':
        vlapl, vtau = vxc[2:]

        if vlapl is None:
            vlapl = 0
        else:
            if spin != 0:
                if transpose_for_uks:
                    vlapl = vlapl.T
                vlapl = vlapl[0]
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            #:aow = numpy.einsum('pi,p->pi', ao2, .5 * weight * vlapl, out=aow)
            aow = _scale_ao(ao2, .5 * weight * vlapl, out=aow)
            mat += _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)

        if spin != 0:
            if transpose_for_uks:
                vtau = vtau.T
            vtau = vtau[0]
        wv = weight * (.25*vtau + vlapl)
        #:aow = numpy.einsum('pi,p->pi', ao[1], wv, out=aow)
        aow = _scale_ao(ao[1], wv, out=aow)
        mat += _dot_ao_ao(mol, ao[1], aow, non0tab, shls_slice, ao_loc)
        #:aow = numpy.einsum('pi,p->pi', ao[2], wv, out=aow)
        aow = _scale_ao(ao[2], wv, out=aow)
        mat += _dot_ao_ao(mol, ao[2], aow, non0tab, shls_slice, ao_loc)
        #:aow = numpy.einsum('pi,p->pi', ao[3], wv, out=aow)
        aow = _scale_ao(ao[3], wv, out=aow)
        mat += _dot_ao_ao(mol, ao[3], aow, non0tab, shls_slice, ao_loc)

    return mat + mat.T.conj()


def _dot_ao_ao(mol, ao1, ao2, non0tab, shls_slice, ao_loc, hermi=0):
    '''return numpy.dot(ao1.T, ao2)'''
    ngrids, nao = ao1.shape
    if nao < SWITCH_SIZE:
        return lib.dot(ao1.T.conj(), ao2)

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

    if non0tab is None or shls_slice is None or ao_loc is None:
        pnon0tab = pshls_slice = pao_loc = lib.c_null_ptr()
    else:
        pnon0tab    = non0tab.ctypes.data_as(ctypes.c_void_p)
        pshls_slice = (ctypes.c_int*2)(*shls_slice)
        pao_loc     = ao_loc.ctypes.data_as(ctypes.c_void_p)

    vv = numpy.empty((nao,nao), dtype=ao1.dtype)
    fn(vv.ctypes.data_as(ctypes.c_void_p),
       ao1.ctypes.data_as(ctypes.c_void_p),
       ao2.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(ngrids),
       ctypes.c_int(mol.nbas), ctypes.c_int(hermi),
       pnon0tab, pshls_slice, pao_loc)
    return vv

def _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    '''return numpy.dot(ao, dm)'''
    ngrids, nao = ao.shape
    if nao < SWITCH_SIZE:
        return lib.dot(dm.T, ao.T).T

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if ao.dtype == dm.dtype == numpy.double:
        fn = libdft.VXCdot_ao_dm
    else:
        fn = libdft.VXCzdot_ao_dm
        ao = numpy.asarray(ao, numpy.complex128)
        dm = numpy.asarray(dm, numpy.complex128)

    if non0tab is None or shls_slice is None or ao_loc is None:
        pnon0tab = pshls_slice = pao_loc = lib.c_null_ptr()
    else:
        pnon0tab    = non0tab.ctypes.data_as(ctypes.c_void_p)
        pshls_slice = (ctypes.c_int*2)(*shls_slice)
        pao_loc     = ao_loc.ctypes.data_as(ctypes.c_void_p)

    vm = numpy.ndarray((ngrids,dm.shape[1]), dtype=ao.dtype, order='F', buffer=out)
    dm = numpy.asarray(dm, order='C')
    fn(vm.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       dm.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       pnon0tab, pshls_slice, pao_loc)
    return vm

def _scale_ao(ao, wv, out=None):
    #:aow = numpy.einsum('npi,np->pi', ao[:4], wv)
    if wv.ndim == 2:
        ao = ao.transpose(0,2,1)
    else:
        ngrids, nao = ao.shape
        ao = ao.T.reshape(1,nao,ngrids)
        wv = wv.reshape(1,ngrids)

    wv = numpy.asarray(wv, order='C')
    comp, nao, ngrids = ao.shape
    aow = numpy.ndarray((nao,ngrids), dtype=ao.dtype, buffer=out).T

    if not ao.flags.c_contiguous:
        aow = numpy.einsum('nip,np->pi', ao, wv)
    elif aow.dtype == numpy.double:
        libdft.VXC_dscale_ao(aow.ctypes.data_as(ctypes.c_void_p),
                             ao.ctypes.data_as(ctypes.c_void_p),
                             wv.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(comp), ctypes.c_int(nao),
                             ctypes.c_int(ngrids))
    elif aow.dtype == numpy.complex128:
        libdft.VXC_zscale_ao(aow.ctypes.data_as(ctypes.c_void_p),
                             ao.ctypes.data_as(ctypes.c_void_p),
                             wv.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(comp), ctypes.c_int(nao),
                             ctypes.c_int(ngrids))
    else:
        aow = numpy.einsum('nip,np->pi', ao, wv)
    return aow

def _contract_rho(bra, ket):
    #:rho  = numpy.einsum('pi,pi->p', bra.real, ket.real)
    #:rho += numpy.einsum('pi,pi->p', bra.imag, ket.imag)
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

def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
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
    >>> dm = numpy.random.random((nao,nao))
    >>> ni = dft.numint.NumInt()
    >>> nelec, exc, vxc = ni.nr_rks(mol, grids, 'lda,vwn', dm)
    '''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    if isinstance(dms, numpy.ndarray):
        vmat = numpy.zeros((nset,nao,nao), dtype=dms.dtype)
    else:
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.result_type(*dms))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'LDA')
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                      verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho * weight
                nelec[idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                # *.5 because vmat + vmat.T
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho, out=aow)
                aow = _scale_ao(ao, .5*weight*vrho, out=aow)
                vmat[idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'GGA')
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                      verbose=verbose)[:2]
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
# ref eval_mat function
                wv = _rks_gga_wv0(rho, vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=aow)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho = exc = vxc = wv = None
    elif xctype == 'NLC':
        nlc_pars = ni.nlc_coeff(xc_code[:-6])
        if nlc_pars == [0,0]:
            raise NotImplementedError('VV10 cannot be used with %s. '
                                      'The supported functionals are %s' %
                                      (xc_code[:-6], ni.libxc.VV10_XC))
        ao_deriv = 1
        vvrho=numpy.empty([nset,4,0])
        vvweight=numpy.empty([nset,0])
        vvcoords=numpy.empty([nset,0,3])
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhotmp = numpy.empty([0,4,weight.size])
            weighttmp = numpy.empty([0,weight.size])
            coordstmp = numpy.empty([0,weight.size,3])
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'GGA')
                rho = numpy.expand_dims(rho,axis=0)
                rhotmp = numpy.concatenate((rhotmp,rho),axis=0)
                weighttmp = numpy.concatenate((weighttmp,numpy.expand_dims(weight,axis=0)),axis=0)
                coordstmp = numpy.concatenate((coordstmp,numpy.expand_dims(coords,axis=0)),axis=0)
                rho = None
            vvrho=numpy.concatenate((vvrho,rhotmp),axis=2)
            vvweight=numpy.concatenate((vvweight,weighttmp),axis=1)
            vvcoords=numpy.concatenate((vvcoords,coordstmp),axis=1)
            rhotmp = weighttmp = coordstmp = None
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'GGA')
                exc, vxc = _vv10nlc(rho,coords,vvrho[idm],vvweight[idm],vvcoords[idm],nlc_pars)
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
# ref eval_mat function
                wv = _rks_gga_wv0(rho, vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=aow)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho = exc = vxc = wv = None
        vvrho = vvweight = vvcoords = None
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'MGGA')
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1,
                                      verbose=verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)

                wv = _rks_gga_wv0(rho, vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao[:4], wv, out=aow)
                aow = _scale_ao(ao[:4], wv, out=aow)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
                wv = (.5 * .5 * weight * vtau).reshape(-1,1)
                vmat[idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)

                rho = exc = vxc = vrho = vsigma = wv = None

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].conj().T
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, vmat

def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
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
            Input density matrices symmetric or not
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
    if xctype == 'NLC':
        dms_sf = dms[0] + dms[1]
        nelec, excsum, vmat = nr_rks(ni, mol, grids, xc_code, dms_sf, relativity, hermi,
                                     max_memory, verbose)
        return [nelec,nelec], excsum, numpy.asarray([vmat,vmat])

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((2,nset,nao,nao), dtype=numpy.result_type(dma, dmb))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b * weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)

                # *.5 due to +c.c. in the end
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,0], out=aow)
                aow = _scale_ao(ao, .5*weight*vrho[:,0], out=aow)
                vmat[0,idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1], out=aow)
                aow = _scale_ao(ao, .5*weight*vrho[:,1], out=aow)
                vmat[1,idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose=verbose)[:2]
                den = rho_a[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)

                wva, wvb = _uks_gga_wv0((rho_a,rho_b), vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wva, out=aow)
                aow = _scale_ao(ao, wva, out=aow)
                vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = _scale_ao(ao, wvb, out=aow)
                vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = wva = wvb = None
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose=verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho_a[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)

                wva, wvb = _uks_gga_wv0((rho_a,rho_b), vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao[:4], wva, out=aow)
                aow = _scale_ao(ao[:4], wva, out=aow)
                vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('npi,np->pi', ao[:4], wvb, out=aow)
                aow = _scale_ao(ao[:4], wvb, out=aow)
                vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
                wv = (.25 * weight * vtau[:,0]).reshape(-1,1)
                vmat[0,idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[0,idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[0,idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
                wv = (.25 * weight * vtau[:,1]).reshape(-1,1)
                vmat[1,idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
                vmat[1,idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
                vmat[1,idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = vrho = vsigma = wva = wvb = None

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].conj().T
        vmat[1,i] = vmat[1,i] + vmat[1,i].conj().T
    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
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

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    if isinstance(dms, numpy.ndarray):
        vmat = numpy.zeros((nset,nao,nao), dtype=dms.dtype)
    else:
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.result_type(*dms))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            if fxc is None:
                rho = make_rho0(0, ao, mask, 'LDA')
                fxc0 = ni.eval_xc(xc_code, rho, 0, relativity, 2,
                                  verbose=verbose)[2]
                frr = fxc0[0]
            else:
                frr = fxc[0][ip:ip+ngrid]
                ip += ngrid

            for i in range(nset):
                rho1 = make_rho(i, ao, mask, 'LDA')
                #:aow = numpy.einsum('pi,p->pi', ao, weight*frr*rho1, out=aow)
                aow = _scale_ao(ao, weight*frr*rho1, out=aow)
                vmat[i] += _dot_ao_ao(mol, aow, ao, mask, shls_slice, ao_loc)
                rho1 = None

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            if rho0 is None:
                rho = make_rho0(0, ao, mask, 'GGA')
            else:
                rho = numpy.asarray(rho0[:,ip:ip+ngrid], order='C')
            if vxc is None or fxc is None:
                vxc0, fxc0 = ni.eval_xc(xc_code, rho, 0, relativity, 2,
                                        verbose=verbose)[1:3]
            else:
                vxc0 = (None, vxc[1][ip:ip+ngrid])
                fxc0 = (fxc[0][ip:ip+ngrid], fxc[1][ip:ip+ngrid], fxc[2][ip:ip+ngrid])
                ip += ngrid

            for i in range(nset):
                rho1 = make_rho(i, ao, mask, 'GGA')
                wv = _rks_gga_wv1(rho, rho1, vxc0, fxc0, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=aow)
                vmat[i] += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
                rho1 = sigma1 = None

        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmat[i] = vmat[i] + vmat[i].T.conj()

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat

def nr_rks_fxc_st(ni, mol, grids, xc_code, dm0, dms_alpha, relativity=0, singlet=True,
                  rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Associated to singlet or triplet Hessian
    Note the difference to nr_rks_fxc, dms_alpha is the response density
    matrices of alpha spin, alpha+/-beta DM is applied due to singlet/triplet
    coupling

    Ref. CPL, 256, 454
    '''
    xctype = ni._xc_type(xc_code)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms_alpha, hermi=0)
    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        make_rho0 = ni._gen_rho_evaluator(mol, dm0, hermi=1)[0]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    if isinstance(dms_alpha, numpy.ndarray):
        vmat = numpy.zeros((nset,nao,nao), dtype=dms_alpha.dtype)
    else:
        vmat = numpy.zeros((nset,nao,nao), dtype=numpy.result_type(*dms_alpha))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            if fxc is None:
                rho = make_rho0(0, ao, mask, 'LDA')
                rho *= .5  # alpha density
                fxc0 = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                if fxc[0].ndim == 1:
                    raise RuntimeError('cached (rho, vxc, fxc) need to be '
                                       'generated by cache_xc_kernel with flag spin=1')
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                ip += ngrid
            if singlet:
                frho = u_u + u_d
                if 0:
                    rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, 'LDA')
                    fxc_test = ni.eval_xc(xc_code, rho, 0, deriv=2)[2]
                    assert(numpy.linalg.norm(fxc_test[0]*2-frho) < 1e-4)
            else:
                frho = u_u - u_d

            for i in range(nset):
                rho1 = make_rho(i, ao, mask, 'LDA')
                #:aow = numpy.einsum('pi,p->pi', ao, weight*frho*rho1, out=aow)
                aow = _scale_ao(ao, weight*frho*rho1, out=aow)
                vmat[i] += _dot_ao_ao(mol, aow, ao, mask, shls_slice, ao_loc)
                rho1 = None

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            if vxc is None or fxc is None:
                rho = make_rho0(0, ao, mask, 'GGA')
                rho *= .5  # alpha density
                vxc0, fxc0 = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[1:3]

                vsigma = vxc0[1].T
                u_u, u_d, d_d = fxc0[0].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc0[1].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc0[2].T  # v2sigma2
            else:
                if rho0[0].ndim == 1:
                    raise RuntimeError('cached (rho, vxc, fxc) need to be '
                                       'generated by cache_xc_kernel with flag spin=1')
                rho = rho0[0][:,ip:ip+ngrid]
                vsigma = vxc[1][ip:ip+ngrid].T
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1][ip:ip+ngrid].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2][ip:ip+ngrid].T  # v2sigma2
                ip += ngrid

            # Factorization differs to CPL, 256, 454, to use _rks_gga_wv1 function
            if singlet:
                fgamma = vsigma[0] + vsigma[1] * .5
                frho = u_u + u_d
                fgg = uu_uu + .5*ud_ud + 2*uu_ud + uu_dd
                frhogamma = u_uu + u_dd + u_ud
            else:
                fgamma = vsigma[0] - vsigma[1] * .5
                frho = u_u - u_d
                fgg = uu_uu - uu_dd
                frhogamma = u_uu - u_dd

            for i in range(nset):
                # rho1[0 ] = |b><j| z_{bj}
                # rho1[1:] = \nabla(|b><j|) z_{bj}
                rho1 = make_rho(i, ao, mask, 'GGA')
                wv = _rks_gga_wv1(rho, rho1, (None,fgamma), (frho,frhogamma,fgg), weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=aow)
                vmat[i] += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
                rho1 = sigma1 = None

        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmat[i] = vmat[i] + vmat[i].T.conj()

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if isinstance(dms_alpha, numpy.ndarray) and dms_alpha.ndim == 2:
        vmat = vmat[0]
    return vmat

def _rks_gga_wv0(rho, vxc, weight):
    vrho, vgamma = vxc[:2]
    ngrid = vrho.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = weight * vrho
    wv[1:] = (weight * vgamma * 2) * rho[1:4]
    wv[0] *= .5  # v+v.T should be applied in the caller
    return wv

def _rks_gga_wv1(rho0, rho1, vxc, fxc, weight):
    vgamma = vxc[1]
    frho, frhogamma, fgg = fxc[:3]
    # sigma1 ~ \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    ngrid = vgamma.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = frho * rho1[0]
    wv[0] += frhogamma * sigma1 * 2
    wv[1:] = (fgg * sigma1 * 4 + frhogamma * rho1[0] * 2) * rho0[1:4]
    wv[1:]+= vgamma * rho1[1:4] * 2
    wv *= weight
    wv[0] *= .5  # v+v.T should be applied in the caller
    return wv

def _rks_gga_wv2(rho0, rho1, fxc, kxc, weight):
    frr, frg, fgg = fxc[:3]
    frrr, frrg, frgg, fggg = kxc
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:], rho1[1:])
    r1r1 = rho1[0]**2
    s1s1 = sigma1**2
    r1s1 = rho1[0] * sigma1
    sigma2 = numpy.einsum('xi,xi->i', rho1[1:], rho1[1:])
    ngrid = frrr.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = frrr * r1r1
    wv[0] += 4 * frrg * r1s1
    wv[0] += 4 * frgg * s1s1
    wv[0] += 2 * frg * sigma2
    wv[1:]  = 2 * frrg * r1r1 * rho0[1:]
    wv[1:] += 8 * frgg * r1s1 * rho0[1:]
    wv[1:] += 4 * frg * rho1[0] * rho1[1:]
    wv[1:] += 4 * fgg * sigma2 * rho0[1:]
    wv[1:] += 8 * fgg * sigma1 * rho1[1:]
    wv[1:] += 8 * fggg * s1s1 * rho0[1:]
    wv *= weight
    wv[0]*=.5  # v+v.T should be applied in the caller
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

    dma, dmb = _format_uks_dm(dms)
    nao = dms.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi)[0]

    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        make_rho0 = ni._gen_rho_evaluator(mol, _format_uks_dm(dm0), 1)[0]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((2,nset,nao,nao), dtype=numpy.result_type(dma, dmb))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            if fxc is None:
                rho0a = make_rho0(0, ao, mask, xctype)
                rho0b = make_rho0(1, ao, mask, xctype)
                fxc0 = ni.eval_xc(xc_code, (rho0a,rho0b), 1, relativity, 2,
                                  verbose=verbose)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                ip += ngrid

            for i in range(nset):
                rho1a = make_rhoa(i, ao, mask, xctype)
                rho1b = make_rhob(i, ao, mask, xctype)
                wv = u_u * rho1a + u_d * rho1b
                wv *= weight
                #:aow = numpy.einsum('pi,p->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=aow)
                vmat[0,i] += _dot_ao_ao(mol, aow, ao, mask, shls_slice, ao_loc)
                wv = u_d * rho1a + d_d * rho1b
                wv *= weight
                #:aow = numpy.einsum('pi,p->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=aow)
                vmat[1,i] += _dot_ao_ao(mol, aow, ao, mask, shls_slice, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            if rho0 is None:
                rho0a = make_rho0(0, ao, mask, xctype)
                rho0b = make_rho0(1, ao, mask, xctype)
            else:
                rho0a = rho0[0][:,ip:ip+ngrid]
                rho0b = rho0[1][:,ip:ip+ngrid]
            if vxc is None or fxc is None:
                vxc0, fxc0 = ni.eval_xc(xc_code, (rho0a,rho0b), 1, relativity, 2,
                                        verbose=verbose)[1:3]
            else:
                vxc0 = (None, vxc[1][ip:ip+ngrid])
                fxc0 = (fxc[0][ip:ip+ngrid], fxc[1][ip:ip+ngrid], fxc[2][ip:ip+ngrid])
                ip += ngrid

            for i in range(nset):
                rho1a = make_rhoa(i, ao, mask, xctype)
                rho1b = make_rhob(i, ao, mask, xctype)
                wva, wvb = _uks_gga_wv1((rho0a,rho0b), (rho1a,rho1b), vxc0, fxc0, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wva, out=aow)
                aow = _scale_ao(ao, wva, out=aow)
                vmat[0,i] += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = _scale_ao(ao, wvb, out=aow)
                vmat[1,i] += _dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmat[0,i] = vmat[0,i] + vmat[0,i].T.conj()
            vmat[1,i] = vmat[1,i] + vmat[1,i].T.conj()

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
    return vmat

def _uks_gga_wv0(rho, vxc, weight):
    rhoa, rhob = rho
    vrho, vsigma = vxc[:2]
    ngrid = vrho.shape[0]
    wva = numpy.empty((4,ngrid))
    wva[0]  = weight * vrho[:,0] * .5  # v+v.T should be applied in the caller
    wva[1:] = rhoa[1:4] * (weight * vsigma[:,0] * 2)  # sigma_uu
    wva[1:]+= rhob[1:4] * (weight * vsigma[:,1])      # sigma_ud
    wvb = numpy.empty((4,ngrid))
    wvb[0]  = weight * vrho[:,1] * .5  # v+v.T should be applied in the caller
    wvb[1:] = rhob[1:4] * (weight * vsigma[:,2] * 2)  # sigma_dd
    wvb[1:]+= rhoa[1:4] * (weight * vsigma[:,1])      # sigma_ud
    return wva, wvb

def _uks_gga_wv1(rho0, rho1, vxc, fxc, weight):
    uu, ud, dd = vxc[1].T
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    ngrid = uu.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4])
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4])

    wva = numpy.empty((4,ngrid))
    wvb = numpy.empty((4,ngrid))
    # alpha = alpha-alpha * alpha
    wva[0]  = u_u * rho1a[0]
    wva[0] += u_uu * a0a1 * 2
    wva[0] += u_ud * b0a1
    wva[1:] = uu * rho1a[1:4] * 2
    wva[1:]+= u_uu * rho1a[0] * rho0a[1:4] * 2
    wva[1:]+= u_ud * rho1a[0] * rho0b[1:4]
    wva[1:]+= uu_uu * a0a1 * rho0a[1:4] * 4
    wva[1:]+= uu_ud * a0a1 * rho0b[1:4] * 2
    wva[1:]+= uu_ud * b0a1 * rho0a[1:4] * 2
    wva[1:]+= ud_ud * b0a1 * rho0b[1:4]

    # alpha = alpha-beta  * beta
    wva[0] += u_d * rho1b[0]
    wva[0] += u_ud * a0b1
    wva[0] += u_dd * b0b1 * 2
    wva[1:]+= ud * rho1b[1:4]
    wva[1:]+= d_uu * rho1b[0] * rho0a[1:4] * 2
    wva[1:]+= d_ud * rho1b[0] * rho0b[1:4]
    wva[1:]+= uu_ud * a0b1 * rho0a[1:4] * 2
    wva[1:]+= ud_ud * a0b1 * rho0b[1:4]
    wva[1:]+= uu_dd * b0b1 * rho0a[1:4] * 4
    wva[1:]+= ud_dd * b0b1 * rho0b[1:4] * 2
    wva *= weight
    wva[0] *= .5  # v+v.T should be applied in the caller

    # beta = beta-alpha * alpha
    wvb[0]  = u_d * rho1a[0]
    wvb[0] += d_ud * b0a1
    wvb[0] += d_uu * a0a1 * 2
    wvb[1:] = ud * rho1a[1:4]
    wvb[1:]+= u_dd * rho1a[0] * rho0b[1:4] * 2
    wvb[1:]+= u_ud * rho1a[0] * rho0a[1:4]
    wvb[1:]+= ud_dd * b0a1 * rho0b[1:4] * 2
    wvb[1:]+= ud_ud * b0a1 * rho0a[1:4]
    wvb[1:]+= uu_dd * a0a1 * rho0b[1:4] * 4
    wvb[1:]+= uu_ud * a0a1 * rho0a[1:4] * 2

    # beta = beta-beta  * beta
    wvb[0] += d_d * rho1b[0]
    wvb[0] += d_dd * b0b1 * 2
    wvb[0] += d_ud * a0b1
    wvb[1:]+= dd * rho1b[1:4] * 2
    wvb[1:]+= d_dd * rho1b[0] * rho0b[1:4] * 2
    wvb[1:]+= d_ud * rho1b[0] * rho0a[1:4]
    wvb[1:]+= dd_dd * b0b1 * rho0b[1:4] * 4
    wvb[1:]+= ud_dd * b0b1 * rho0a[1:4] * 2
    wvb[1:]+= ud_dd * a0b1 * rho0b[1:4] * 2
    wvb[1:]+= ud_ud * a0b1 * rho0a[1:4]
    wvb *= weight
    wvb[0] *= .5  # v+v.T should be applied in the caller
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
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4])
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4])
    a1a1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1a[1:4])
    a1b1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1b[1:4])
    b1a1 = a1b1
    b1b1 = numpy.einsum('xi,xi->i', rho1b[1:4], rho1b[1:4])
    a0a1_a0a1 = numpy.einsum('i,i->i', a0a1, a0a1)
    a0a1_a0b1 = numpy.einsum('i,i->i', a0a1, a0b1)
    a0a1_b0a1 = numpy.einsum('i,i->i', a0a1, b0a1)
    a0a1_b0b1 = numpy.einsum('i,i->i', a0a1, b0b1)
    a0b1_a0a1 = a0a1_a0b1
    a0b1_a0b1 = numpy.einsum('i,i->i', a0b1, a0b1)
    a0b1_b0a1 = numpy.einsum('i,i->i', a0b1, b0a1)
    a0b1_b0b1 = numpy.einsum('i,i->i', a0b1, b0b1)
    b0a1_a0a1 = a0a1_b0a1
    b0a1_a0b1 = a0b1_b0a1
    b0a1_b0a1 = numpy.einsum('i,i->i', b0a1, b0a1)
    b0a1_b0b1 = numpy.einsum('i,i->i', b0a1, b0b1)
    b0b1_a0a1 = a0a1_b0b1
    b0b1_a0b1 = a0b1_b0b1
    b0b1_b0a1 = b0a1_b0b1
    b0b1_b0b1 = numpy.einsum('i,i->i', b0b1, b0b1)

    wva = numpy.zeros((4,ngrid))
    wva[0] += numpy.einsum('i,i,i->i', u_u_u, rho1a[0], rho1a[0])
    wva[0] += numpy.einsum('i,i,i->i', u_u_d, rho1a[0], rho1b[0]) * 2
    wva[0] += numpy.einsum('i,i,i->i', u_d_d, rho1b[0], rho1b[0])
    wva[0] += numpy.einsum('i,i->i', u_uu, a1a1) * 2
    wva[0] += numpy.einsum('i,i->i', u_ud, a1b1) * 2
    wva[0] += numpy.einsum('i,i->i', u_dd, b1b1) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', u_uu, rho1a[0], rho1a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', d_uu, rho1b[0], rho1a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', u_ud, rho1a[0], rho1b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', d_ud, rho1b[0], rho1b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu, a0a1, rho1a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud, a0a1, rho1b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud, a0b1, rho1a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud, a0b1, rho1b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud, b0a1, rho1a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_dd, b0b1, rho1a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud, b0a1, rho1b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_dd, b0b1, rho1b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu, a1a1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud, a1b1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_dd, b1b1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud, a1a1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud, a1b1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_dd, b1b1, rho0b[1:]) * 2
    wva[0] += numpy.einsum('i,i,i->i', u_u_uu, rho1a[0], a0a1) * 4
    wva[0] += numpy.einsum('i,i,i->i', u_d_uu, rho1b[0], a0a1) * 4
    wva[0] += numpy.einsum('i,i,i->i', u_u_ud, rho1a[0], a0b1) * 2
    wva[0] += numpy.einsum('i,i,i->i', u_d_ud, rho1b[0], a0b1) * 2
    wva[0] += numpy.einsum('i,i,i->i', u_u_ud, rho1a[0], b0a1) * 2
    wva[0] += numpy.einsum('i,i,i->i', u_d_ud, rho1b[0], b0a1) * 2
    wva[0] += numpy.einsum('i,i,i->i', u_u_dd, rho1a[0], b0b1) * 4
    wva[0] += numpy.einsum('i,i,i->i', u_d_dd, rho1b[0], b0b1) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_u_uu, rho1a[0], rho1a[0], rho0a[1:]) * 2
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_d_uu, rho1a[0], rho1b[0], rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_d_uu, rho1b[0], rho1b[0], rho0a[1:]) * 2
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_u_ud, rho1a[0], rho1a[0], rho0a[1:])
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_d_ud, rho1a[0], rho1b[0], rho0a[1:]) * 2
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_d_ud, rho1b[0], rho1b[0], rho0a[1:])
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_uu_uu, rho1a[0], a0a1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_uu_uu, rho1b[0], a0a1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_uu_ud, rho1a[0], a0b1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_uu_ud, rho1b[0], a0b1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_uu_ud, rho1a[0], b0a1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_uu_ud, rho1b[0], b0a1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_uu_dd, rho1a[0], b0b1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_uu_dd, rho1b[0], b0b1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_uu_ud, rho1a[0], a0a1, rho0b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_uu_ud, rho1b[0], a0a1, rho0b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_ud, rho1a[0], a0b1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_ud, rho1b[0], a0b1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_ud, rho1a[0], b0a1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_ud, rho1b[0], b0a1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_dd, rho1a[0], b0b1, rho0b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_dd, rho1b[0], b0b1, rho0b[1:]) * 4
    wva[0] += numpy.einsum('i,i->i', u_uu_uu, a0a1_a0a1) * 4
    wva[0] += numpy.einsum('i,i->i', u_uu_ud, a0a1_a0b1) * 2
    wva[0] += numpy.einsum('i,i->i', u_uu_ud, a0b1_a0a1) * 2
    wva[0] += numpy.einsum('i,i->i', u_ud_ud, a0b1_a0b1)
    wva[0] += numpy.einsum('i,i->i', u_uu_ud, a0a1_b0a1) * 4
    wva[0] += numpy.einsum('i,i->i', u_ud_ud, a0b1_b0a1) * 2
    wva[0] += numpy.einsum('i,i->i', u_uu_dd, a0a1_b0b1) * 8
    wva[0] += numpy.einsum('i,i->i', u_ud_dd, a0b1_b0b1) * 4
    wva[0] += numpy.einsum('i,i->i', u_ud_ud, b0a1_b0a1)
    wva[0] += numpy.einsum('i,i->i', u_ud_dd, b0a1_b0b1) * 2
    wva[0] += numpy.einsum('i,i->i', u_ud_dd, b0b1_b0a1) * 2
    wva[0] += numpy.einsum('i,i->i', u_dd_dd, b0b1_b0b1) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu_uu, a0a1_a0a1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu_ud, a0a1_a0b1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu_ud, a0b1_a0a1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, a0b1_a0b1, rho0a[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu_ud, a0a1_a0a1, rho0b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, a0a1_a0b1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, a0b1_a0a1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud_ud, a0b1_a0b1, rho0b[1:])
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu_ud, a0a1_b0a1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, a0b1_b0a1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_uu_dd, a0a1_b0b1, rho0a[1:]) * 16
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, a0b1_b0b1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, a0a1_b0a1, rho0b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud_ud, a0b1_b0a1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, a0a1_b0b1, rho0b[1:]) * 8
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, a0b1_b0b1, rho0b[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, b0a1_b0a1, rho0a[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, b0b1_b0a1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, b0a1_b0b1, rho0a[1:]) * 4
    wva[1:] += numpy.einsum('i,i,xi->xi', uu_dd_dd, b0b1_b0b1, rho0a[1:]) * 8
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud_ud, b0a1_b0a1, rho0b[1:])
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, b0b1_b0a1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, b0a1_b0b1, rho0b[1:]) * 2
    wva[1:] += numpy.einsum('i,i,xi->xi', ud_dd_dd, b0b1_b0b1, rho0b[1:]) * 4
    wva *= weight
    wva[0]*=.5  # v+v.T should be applied in the caller

    wvb = numpy.zeros((4,ngrid))
    wvb[0] += numpy.einsum('i,i,i->i', d_d_d, rho1b[0], rho1b[0])
    wvb[0] += numpy.einsum('i,i,i->i', u_d_d, rho1b[0], rho1a[0]) * 2
    wvb[0] += numpy.einsum('i,i,i->i', u_u_d, rho1a[0], rho1a[0])
    wvb[0] += numpy.einsum('i,i->i', d_dd, b1b1) * 2
    wvb[0] += numpy.einsum('i,i->i', d_ud, b1a1) * 2
    wvb[0] += numpy.einsum('i,i->i', d_uu, a1a1) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', d_dd, rho1b[0], rho1b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', u_dd, rho1a[0], rho1b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', d_ud, rho1b[0], rho1a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', u_ud, rho1a[0], rho1a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', dd_dd, b0b1, rho1b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd, b0b1, rho1a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd, b0a1, rho1b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud, b0a1, rho1a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd, a0b1, rho1b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_dd, a0a1, rho1b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud, a0b1, rho1a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud, a0a1, rho1a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', dd_dd, b1b1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd, b1a1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_dd, a1a1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd, b1b1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud, b1a1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud, a1a1, rho0a[1:]) * 2
    wvb[0] += numpy.einsum('i,i,i->i', d_d_dd, rho1b[0], b0b1) * 4
    wvb[0] += numpy.einsum('i,i,i->i', u_d_dd, rho1a[0], b0b1) * 4
    wvb[0] += numpy.einsum('i,i,i->i', d_d_ud, rho1b[0], b0a1) * 2
    wvb[0] += numpy.einsum('i,i,i->i', u_d_ud, rho1a[0], b0a1) * 2
    wvb[0] += numpy.einsum('i,i,i->i', d_d_ud, rho1b[0], a0b1) * 2
    wvb[0] += numpy.einsum('i,i,i->i', u_d_ud, rho1a[0], a0b1) * 2
    wvb[0] += numpy.einsum('i,i,i->i', d_d_uu, rho1b[0], a0a1) * 4
    wvb[0] += numpy.einsum('i,i,i->i', u_d_uu, rho1a[0], a0a1) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_d_dd, rho1b[0], rho1b[0], rho0b[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_d_dd, rho1b[0], rho1a[0], rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_u_dd, rho1a[0], rho1a[0], rho0b[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_d_ud, rho1b[0], rho1b[0], rho0b[1:])
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_d_ud, rho1b[0], rho1a[0], rho0b[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_u_ud, rho1a[0], rho1a[0], rho0b[1:])
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_dd_dd, rho1b[0], b0b1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_dd_dd, rho1a[0], b0b1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_dd, rho1b[0], b0a1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_dd, rho1a[0], b0a1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_dd, rho1b[0], a0b1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_dd, rho1a[0], a0b1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_uu_dd, rho1b[0], a0a1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_uu_dd, rho1a[0], a0a1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_dd, rho1b[0], b0b1, rho0a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_dd, rho1a[0], b0b1, rho0a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_ud, rho1b[0], b0a1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_ud, rho1a[0], b0a1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_ud_ud, rho1b[0], a0b1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_ud_ud, rho1a[0], a0b1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', d_uu_ud, rho1b[0], a0a1, rho0a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,i,xi->xi', u_uu_ud, rho1a[0], a0a1, rho0a[1:]) * 4
    wvb[0] += numpy.einsum('i,i->i', d_dd_dd, b0b1_b0b1) * 4
    wvb[0] += numpy.einsum('i,i->i', d_ud_dd, b0b1_b0a1) * 2
    wvb[0] += numpy.einsum('i,i->i', d_ud_dd, b0a1_b0b1) * 2
    wvb[0] += numpy.einsum('i,i->i', d_ud_ud, b0a1_b0a1)
    wvb[0] += numpy.einsum('i,i->i', d_ud_dd, b0b1_a0b1) * 4
    wvb[0] += numpy.einsum('i,i->i', d_ud_ud, b0a1_a0b1) * 2
    wvb[0] += numpy.einsum('i,i->i', d_uu_dd, b0b1_a0a1) * 8
    wvb[0] += numpy.einsum('i,i->i', d_uu_ud, b0a1_a0a1) * 4
    wvb[0] += numpy.einsum('i,i->i', d_ud_ud, a0b1_a0b1)
    wvb[0] += numpy.einsum('i,i->i', d_uu_ud, a0b1_a0a1) * 2
    wvb[0] += numpy.einsum('i,i->i', d_uu_ud, a0a1_a0b1) * 2
    wvb[0] += numpy.einsum('i,i->i', d_uu_uu, a0a1_a0a1) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', dd_dd_dd, b0b1_b0b1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd_dd, b0b1_b0a1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd_dd, b0a1_b0b1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, b0a1_b0a1, rho0b[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd_dd, b0b1_b0b1, rho0a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, b0b1_b0a1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, b0a1_b0b1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_ud, b0a1_b0a1, rho0a[1:])
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_dd_dd, b0b1_a0b1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, b0a1_a0b1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_dd_dd, b0b1_a0a1, rho0b[1:]) * 16
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, b0a1_a0a1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, b0b1_a0b1, rho0a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_ud, b0a1_a0b1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, b0b1_a0a1, rho0a[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, b0a1_a0a1, rho0a[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_dd, a0b1_a0b1, rho0b[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, a0a1_a0b1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud_dd, a0b1_a0a1, rho0b[1:]) * 4
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_uu_dd, a0a1_a0a1, rho0b[1:]) * 8
    wvb[1:] += numpy.einsum('i,i,xi->xi', ud_ud_ud, a0b1_a0b1, rho0a[1:])
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, a0a1_a0b1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_ud_ud, a0b1_a0a1, rho0a[1:]) * 2
    wvb[1:] += numpy.einsum('i,i,xi->xi', uu_uu_ud, a0a1_a0a1, rho0a[1:]) * 4
    wvb *= weight
    wvb[0]*=.5

    return wva, wvb

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
    ao_deriv = 0
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if spin == 0:
        nao = mo_coeff.shape[0]
        rho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho.append(ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, xctype))
        rho = numpy.hstack(rho)
    else:
        nao = mo_coeff[0].shape[0]
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa.append(ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype))
            rhob.append(ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc(xc_code, rho, spin, 0, 2, 0)[1:3]
    return rho, vxc, fxc

def get_rho(ni, mol, dm, grids, max_memory=2000):
    '''Density in real space
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1)
    assert(nset == 1)
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao, mask, 'LDA')
    return rho


class NumInt(object):
    def __init__(self):
        self.libxc = libxc
        self.omega = None  # RSH paramter

    @lib.with_doc(nr_vxc.__doc__)
    def nr_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        if spin == 0:
            return self.nr_rks(mol, grids, xc_code, dms, relativity, hermi,
                               max_memory, verbose)
        else:
            return self.nr_uks(mol, grids, xc_code, dms, relativity, hermi,
                               max_memory, verbose)

    @lib.with_doc(nr_fxc.__doc__)
    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        if spin == 0:
            return self.nr_rks_fxc(mol, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, max_memory, verbose)
        else:
            return self.nr_uks_fxc(mol, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, max_memory, verbose)

    nr_rks = nr_rks
    nr_uks = nr_uks
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    cache_xc_kernel  = cache_xc_kernel
    get_rho = get_rho

    @lib.with_doc(eval_ao.__doc__)
    def eval_ao(self, mol, coords, deriv=0, shls_slice=None,
                non0tab=None, out=None, verbose=None):
        return eval_ao(mol, coords, deriv, shls_slice, non0tab, out, verbose)

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, mol, coords, relativity=0, shls_slice=None,
                  verbose=None):
        return make_mask(mol, coords, relativity, shls_slice, verbose)

    @lib.with_doc(eval_rho2.__doc__)
    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        return eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab, xctype, verbose)

    @lib.with_doc(eval_rho.__doc__)
    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        return eval_rho(mol, ao, dm, non0tab, xctype, hermi, verbose)

    def block_loop(self, mol, grids, nao, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ngrids = grids.coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nao*8*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*1200))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                 dtype=numpy.uint8)
        if buf is None:
            buf = numpy.empty((comp,blksize,nao))
        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao = self.eval_ao(mol, coords, deriv=deriv, non0tab=non0, out=buf)
            yield ao, non0, weight, coords

    def _gen_rho_evaluator(self, mol, dms, hermi=0):
        if getattr(dms, 'mo_coeff', None) is not None:
#TODO: test whether dm.mo_coeff matching dm
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
            nao = mo_coeff[0].shape[0]
            ndms = len(mo_occ)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(mol, ao, mo_coeff[idm], mo_occ[idm],
                                      non0tab, xctype)
        else:
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                dms = [dms]
            if not hermi:
# For eval_rho when xctype==GGA, which requires hermitian DMs
                dms = [(dm+dm.conj().T)*.5 for dm in dms]
            nao = dms[0].shape[0]
            ndms = len(dms)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho(mol, ao, dms[idm], non0tab, xctype, hermi=1)
        return make_rho, ndms, nao

####################
# Overwrite following functions to use custom XC functional

    def hybrid_coeff(self, xc_code, spin=0):
        return self.libxc.hybrid_coeff(xc_code, spin)

    def nlc_coeff(self, xc_code):
        return self.libxc.nlc_coeff(xc_code)

    def rsh_coeff(self, xc_code):
        return self.libxc.rsh_coeff(xc_code)

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        if omega is None: omega = self.omega
        return self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv,
                                  omega, verbose)
    eval_xc.__doc__ = libxc.eval_xc.__doc__

    def _xc_type(self, xc_code):
        return self.libxc.xc_type(xc_code)

    def rsh_and_hybrid_coeff(self, xc_code, spin=0):
        '''Range-separated parameter and HF exchange components: omega, alpha, beta

        Exc_RSH = c_SR * SR_HFX + c_LR * LR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
                = alpha * HFX + beta * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
                = alpha * LR_HFX + hyb * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec

        SR_HFX = < pi | e^{-omega r_{12}}/r_{12} | iq >
        LR_HFX = < pi | (1-e^{-omega r_{12}})/r_{12} | iq >
        alpha = c_LR
        beta = c_SR - c_LR
        '''
        omega, alpha, beta = self.rsh_coeff(xc_code)
        if self.omega is not None:
            omega = self.omega

        if abs(omega) > 1e-10:
            hyb = alpha + beta
        else:
            hyb = self.hybrid_coeff(xc_code, spin)
        return omega, alpha, hyb
_NumInt = NumInt


if __name__ == '__main__':
    import time
    from pyscf import gto
    from pyscf import dft

    mol = gto.M(
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = '6311g**',)
    mf = dft.RKS(mol)
    mf.grids.atom_grid = {"H": (30, 194), "O": (30, 194),}
    mf.grids.prune = None
    mf.grids.build()
    dm = mf.get_init_guess(key='minao')

    numpy.random.seed(1)
    dm1 = numpy.random.random((dm.shape))
    dm1 = lib.hermi_triu(dm1)
    print(time.clock())
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm1, spin=0)
    print(res[1] - -37.084047825971282)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm1,dm1), spin=1)
    print(res[1] - -92.436362308687094)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm, spin=0)
    print(res[1] - -8.6313329288394947)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm,dm), spin=1)
    print(res[1] - -21.520301399504582)
    print(time.clock())
