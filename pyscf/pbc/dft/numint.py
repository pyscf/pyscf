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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import ctypes
import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import eval_mat, _dot_ao_ao, _dot_ao_dm
from pyscf.dft.numint import _scale_ao, _contract_rho
from pyscf.dft.numint import _rks_gga_wv0, _rks_gga_wv1
from pyscf.dft.numint import _uks_gga_wv0, _uks_gga_wv1
from pyscf.dft.numint import OCCDROP
from pyscf.pbc.dft.gen_grid import libpbc, make_mask, BLKSIZE
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member

#try:
### Moderate speedup by caching eval_ao
#    from pyscf import pbc
#    from joblib import Memory
#    memory = Memory(cachedir='./tmp/', mmap_mode='r', verbose=0)
#    def memory_cache(f):
#        g = memory.cache(f)
#        def maybe_cache(*args, **kwargs):
#            if pbc.DEBUG:
#                return g(*args, **kwargs)
#            else:
#                return f(*args, **kwargs)
#        return maybe_cache
#except:
#    memory_cache = lambda f: f

def eval_ao(cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    '''Collocate AO crystal orbitals (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Cell`

        coords : (nx*ny*nz, 3) ndarray
            The real-space grid point coordinates.

    Kwargs:
        kpt : (3,) ndarray
            The k-point corresponding to the crystal AO.
        deriv : int
            AO derivative order.  It affects the shape of the return array.
            If deriv=0, the returned AO values are stored in a (N,nao) array.
            Otherwise the AO values are stored in an array of shape (M,N,nao).
            Here N is the number of grids, nao is the number of AO functions,
            M is the size associated to the derivative deriv.

    Returns:
        aoR : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If deriv=1, also contains the value of the orbitals gradient in the
            x, y, and z directions.  It can be either complex or float array,
            depending on the kpt argument.  If kpt is not given (gamma point),
            aoR is a float array.

    See Also:
        pyscf.dft.numint.eval_ao

    '''
    ao_kpts = eval_ao_kpts(cell, coords, numpy.reshape(kpt, (-1,3)), deriv,
                           relativity, shls_slice, non0tab, out, verbose)
    return ao_kpts[0]


#@memory_cache
def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shls_slice=None, non0tab=None, out=None, verbose=None, **kwargs):
    '''
    Returns:
        ao_kpts: (nkpts, [comp], ngrids, nao) ndarray
            AO values at each k-point
    '''
    if kpts is None:
        if 'kpt' in kwargs:
            sys.stderr.write('WARN: KNumInt.eval_ao function finds keyword '
                             'argument "kpt" and converts it to "kpts"\n')
            kpts = kwargs['kpt']
        else:
            kpts = numpy.zeros((1,3))
    kpts = numpy.reshape(kpts, (-1,3))

    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    if cell.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    return cell.pbc_eval_gto(feval, coords, comp, kpts,
                             shls_slice=shls_slice, non0tab=non0tab, out=out)


def eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Mole` or :class:`Cell`

        ao : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If xctype='GGA', also contains the value of the gradient in the x, y,
            and z directions.

    Returns:
        rho : ([4,] nx*ny*nz) ndarray
            The value of the density on the real-space grid. If xctype='GGA',
            also contains the value of the gradient in the x, y, and z
            directions.

    See Also:
        pyscf.dft.numint.eval_rho

    '''

    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE, cell.nbas),
                              dtype=numpy.uint8)
        non0tab[:] = 0xff

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(dm):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        dm = dm.astype(numpy.complex128)
# For GGA, function eval_rho returns   real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + conj(|\nabla j> conj(D_ij) < i|))
#       = real(|\nabla i> D_ij <j|) + real(|\nabla j> conj(D_ij) < i|)
#       = real(|\nabla i> [D_ij + (D^\dagger)_ij] <j|)
# symmetrization dm (D + D.conj().T) then /2 because the code below computes
#       2*real(|\nabla i> D_ij <j|)
        if not hermi:
            dm = (dm + dm.conj().T) * .5

        def dot_bra(bra, aodm):
            # rho = numpy.einsum('pi,pi->p', bra.conj(), aodm).real
            #:rho  = numpy.einsum('pi,pi->p', bra.real, aodm.real)
            #:rho += numpy.einsum('pi,pi->p', bra.imag, aodm.imag)
            #:return rho
            return _contract_rho(bra, aodm)

        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(cell, ao, dm, non0tab, shls_slice, ao_loc)
            rho = dot_bra(ao, c0)

        elif xctype == 'GGA':
            rho = numpy.empty((4,ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            rho[0] = dot_bra(ao[0], c0)
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0) * 2

        else:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            rho[0] = dot_bra(ao[0], c0)
            rho[5] = 0
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0) * 2  # *2 for +c.c.
                c1 = _dot_ao_dm(cell, ao[i], dm, non0tab, shls_slice, ao_loc)
                rho[5] += dot_bra(ao[i], c1)
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            rho[4] = dot_bra(ao2, c0)
            rho[4] += rho[5]
            rho[4] *= 2 # *2 for +c.c.
            rho[5] *= .5
    else:
        # real orbitals and real DM
        rho = numint.eval_rho(cell, ao, dm, non0tab, xctype, hermi, verbose)
    return rho

def eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              verbose=None):
    '''Refer to `pyscf.dft.numint.eval_rho2` for full documentation.
    '''
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                             dtype=numpy.uint8)
        non0tab[:] = 0xff

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(mo_coeff):
        def dot(bra, ket):
            #:rho  = numpy.einsum('pi,pi->p', bra.real, ket.real)
            #:rho += numpy.einsum('pi,pi->p', bra.imag, ket.imag)
            #:return rho
            return _contract_rho(bra, ket)

        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        pos = mo_occ > OCCDROP
        cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))

        if pos.sum() > 0:
            if xctype == 'LDA' or xctype == 'HF':
                c0 = _dot_ao_dm(cell, ao, cpos, non0tab, shls_slice, ao_loc)
                rho = dot(c0, c0)
            elif xctype == 'GGA':
                rho = numpy.empty((4,ngrids))
                c0 = _dot_ao_dm(cell, ao[0], cpos, non0tab, shls_slice, ao_loc)
                rho[0] = dot(c0, c0)
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cpos, non0tab, shls_slice, ao_loc)
                    rho[i] = dot(c0, c1) * 2  # *2 for +c.c.
            else: # meta-GGA
                # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
                rho = numpy.empty((6,ngrids))
                c0 = _dot_ao_dm(cell, ao[0], cpos, non0tab, shls_slice, ao_loc)
                rho[0] = dot(c0, c0)
                rho[5] = 0
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cpos, non0tab, shls_slice, ao_loc)
                    rho[i] = dot(c0, c1) * 2  # *2 for +c.c.
                    rho[5]+= dot(c1, c1)
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                c1 = _dot_ao_dm(cell, ao2, cpos, non0tab, shls_slice, ao_loc)
                rho[4] = dot(c0, c1)
                rho[4]+= rho[5]
                rho[4]*= 2
                rho[5]*= .5
        else:
            if xctype == 'LDA' or xctype == 'HF':
                rho = numpy.zeros(ngrids)
            elif xctype == 'GGA':
                rho = numpy.zeros((4,ngrids))
            else:
                rho = numpy.zeros((6,ngrids))

        neg = mo_occ < -OCCDROP
        if neg.sum() > 0:
            cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
            if xctype == 'LDA' or xctype == 'HF':
                c0 = _dot_ao_dm(cell, ao, cneg, non0tab, shls_slice, ao_loc)
                rho -= dot(c0, c0)
            elif xctype == 'GGA':
                c0 = _dot_ao_dm(cell, ao[0], cneg, non0tab, shls_slice, ao_loc)
                rho[0] -= dot(c0, c0)
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cneg, non0tab, shls_slice, ao_loc)
                    rho[i] -= dot(c0, c1) * 2  # *2 for +c.c.
            else:
                c0 = _dot_ao_dm(cell, ao[0], cneg, non0tab, shls_slice, ao_loc)
                rho[0] -= dot(c0, c0)
                rho5 = 0
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cneg, non0tab, shls_slice, ao_loc)
                    rho[i] -= dot(c0, c1) * 2  # *2 for +c.c.
                    rho5 -= dot(c1, c1)
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                c1 = _dot_ao_dm(cell, ao2, cneg, non0tab, shls_slice, ao_loc)
                rho[4] -= dot(c0, c1) * 2
                rho[4] -= rho5 * 2
                rho[5] -= rho5 * .5
    else:
        rho = numint.eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab, xctype, verbose)
    return rho


def nr_rks(ni, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.
    Faster function uses eval_rho2 which is not yet implemented.

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D/3D array or a list of 2D/3D arrays
            Density matrices (2D) / density matrices for k-points (3D)

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
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))

    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi)

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = [0]*nset
    if xctype == 'LDA':
        ao_deriv = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1)[:2]
                vrho = vxc[0]
                den = rho*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1)[:2]
                den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1)[:2]
                den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, numpy.asarray(vmat)

def nr_uks(ni, cell, grids, xc_code, dms, spin=1, relativity=0, hermi=0,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.
    Faster function uses eval_rho2 which is not yet implemented.

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms :
            Density matrices

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
            kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))

    xctype = ni._xc_type(xc_code)
    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmb, hermi)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmata = [0]*nset
    vmatb = [0]*nset
    if xctype == 'LDA':
        ao_deriv = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b * weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                vmata[i] += ni.eval_mat(cell, ao_k1, weight, rho_a, vrho[:,0],
                                        mask, xctype, 1, verbose)
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, rho_b, vrho[:,1],
                                        mask, xctype, 1, verbose)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts,
                                 kpts_band, max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose=verbose)[:2]
                vrho, vsigma = vxc[:2]
                den = rho_a[0]*weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b[0]*weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                vmata[i] += ni.eval_mat(cell, ao_k1, weight, (rho_a,rho_b),
                                        (vrho[:,0], (vsigma[:,0],vsigma[:,1])),
                                        mask, xctype, 1, verbose)
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, (rho_b,rho_a),
                                        (vrho[:,1], (vsigma[:,2],vsigma[:,1])),
                                        mask, xctype, 1, verbose)
    elif xctype == 'MGGA':
        assert(all(x not in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')))
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose=verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc
                den = rho_a[0]*weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b[0]*weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                v = (vrho[:,0], (vsigma[:,0],vsigma[:,1]), None, vtau[:,0])
                vmata[i] += ni.eval_mat(cell, ao_k1, weight, (rho_a,rho_b), v,
                                        mask, xctype, 1, verbose)
                v = (vrho[:,1], (vsigma[:,2],vsigma[:,1]), None, vtau[:,1])
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, (rho_b,rho_a), v,
                                        mask, xctype, 1, verbose)
                v = None

    if dma.ndim == vmata[0].ndim:  # One set of DMs in the input
        nelec = nelec[:,0]
        excsum = excsum[0]
        vmata = vmata[0]
        vmatb = vmatb[0]
    return nelec, excsum, numpy.asarray((vmata,vmatb))

def _format_uks_dm(dms):
    dma, dmb = dms
    if getattr(dms, 'mo_coeff', None) is not None:
#TODO: test whether dm.mo_coeff matching dm
        mo_coeff = dms.mo_coeff
        mo_occ = dms.mo_occ
        if (isinstance(mo_coeff[0], numpy.ndarray) and
            mo_coeff[0].ndim < dma.ndim): # handle ROKS
            mo_occa = [numpy.array(occ> 0, dtype=numpy.double) for occ in mo_occ]
            mo_occb = [numpy.array(occ==2, dtype=numpy.double) for occ in mo_occ]
            dma = lib.tag_array(dma, mo_coeff=mo_coeff, mo_occ=mo_occa)
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff, mo_occ=mo_occb)
        else:
            dma = lib.tag_array(dma, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
    return dma, dmb

nr_rks_vxc = nr_rks
nr_uks_vxc = nr_uks

def nr_rks_fxc(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
               verbose=None):
    '''Contract RKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D/3D array or a list of 2D/3D arrays
            Density matrices (2D) / density matrices for k-points (3D)

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

    Examples:

    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)

    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi)
    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        make_rho0 = ni._gen_rho_evaluator(cell, dm0, 1)[0]

    ao_loc = cell.ao_loc_nr()
    vmat = [0] * nset
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
                fxc0 = ni.eval_xc(xc_code, rho, 0, relativity, 2,
                                  verbose=verbose)[2]
                frr = fxc0[0]
            else:
                frr = fxc[0][ip:ip+ngrid]
                ip += ngrid

            for i in range(nset):
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = weight * frr * rho1
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if rho0 is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
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
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = _rks_gga_wv1(rho, rho1, vxc0, fxc0, weight)
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

        # call swapaxes method to swap last two indices because vmat may be a 3D
        # array (nset,nao,nao) in single k-point mode or a 4D array
        # (nset,nkpts,nao,nao) in k-points mode
        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmat[i] = vmat[i] + vmat[i].swapaxes(-2,-1).conj()

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if isinstance(dms, numpy.ndarray) and dms.ndim == vmat[0].ndim:
        # One set of DMs in the input
        vmat = vmat[0]
    return numpy.asarray(vmat)

def nr_rks_fxc_st(ni, cell, grids, xc_code, dm0, dms_alpha, relativity=0, singlet=True,
                  rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
                  verbose=None):
    '''Associated to singlet or triplet Hessian
    Note the difference to nr_rks_fxc, dms_alpha is the response density
    matrices of alpha spin, alpha+/-beta DM is applied due to singlet/triplet
    coupling

    Ref. CPL, 256, 454
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)

    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms_alpha)
    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        make_rho0 = ni._gen_rho_evaluator(cell, dm0, 1)[0]

    ao_loc = cell.ao_loc_nr()
    vmat = [0] * nset
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
                rho *= .5  # alpha density
                fxc0 = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                ip += ngrid
            if singlet:
                frho = u_u + u_d
            else:
                frho = u_u - u_d

            for i in range(nset):
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = weight * frho * rho1
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if vxc is None or fxc is None:
                rho = make_rho0(0, ao_k1, mask, xctype)
                rho *= .5  # alpha density
                vxc0, fxc0 = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[1:3]

                vsigma = vxc0[1].T
                u_u, u_d, d_d = fxc0[0].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc0[1].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc0[2].T  # v2sigma2
            else:
                rho = rho0[0][:,ip:ip+ngrid]
                vsigma = vxc[1][ip:ip+ngrid].T
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1][ip:ip+ngrid].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2][ip:ip+ngrid].T  # v2sigma2

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
                rho1 = make_rho(i, ao_k1, mask, xctype)
                wv = _rks_gga_wv1(rho, rho1, (None,fgamma),
                                  (frho,frhogamma,fgg), weight)
                vmat[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmat[i] = vmat[i] + vmat[i].swapaxes(-2,-1).conj()

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if isinstance(dms_alpha, numpy.ndarray) and dms_alpha.ndim == vmat[0].ndim:
        vmat = vmat[0]
    return numpy.asarray(vmat)


def nr_uks_fxc(ni, cell, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
               verbose=None):
    '''Contract UKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt` or :class:`KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

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
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmb, hermi)[0]

    if ((xctype == 'LDA' and fxc is None) or
        (xctype == 'GGA' and rho0 is None)):
        dm0a, dm0b = _format_uks_dm(dm0)
        make_rho0a = ni._gen_rho_evaluator(cell, dm0a, 1)[0]
        make_rho0b = ni._gen_rho_evaluator(cell, dm0b, 1)[0]

    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc_nr()

    vmata = [0] * nset
    vmatb = [0] * nset
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho0a = make_rho0a(0, ao_k1, mask, xctype)
                rho0b = make_rho0b(0, ao_k1, mask, xctype)
                fxc0 = ni.eval_xc(xc_code, (rho0a,rho0b), 1, relativity, 2,
                                  verbose=verbose)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                ip += ngrid

            for i in range(nset):
                rho1a = make_rhoa(i, ao_k1, mask, xctype)
                rho1b = make_rhob(i, ao_k1, mask, xctype)
                wv = u_u * rho1a + u_d * rho1b
                wv *= weight
                vmata[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)
                wv = u_d * rho1a + d_d * rho1b
                wv *= weight
                vmatb[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if rho0 is None:
                rho0a = make_rho0a(0, ao_k1, mask, xctype)
                rho0b = make_rho0b(0, ao_k1, mask, xctype)
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
                rho1a = make_rhoa(i, ao_k1, mask, xctype)
                rho1b = make_rhob(i, ao_k1, mask, xctype)
                wva, wvb = _uks_gga_wv1((rho0a,rho0b), (rho1a,rho1b),
                                        vxc0, fxc0, weight)
                vmata[i] += ni._fxc_mat(cell, ao_k1, wva, mask, xctype, ao_loc)
                vmatb[i] += ni._fxc_mat(cell, ao_k1, wvb, mask, xctype, ao_loc)

        for i in range(nset):  # for (\nabla\mu) \nu + \mu (\nabla\nu)
            vmata[i] = vmata[i] + vmata[i].swapaxes(-1,-2).conj()
            vmatb[i] = vmatb[i] + vmatb[i].swapaxes(-1,-2).conj()
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    if dma.ndim == vmata[0].ndim:  # One set of DMs in the input
        vmata = vmata[0]
        vmatb = vmatb[0]
    return numpy.asarray((vmata,vmatb))

def _fxc_mat(cell, ao, wv, non0tab, xctype, ao_loc):
    shls_slice = (0, cell.nbas)

    if xctype == 'LDA' or xctype == 'HF':
        #:aow = numpy.einsum('pi,p->pi', ao, wv)
        aow = _scale_ao(ao, wv)
        mat = _dot_ao_ao(cell, ao, aow, non0tab, shls_slice, ao_loc)
    else:
        #:aow = numpy.einsum('npi,np->pi', ao, wv)
        aow = _scale_ao(ao, wv)
        mat = _dot_ao_ao(cell, ao[0], aow, non0tab, shls_slice, ao_loc)
    return mat

def cache_xc_kernel(ni, cell, grids, xc_code, mo_coeff, mo_occ, spin=0,
                    kpts=None, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)
    ao_deriv = 0
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    nao = cell.nao_nr()
    if spin == 0:
        rho = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rho.append(ni.eval_rho2(cell, ao_k1, mo_coeff, mo_occ, mask, xctype))
        rho = numpy.hstack(rho)
    else:
        rhoa = []
        rhob = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rhoa.append(ni.eval_rho2(cell, ao_k1, mo_coeff[0], mo_occ[0], mask, xctype))
            rhob.append(ni.eval_rho2(cell, ao_k1, mo_coeff[1], mo_occ[1], mask, xctype))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc(xc_code, rho, spin, 0, 2, 0)[1:3]
    return rho, vxc, fxc


def get_rho(ni, cell, dm, grids, kpts=numpy.zeros((1,3)), max_memory=2000):
    '''Density in real space
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm)
    assert(nset == 1)
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao_k1, ao_k2, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, 0, kpts, None, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao_k1, mask, 'LDA')
    return rho


class NumInt(numint.NumInt):
    '''Generalization of pyscf's NumInt class for a single k-point shift and
    periodic images.
    '''
    def eval_ao(self, cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0,
                shls_slice=None, non0tab=None, out=None, verbose=None):
        return eval_ao(cell, coords, kpt, deriv, relativity, shls_slice,
                       non0tab, out, verbose)

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, cell, coords, relativity=0, shls_slice=None,
                  verbose=None):
        return make_mask(cell, coords, relativity, shls_slice, verbose)

    @lib.with_doc(eval_rho.__doc__)
    def eval_rho(self, cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        return eval_rho(cell, ao, dm, non0tab, xctype, hermi, verbose)

    def eval_rho2(self, cell, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        return eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab, xctype, verbose)

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix.
        See :func:`nr_rks` and :func:`nr_uks` for more details.
        '''
        if spin == 0:
            return self.nr_rks(cell, grids, xc_code, dms, hermi,
                               kpt, kpts_band, max_memory, verbose)
        else:
            return self.nr_uks(cell, grids, xc_code, dms, hermi,
                               kpt, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, hermi=0,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
# To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = KNumInt()
            ni.__dict__.update(self.__dict__)
            nao = dms.shape[-1]
            return ni.nr_rks(cell, grids, xc_code, dms.reshape(-1,1,nao,nao),
                             hermi, kpt.reshape(1,3), kpts_band, max_memory,
                             verbose)
        return nr_rks(self, cell, grids, xc_code, dms,
                      0, 0, hermi, kpt, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, hermi=0,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
# To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = KNumInt()
            ni.__dict__.update(self.__dict__)
            nao = dms[0].shape[-1]
            return ni.nr_uks(cell, grids, xc_code, dms.reshape(-1,1,nao,nao),
                             hermi, kpt.reshape(1,3), kpts_band, max_memory,
                             verbose)
        return nr_uks(self, cell, grids, xc_code, dms,
                      1, 0, hermi, kpt, kpts_band, max_memory, verbose)

    def eval_mat(self, cell, ao, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
# Guess whether ao is evaluated for kpts_band.  When xctype is LDA, ao on grids
# should be a 2D array.  For other xc functional, ao should be a 3D array.
        if ao.ndim == 2 or (xctype != 'LDA' and ao.ndim == 3):
            mat = eval_mat(cell, ao, weight, rho, vxc, non0tab, xctype, spin, verbose)
        else:
            nkpts = len(ao)
            nao = ao[0].shape[-1]
            mat = numpy.empty((nkpts,nao,nao), dtype=numpy.complex128)
            for k in range(nkpts):
                mat[k] = eval_mat(cell, ao[k], weight, rho, vxc,
                                  non0tab, xctype, spin, verbose)
        return mat

    def _fxc_mat(self, cell, ao, wv, non0tab, xctype, ao_loc):
        return _fxc_mat(cell, ao, wv, non0tab, xctype, ao_loc)

    def block_loop(self, cell, grids, nao, deriv=0, kpt=numpy.zeros(3),
                   kpts_band=None, max_memory=2000, non0tab=None, blksize=None):
        '''Define this macro to loop over grids by blocks.
        '''
# For UniformGrids, grids.coords does not indicate whehter grids are initialized
        if grids.non0tab is None:
            grids.build(with_non0tab=True)
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nao*16*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*1200))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = 0xff
        kpt = numpy.reshape(kpt, 3)
        if kpts_band is None:
            kpt1 = kpt2 = kpt
        else:
            kpt1 = kpts_band
            kpt2 = kpt

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao_k2 = self.eval_ao(cell, coords, kpt2, deriv=deriv, non0tab=non0)
            if abs(kpt1-kpt2).sum() < 1e-9:
                ao_k1 = ao_k2
            else:
                ao_k1 = self.eval_ao(cell, coords, kpt1, deriv=deriv)
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

    def _gen_rho_evaluator(self, cell, dms, hermi=0):
        return numint.NumInt._gen_rho_evaluator(self, cell, dms, hermi)

    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    cache_xc_kernel  = cache_xc_kernel
    get_rho = get_rho

_NumInt = NumInt


class KNumInt(numint.NumInt):
    '''Generalization of pyscf's NumInt class for k-point sampling and
    periodic images.
    '''
    def __init__(self, kpts=numpy.zeros((1,3))):
        numint.NumInt.__init__(self)
        self.kpts = numpy.reshape(kpts, (-1,3))

    def eval_ao(self, cell, coords, kpts=numpy.zeros((1,3)), deriv=0, relativity=0,
                shls_slice=None, non0tab=None, out=None, verbose=None, **kwargs):
        return eval_ao_kpts(cell, coords, kpts, deriv,
                            relativity, shls_slice, non0tab, out, verbose)

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, cell, coords, relativity=0, shls_slice=None,
                  verbose=None):
        return make_mask(cell, coords, relativity, shls_slice, verbose)

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 hermi=0, verbose=None):
        '''Collocate the *real* density (opt. gradients) on the real-space grid.

        Args:
            cell : Mole or Cell object
            ao_kpts : (nkpts, ngrids, nao) ndarray
                AO values at each k-point
            dm_kpts: (nkpts, nao, nao) ndarray
                Density matrix at each k-point

        Returns:
           rhoR : (ngrids,) ndarray
        '''
        nkpts = len(ao_kpts)
        rhoR = 0
        for k in range(nkpts):
            rhoR += eval_rho(cell, ao_kpts[k], dm_kpts[k], non0tab, xctype,
                             hermi, verbose)
        rhoR *= 1./nkpts
        return rhoR

    def eval_rho2(self, cell, ao_kpts, mo_coeff_kpts, mo_occ_kpts,
                  non0tab=None, xctype='LDA', verbose=None):
        nkpts = len(ao_kpts)
        rhoR = 0
        for k in range(nkpts):
            rhoR += eval_rho2(cell, ao_kpts[k], mo_coeff_kpts[k],
                              mo_occ_kpts[k], non0tab, xctype, verbose)
        rhoR *= 1./nkpts
        return rhoR

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix.
        See :func:`nr_rks` and :func:`nr_uks` for more details.
        '''
        if spin == 0:
            return self.nr_rks(cell, grids, xc_code, dms, hermi,
                               kpts, kpts_band, max_memory, verbose)
        else:
            return self.nr_uks(cell, grids, xc_code, dms, hermi,
                               kpts, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, hermi=0, kpts=None, kpts_band=None,
               max_memory=2000, verbose=None, **kwargs):
        if kpts is None:
            if 'kpt' in kwargs:
                sys.stderr.write('WARN: KNumInt.nr_rks function finds keyword '
                                 'argument "kpt" and converts it to "kpts"\n')
                kpts = kwargs['kpt']
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1,3)

        return nr_rks(self, cell, grids, xc_code, dms, 0, 0,
                      hermi, kpts, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, hermi=0, kpts=None, kpts_band=None,
               max_memory=2000, verbose=None, **kwargs):
        if kpts is None:
            if 'kpt' in kwargs:
                sys.stderr.write('WARN: KNumInt.nr_uks function finds keyword '
                                 'argument "kpt" and converts it to "kpts"\n')
                kpts = kwargs['kpt']
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1,3)

        return nr_uks(self, cell, grids, xc_code, dms, 1, 0,
                      hermi, kpts, kpts_band, max_memory, verbose)

    def eval_mat(self, cell, ao_kpts, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        nkpts = len(ao_kpts)
        nao = ao_kpts[0].shape[-1]
        dtype = numpy.result_type(*ao_kpts)
        mat = numpy.empty((nkpts,nao,nao), dtype=dtype)
        for k in range(nkpts):
            mat[k] = eval_mat(cell, ao_kpts[k], weight, rho, vxc,
                              non0tab, xctype, spin, verbose)
        return mat

    def _fxc_mat(self, cell, ao_kpts, wv, non0tab, xctype, ao_loc):
        nkpts = len(ao_kpts)
        nao = ao_kpts[0].shape[-1]
        dtype = numpy.result_type(*ao_kpts)
        mat = numpy.empty((nkpts,nao,nao), dtype=dtype)
        for k in range(nkpts):
            mat[k] = _fxc_mat(cell, ao_kpts[k], wv, non0tab, xctype, ao_loc)
        return mat

    def block_loop(self, cell, grids, nao, deriv=0, kpts=numpy.zeros((1,3)),
                   kpts_band=None, max_memory=2000, non0tab=None, blksize=None):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        nkpts = len(kpts)
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nkpts*nao*16*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*1200))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = 0xff
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            where = [member(k, kpts) for k in kpts_band]
            where = [k_id[0] if len(k_id)>0 else None for k_id in where]

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao_k1 = ao_k2 = self.eval_ao(cell, coords, kpts, deriv=deriv, non0tab=non0)
            if kpts_band is not None:
                ao_k1 = self.eval_ao(cell, coords, kpts_band, deriv=deriv, non0tab=non0)
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

    def _gen_rho_evaluator(self, cell, dms, hermi=0):
        if getattr(dms, 'mo_coeff', None) is not None:
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
            nao = cell.nao_nr()
            ndms = len(mo_occ)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(cell, ao, mo_coeff[idm], mo_occ[idm],
                                      non0tab, xctype)
        else:
            if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
                dms = [numpy.stack(dms)]
            #if not hermi:
            # Density (or response of density) is always real for DFT.
            # Symmetrizing DM for gamma point should not change the value of
            # density. However, when k-point is considered, unless dm and
            # dm.conj().transpose produce the same real part of density, the
            # symmetrization code below may be incorrect (proof is needed).
            #    # dm.shape = (nkpts, nao, nao)
            #    dms = [(dm+dm.conj().transpose(0,2,1))*.5 for dm in dms]
            nao = dms[0].shape[-1]
            ndms = len(dms)
            def make_rho(idm, ao_kpts, non0tab, xctype):
                return self.eval_rho(cell, ao_kpts, dms[idm], non0tab, xctype,
                                     hermi=hermi)
        return make_rho, ndms, nao

    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    cache_xc_kernel  = cache_xc_kernel
    get_rho = get_rho

_KNumInt = KNumInt
