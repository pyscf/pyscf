#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.numint import eval_mat, _dot_ao_ao, _dot_ao_dm, _tau_dot
from pyscf.dft.numint import _scale_ao, _contract_rho
from pyscf.dft.numint import OCCDROP
from pyscf.dft.gen_grid import NBINS, CUTOFF, ALIGNMENT_UNIT
from pyscf.pbc.dft.gen_grid import make_mask, BLKSIZE
from pyscf.pbc.lib.kpts_helper import is_zero, KPT_DIFF_TOL
from pyscf.pbc.lib.kpts import KPoints


def eval_ao(cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0, shls_slice=None,
            non0tab=None, cutoff=None, out=None, verbose=None):
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
                           relativity, shls_slice, non0tab, cutoff, out, verbose)
    return ao_kpts[0]


def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shls_slice=None, non0tab=None, cutoff=None, out=None,
                 verbose=None, **kwargs):
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
    return cell.pbc_eval_gto(feval, coords, comp, kpts, shls_slice=shls_slice,
                             non0tab=non0tab, cutoff=cutoff, out=out)


def eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, with_lapl=True,
             verbose=None):
    '''Collocate the density (opt. gradients) on the real-space grid.

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
    ngrids, nao = ao.shape[-2:]

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(dm):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        dm = dm.astype(numpy.complex128)

        if hermi == 1:
            dot_bra = _contract_rho
            dtype = numpy.float64
        else:
            def dot_bra(bra, aodm):
                return numpy.einsum('pi,pi->p', bra.conj(), aodm)
            dtype = numpy.complex128

        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(cell, ao, dm, non0tab, shls_slice, ao_loc)
            rho = dot_bra(ao, c0)

        elif xctype == 'GGA':
            rho = numpy.empty((4,ngrids), dtype=dtype)
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            rho[0] = dot_bra(ao[0], c0)
            for i in range(1, 4):
                rho[i] = dot_bra(ao[i], c0)
            if hermi == 1:
                rho[1:4] *= 2
            else:
                c1 = _dot_ao_dm(cell, ao[0], dm.conj().T, non0tab, shls_slice, ao_loc)
                for i in range(1, 4):
                    rho[i] += dot_bra(c1, ao[i])

        else:
            if with_lapl:
                # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
                rho = numpy.empty((6,ngrids), dtype=dtype)
                tau_idx = 5
            else:
                rho = numpy.empty((5,ngrids), dtype=dtype)
                tau_idx = 4
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            rho[0] = dot_bra(ao[0], c0)
            rho[tau_idx] = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(cell, ao[i], dm, non0tab, shls_slice, ao_loc)
                rho[tau_idx] += dot_bra(ao[i], c1)
                rho[i] = dot_bra(ao[i], c0)
                if hermi == 1:
                    rho[i] *= 2
                else:
                    rho[i] += dot_bra(ao[0], c1)
            if with_lapl:
                if ao.shape[0] > 4:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    rho[4] = dot_bra(ao2, c0)
                    rho[4] += rho[5]
                    if hermi == 1:
                        rho[4] *= 2
                    else:
                        c2 = _dot_ao_dm(cell, ao2, dm, non0tab, shls_slice, ao_loc)
                        rho[4] += dot_bra(ao[0], c2)
                        rho[4] += rho[5]
                else:
                    raise ValueError('Not enough derivatives in ao')
            rho[tau_idx] *= .5
    else:
        # real orbitals and real DM
        rho = numint.eval_rho(cell, ao, dm, non0tab, xctype, hermi, with_lapl, verbose)
    return rho

def eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              with_lapl=True, verbose=None):
    '''Refer to `pyscf.dft.numint.eval_rho2` for full documentation.
    '''
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(mo_coeff):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        pos = mo_occ > OCCDROP
        cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))

        if pos.sum() > 0:
            if xctype == 'LDA' or xctype == 'HF':
                c0 = _dot_ao_dm(cell, ao, cpos, non0tab, shls_slice, ao_loc)
                rho = _contract_rho(c0, c0)
            elif xctype == 'GGA':
                rho = numpy.empty((4,ngrids))
                c0 = _dot_ao_dm(cell, ao[0], cpos, non0tab, shls_slice, ao_loc)
                rho[0] = _contract_rho(c0, c0)
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cpos, non0tab, shls_slice, ao_loc)
                    rho[i] = _contract_rho(c0, c1) * 2  # *2 for +c.c.
            else: # meta-GGA
                if with_lapl:
                    # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
                    rho = numpy.empty((6,ngrids))
                    tau_idx = 5
                else:
                    rho = numpy.empty((5,ngrids))
                    tau_idx = 4
                c0 = _dot_ao_dm(cell, ao[0], cpos, non0tab, shls_slice, ao_loc)
                rho[0] = _contract_rho(c0, c0)
                rho[tau_idx] = 0
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cpos, non0tab, shls_slice, ao_loc)
                    rho[i] = _contract_rho(c0, c1) * 2  # *2 for +c.c.
                    rho[tau_idx]+= _contract_rho(c1, c1)
                if with_lapl:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    c1 = _dot_ao_dm(cell, ao2, cpos, non0tab, shls_slice, ao_loc)
                    rho[4] = _contract_rho(c0, c1)
                    rho[4]+= rho[5]
                    rho[4]*= 2
                rho[tau_idx] *= .5
        else:
            if xctype == 'LDA' or xctype == 'HF':
                rho = numpy.zeros(ngrids)
            elif xctype == 'GGA':
                rho = numpy.zeros((4,ngrids))
            elif with_lapl:
                # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
                rho = numpy.zeros((6,ngrids))
                tau_idx = 5
            else:
                rho = numpy.zeros((5,ngrids))
                tau_idx = 4

        neg = mo_occ < -OCCDROP
        if neg.sum() > 0:
            cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
            if xctype == 'LDA' or xctype == 'HF':
                c0 = _dot_ao_dm(cell, ao, cneg, non0tab, shls_slice, ao_loc)
                rho -= _contract_rho(c0, c0)
            elif xctype == 'GGA':
                c0 = _dot_ao_dm(cell, ao[0], cneg, non0tab, shls_slice, ao_loc)
                rho[0] -= _contract_rho(c0, c0)
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cneg, non0tab, shls_slice, ao_loc)
                    rho[i] -= _contract_rho(c0, c1) * 2  # *2 for +c.c.
            else:
                c0 = _dot_ao_dm(cell, ao[0], cneg, non0tab, shls_slice, ao_loc)
                rho[0] -= _contract_rho(c0, c0)
                rho5 = 0
                for i in range(1, 4):
                    c1 = _dot_ao_dm(cell, ao[i], cneg, non0tab, shls_slice, ao_loc)
                    rho[i] -= _contract_rho(c0, c1) * 2  # *2 for +c.c.
                    rho5 -= _contract_rho(c1, c1)
                if with_lapl:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    c1 = _dot_ao_dm(cell, ao2, cneg, non0tab, shls_slice, ao_loc)
                    rho[4] -= _contract_rho(c0, c1) * 2
                    rho[4] -= rho5 * 2
                rho[tau_idx] -= rho5 * .5
    else:
        rho = numint.eval_rho2(cell, ao, mo_coeff, mo_occ, non0tab, xctype,
                               with_lapl, verbose)
    return rho


def nr_rks(ni, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.

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
            Whether the input density matrix is hermitian
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 1
    elif xctype == 'HF':
        ao_deriv = 0
    else:
        raise NotImplementedError(f'nr_rks for functional {xc_code}')

    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi, False)

    if xctype in ('LDA', 'GGA', 'MGGA'):
        nelec = numpy.zeros(nset)
        excsum = numpy.zeros(nset)
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc
        deriv = 1
        vmat = [0]*nset
        v_hermi = 1  # the output matrix must be hermitian
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype).real
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[:2]
                if xctype == 'LDA':
                    den = rho*weight
                else:
                    den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += den.dot(exc)
                wv = weight * vxc
                vmat[i] += ni._vxc_mat(cell, ao_k1, wv, mask, xctype,
                                       shls_slice, ao_loc, v_hermi)

        vmat = numpy.stack(vmat)
        # call swapaxes method to swap last two indices because vmat may be a 3D
        # array (nset,nao,nao) in single k-point mode or a 4D array
        # (nset,nkpts,nao,nao) in k-points mode
        vmat = vmat + vmat.conj().swapaxes(-2,-1)
        if nset == 1:
            nelec = nelec[0]
            excsum = excsum[0]
            vmat = vmat[0]
    else:
        nelec = excsum = vmat = 0
    return nelec, excsum, vmat

def nr_uks(ni, cell, grids, xc_code, dms, spin=1, relativity=0, hermi=1,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.

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
            Whether the input density matrix is hermitian
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
            kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts

    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 1
    elif xctype == 'HF':
        ao_deriv = 0
    else:
        raise NotImplementedError(f'nr_uks for functional {xc_code}')

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi, False)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmb, hermi, False)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    if xctype in ('LDA', 'GGA', 'MGGA'):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc
        deriv = 1
        vmata = [0]*nset
        vmatb = [0]*nset
        v_hermi = 1  # the output matrix must be hermitian
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype).real
                rho_b = make_rhob(i, ao_k2, mask, xctype).real
                rho = (rho_a, rho_b)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv, xctype=xctype)[:2]
                if xctype == 'LDA':
                    dena = rho_a * weight
                    denb = rho_b * weight
                else:
                    dena = rho_a[0] * weight
                    denb = rho_b[0] * weight
                nelec[0,i] += dena.sum()
                nelec[1,i] += denb.sum()
                excsum[i] += dena.dot(exc)
                excsum[i] += denb.dot(exc)
                wv = weight * vxc
                vmata[i] += ni._vxc_mat(cell, ao_k1, wv[0], mask, xctype,
                                        shls_slice, ao_loc, v_hermi)
                vmatb[i] += ni._vxc_mat(cell, ao_k1, wv[1], mask, xctype,
                                        shls_slice, ao_loc, v_hermi)

        vmat = numpy.stack([vmata, vmatb])
        # call swapaxes method to swap last two indices because vmat may be a 3D
        # array (nset,nao,nao) in single k-point mode or a 4D array
        # (nset,nkpts,nao,nao) in k-points mode
        vmat = vmat + vmat.conj().swapaxes(-2,-1)
        if nset == 1:
            nelec = nelec[:,0]
            excsum = excsum[0]
            vmat = vmat[:,0]
    else:
        nelec = excsum = vmat = 0
    return nelec, excsum, vmat

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

def nr_nlc_vxc(ni, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    '''Calculate NLC functional and potential matrix on given grids

    Args:
        ni : an instance of :class:`NumInt`

        cell : an instance of :class:`Cell`

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
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi, False)
    assert nset == 1
    xctype = 'GGA'
    ao_deriv = 1
    vvrho = []
    for ao_k1, _, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
        vvrho.append(make_rho(0, ao_k1, mask, xctype))
    rho = numpy.hstack(vvrho)

    exc = 0
    vxc = 0
    nlc_coefs = ni.nlc_coeff(xc_code)
    for nlc_pars, fac in nlc_coefs:
        e, v = numint._vv10nlc(rho, grids.coords, rho, grids.weights,
                               grids.coords, nlc_pars)
        exc += e * fac
        vxc += v * fac
    den = rho[0] * grids.weights
    nelec = den.sum()
    excsum = numpy.dot(den, exc)
    vv_vxc = numint.xc_deriv.transform_vxc(rho, vxc, xctype, spin=0)

    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc_nr()
    vmat = 0
    hermi = 1
    p1 = 0
    for ao_k1, _, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, ao_deriv, kpts_band, None, max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        vmat += ni._vxc_mat(cell, ao_k1, wv, mask, xctype,
                            shls_slice, ao_loc, hermi)
    vmat = vmat + vmat.conj().swapaxes(-2,-1)
    return nelec, excsum, vmat

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
            Whether the input density matrix is hermitian
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
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 1
    elif xctype == 'HF':
        ao_deriv = 0
    else:
        raise NotImplementedError(f'nr_rks_fxc for functional {xc_code}')

    if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
        spin = 0
        fxc = ni.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts,
                                  max_memory=max_memory)[2]
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts_ibz

    v_hermi = 0
    if is_zero(kpts):
        if isinstance(dms, numpy.ndarray) and dms.dtype == numpy.double:
            # for real orbitals and real matrix, K_{ia,bj} = K_{ia,jb}
            # The output matrix v = K*x_{ia} is symmetric
            v_hermi = 1

    if xctype in ('LDA', 'GGA', 'MGGA'):
        make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi, False)
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        vmat = [0] * nset
        p1 = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[:,:,p0:p1]
            for i in range(nset):
                rho1 = make_rho(i, ao_k1, mask, xctype)
                if xctype == 'LDA':
                    vxc1 = numpy.einsum('g,yg->yg', rho1, _fxc[0])
                else:
                    vxc1 = numpy.einsum('xg,xyg->yg', rho1, _fxc)
                wv = weight * vxc1
                vmat[i] += ni._vxc_mat(cell, ao_k1, wv, mask, xctype,
                                       shls_slice, ao_loc, v_hermi)

        vmat = numpy.stack(vmat)
        if v_hermi == 1:
            # call swapaxes method to swap last two indices because vmat may be a 3D
            # array (nset,nao,nao) in single k-point mode or a 4D array
            # (nset,nkpts,nao,nao) in k-points mode
            vmat = vmat + vmat.conj().swapaxes(-2,-1)
        if nset == 1:
            vmat = vmat.reshape(dms.shape)
    else:
        vmat = 0
    return vmat

def nr_rks_fxc_st(ni, cell, grids, xc_code, dm0, dms_alpha, relativity=0, singlet=True,
                  rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
                  verbose=None):
    '''Associated to singlet or triplet Hessian
    Note the difference to nr_rks_fxc, dms_alpha is the response density
    matrices of alpha spin, alpha+/-beta DM is applied due to singlet/triplet
    coupling

    Ref. CPL, 256, 454
    '''
    if fxc is None:
        spin = 1
        fxc = ni.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts,
                                  max_memory=max_memory)[2]
    if singlet:
        fxc = fxc[0,:,0] + fxc[0,:,1]
    else:
        fxc = fxc[0,:,0] - fxc[0,:,1]
    return ni.nr_rks_fxc(cell, grids, xc_code, dm0, dms_alpha, hermi=0, fxc=fxc,
                         kpts=kpts, max_memory=max_memory)

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
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        ao_deriv = 0
    elif xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 1
    elif xctype == 'HF':
        ao_deriv = 0
    else:
        raise NotImplementedError(f'nr_uks_fxc for functional {xc_code}')

    if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
        spin = 1
        fxc = ni.cache_xc_kernel1(cell, grids, xc_code, dm0, spin, kpts,
                                  max_memory=max_memory)[2]
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts_ibz

    dma, dmb = _format_uks_dm(dms)
    v_hermi = 0
    if is_zero(kpts) and dma.dtype == numpy.double:
        # for real orbitals and real matrix, K_{ia,bj} = K_{ia,jb}
        # The output matrix v = K*x_{ia} is symmetric
        v_hermi = 1

    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi, False)[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dmb, hermi, False)[0]

    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc_nr()
    vmata = [0] * nset
    vmatb = [0] * nset
    if xctype in ('LDA', 'GGA', 'MGGA'):
        p1 = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[:,:,:,:,p0:p1]

            for i in range(nset):
                rho1a = make_rhoa(i, ao_k1, mask, xctype)
                rho1b = make_rhob(i, ao_k1, mask, xctype)
                if xctype == 'LDA':
                    wv = rho1a * _fxc[0,0] + rho1b * _fxc[1,0]
                else:
                    wv  = numpy.einsum('xg,xbyg->byg', rho1a, _fxc[0])
                    wv += numpy.einsum('xg,xbyg->byg', rho1b, _fxc[1])
                wv *= weight
                vmata[i] += ni._vxc_mat(cell, ao_k1, wv[0], mask, xctype,
                                        shls_slice, ao_loc, v_hermi)
                vmatb[i] += ni._vxc_mat(cell, ao_k1, wv[1], mask, xctype,
                                        shls_slice, ao_loc, v_hermi)

        vmat = numpy.stack([vmata, vmatb])
        if v_hermi == 1:
            vmat = vmat + vmat.conj().swapaxes(-2,-1)
        if nset == 1:
            vmat = vmat.reshape((2,) + dma.shape)
    else:
        vmat = 0
    return vmat

def _vxc_mat(cell, ao, wv, mask, xctype, shls_slice, ao_loc, hermi):
    # NOTE ao might be complex. wv should be real for vxc_mat. wv can be complex
    # for fxc_mat
    if xctype == 'LDA':
        #:aow = numpy.einsum('pi,p->pi', ao, wv)
        aow = _scale_ao(ao, wv[0])
        mat = _dot_ao_ao(cell, ao, aow, mask, shls_slice, ao_loc)
    elif xctype == 'GGA':
        #:aow = numpy.einsum('npi,np->pi', ao, wv)
        aow = _scale_ao(ao, wv[:4])
        mat = _dot_ao_ao(cell, ao[0], aow, mask, shls_slice, ao_loc)
        if hermi != 1:
            aow = _scale_ao(ao[1:4], wv[1:4].conj())
            mat += _dot_ao_ao(cell, aow, ao[0], mask, shls_slice, ao_loc)
    elif xctype == 'MGGA':
        tau_idx = 4
        aow = _scale_ao(ao, wv[:4])
        mat = _dot_ao_ao(cell, ao[0], aow, mask, shls_slice, ao_loc)
        mat+= _tau_dot(cell, ao, ao, wv[tau_idx], mask, shls_slice, ao_loc)
        if hermi != 1:
            aow = _scale_ao(ao[1:4], wv[1:4].conj())
            mat += _dot_ao_ao(cell, aow, ao[0], mask, shls_slice, ao_loc)
    return mat

def cache_xc_kernel(ni, cell, grids, xc_code, mo_coeff, mo_occ, spin=0,
                    kpts=None, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        if kpts.kpts.size > 3: # multiple k points
            mo_coeff = kpts.transform_mo_coeff(mo_coeff)
            mo_occ = kpts.transform_mo_occ(mo_occ)
        kpts = kpts.kpts
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if numint.MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0

    if isinstance(ni, KNumInt):
        # mo_coeff of KRHF has [mo_k1, mo_k2, ...]
        is_rhf = mo_coeff[0][0].ndim == 1
    else:
        is_rhf = mo_coeff[0].ndim == 1

    with_lapl = False
    nao = cell.nao_nr()
    if is_rhf:
        rho = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rho.append(ni.eval_rho2(cell, ao_k1, mo_coeff, mo_occ, mask, xctype, with_lapl))
        rho = numpy.hstack(rho)
        if spin == 1:
            rho *= .5
            rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
    else:
        assert spin == 1
        rhoa = []
        rhob = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rhoa.append(ni.eval_rho2(cell, ao_k1, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl))
            rhob.append(ni.eval_rho2(cell, ao_k1, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl))
        rho = numpy.stack([numpy.hstack(rhoa), numpy.hstack(rhob)])
    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
    return rho, vxc, fxc

def cache_xc_kernel1(ni, cell, grids, xc_code, dm, spin=0,
                     kpts=None, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if numint.MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0

    if isinstance(ni, KNumInt):
        is_rhf = dm[0][0].ndim == 1
    else:
        is_rhf = dm[0].ndim == 1

    # 0th order density matrix must be hermitian
    hermi = 1
    with_lapl = False
    nao = cell.nao_nr()
    if is_rhf:
        rho = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rho.append(ni.eval_rho1(cell, ao_k1, dm, mask, xctype, hermi, with_lapl))
        rho = numpy.hstack(rho)
        if spin == 1:
            rho *= .5
            rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
    else:
        assert spin == 1
        rhoa = []
        rhob = []
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            rhoa.append(ni.eval_rho1(cell, ao_k1, dm[0], mask, xctype, hermi, with_lapl))
            rhob.append(ni.eval_rho1(cell, ao_k1, dm[1], mask, xctype, hermi, with_lapl))
        rho = numpy.stack([numpy.hstack(rhoa), numpy.hstack(rhob)])
    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype)[1:3]
    return rho, vxc, fxc


def get_rho(ni, cell, dm, grids, kpts=numpy.zeros((1,3)), max_memory=2000):
    '''Density in real space
    '''
    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm, hermi, False)
    assert nset == 1
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao_k1, ao_k2, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, 0, kpts, None, max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao_k1, mask, 'LDA')
    return rho


class NumInt(lib.StreamObject, numint.LibXCMixin):
    '''Generalization of pyscf's NumInt class for a single k-point shift and
    periodic images.
    '''

    cutoff = CUTOFF * 1e2  # cutoff for small AO product

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix.
        See :func:`nr_rks` and :func:`nr_uks` for more details.
        '''
        if spin == 0:
            return self.nr_rks(cell, grids, xc_code, dms, 0, hermi,
                               kpt, kpts_band, max_memory, verbose)
        else:
            return self.nr_uks(cell, grids, xc_code, dms, 0, hermi,
                               kpt, kpts_band, max_memory, verbose)
    get_vxc = nr_vxc

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = self.view(KNumInt)
            nao = dms.shape[-1]
            dms = dms.reshape(-1,nao,nao)
            return ni.nr_rks(cell, grids, xc_code, dms, relativity,
                             hermi, kpt.reshape(1,3), kpts_band, max_memory, verbose)
        spin = 0
        return nr_rks(self, cell, grids, xc_code, dms, spin, relativity,
                      hermi, kpt, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = self.view(KNumInt)
            nao = dms.shape[-1]
            dms = dms.reshape(2,-1,nao,nao)
            return ni.nr_uks(cell, grids, xc_code, dms, relativity,
                             hermi, kpt.reshape(1,3), kpts_band, max_memory, verbose)
        spin = 1
        return nr_uks(self, cell, grids, xc_code, dms, spin, relativity,
                      hermi, kpt, kpts_band, max_memory, verbose)

    def _vxc_mat(self, cell, ao, wv, mask, xctype, shls_slice, ao_loc, hermi):
        if xctype == 'MGGA':
            tau_idx = 4
            wv[tau_idx] *= .5 # *.5 for 1/2 in tau
        if hermi == 1:
            # *.5 because mat + mat.T in the caller when hermi=1
            wv[0] *= .5
            if xctype == 'MGGA':
                wv[tau_idx] *= .5
        return _vxc_mat(cell, ao, wv, mask, xctype, shls_slice, ao_loc, hermi)

    @lib.with_doc(nr_rks_fxc.__doc__)
    def nr_fxc(self, cell, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, kpt=None, max_memory=2000,
               verbose=None):
        if spin == 0:
            return self.nr_rks_fxc(cell, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, kpt, max_memory, verbose)
        else:
            return self.nr_uks_fxc(cell, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, kpt, max_memory, verbose)
    get_fxc = nr_fxc

    def cache_xc_kernel(self, cell, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        kpt=None, max_memory=2000):
        return cache_xc_kernel(self, cell, grids, xc_code, mo_coeff, mo_occ, spin,
                               kpt, max_memory)

    def cache_xc_kernel1(self, cell, grids, xc_code, dm, spin=0,
                         kpt=None, max_memory=2000):
        return cache_xc_kernel1(self, cell, grids, xc_code, dm, spin, kpt, max_memory)

    def block_loop(self, cell, grids, nao=None, deriv=0, kpt=None,
                   kpts_band=None, max_memory=2000, non0tab=None, blksize=None):
        '''Define this macro to loop over grids by blocks.
        '''
        # For UniformGrids, grids.coords does not indicate whehter grids are initialized
        if grids.non0tab is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = cell.nao
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
        # NOTE to index grids.non0tab, blksize needs to be integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nao*16*BLKSIZE))
            blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 2400)) * BLKSIZE
        assert blksize % BLKSIZE == 0
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = 0xff

        if kpt is None:
            kpt = numpy.zeros(3)
        else:
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
            ao_k2 = self.eval_ao(cell, coords, kpt2, deriv=deriv, non0tab=non0,
                                 cutoff=grids.cutoff)
            if abs(kpt1-kpt2).sum() < 1e-9:
                ao_k1 = ao_k2
            else:
                ao_k1 = self.eval_ao(cell, coords, kpt1, deriv=deriv,
                                     cutoff=grids.cutoff)
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

    _gen_rho_evaluator = numint.NumInt._gen_rho_evaluator

    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    nr_nlc_vxc = nr_nlc_vxc

    make_mask = staticmethod(make_mask)
    eval_ao = staticmethod(eval_ao)
    eval_rho = staticmethod(eval_rho)
    eval_rho2 = staticmethod(eval_rho2)
    get_rho = get_rho

    def eval_rho1(self, cell, ao, dm, screen_index=None, xctype='LDA', hermi=0,
                  with_lapl=True, cutoff=None, ao_cutoff=None, verbose=None):
        return self.eval_rho(cell, ao, dm, screen_index, xctype, hermi,
                             with_lapl, verbose)

    def to_gpu(self):
        from gpu4pyscf.pbc.dft.numint import NumInt # type: ignore
        return NumInt()

_NumInt = NumInt


class KNumInt(lib.StreamObject, numint.LibXCMixin):
    '''Generalization of pyscf's NumInt class for k-point sampling and
    periodic images.
    '''
    def __init__(self, kpts=numpy.zeros((1,3))):
        self.kpts = numpy.reshape(kpts, (-1,3))

    eval_ao = staticmethod(eval_ao_kpts)

    @lib.with_doc(make_mask.__doc__)
    def make_mask(self, cell, coords, relativity=0, shls_slice=None,
                  verbose=None):
        return make_mask(cell, coords, relativity, shls_slice, verbose)

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 hermi=0, with_lapl=True, verbose=None):
        '''Collocate the density (opt. gradients) on the real-space grid.

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
        rho_ks = [eval_rho(cell, ao_kpts[k], dm_kpts[k], non0tab, xctype,
                           hermi, with_lapl, verbose)
                  for k in range(nkpts)]
        dtype = numpy.result_type(*rho_ks)
        rho = numpy.zeros(rho_ks[0].shape, dtype=dtype)
        for k in range(nkpts):
            rho += rho_ks[k]
        rho *= 1./nkpts
        return rho

    def eval_rho1(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA', hermi=0,
                  with_lapl=True, cutoff=CUTOFF, grids=None, verbose=None):
        return self.eval_rho(cell, ao_kpts, dm_kpts, non0tab, xctype, hermi,
                             with_lapl, verbose)

    def eval_rho2(self, cell, ao_kpts, mo_coeff_kpts, mo_occ_kpts,
                  non0tab=None, xctype='LDA', with_lapl=True, verbose=None):
        nkpts = len(ao_kpts)
        rhoR = 0
        for k in range(nkpts):
            rhoR += eval_rho2(cell, ao_kpts[k], mo_coeff_kpts[k],
                              mo_occ_kpts[k], non0tab, xctype, with_lapl, verbose)
        rhoR *= 1./nkpts
        return rhoR

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
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
    get_vxc = nr_vxc

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None, **kwargs):
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
    def nr_uks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None, **kwargs):
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

    def _vxc_mat(self, cell, ao_kpts, wv, mask, xctype, shls_slice, ao_loc, hermi):
        r'''Numerical integration \sum_{kpt,i} wv_i * ao_{kpt,i}.conj() * ao_{kpt,i}'''
        nkpts = len(ao_kpts)
        nao = ao_kpts[0].shape[-1]
        dtype = numpy.result_type(wv, *ao_kpts)
        mat = numpy.empty((nkpts,nao,nao), dtype=dtype)
        if xctype == 'MGGA':
            tau_idx = 4
            wv[tau_idx] *= .5 # *.5 for 1/2 in tau
        if hermi == 1:
            # *.5 because mat + mat.T in the caller when hermi=1
            wv[0] *= .5
            if xctype == 'MGGA':
                wv[tau_idx] *= .5

        for k in range(nkpts):
            mat[k] = _vxc_mat(cell, ao_kpts[k], wv, mask, xctype, shls_slice,
                              ao_loc, hermi)
        return mat

    @lib.with_doc(nr_rks_fxc.__doc__)
    def nr_fxc(self, cell, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, kpts=None, max_memory=2000,
               verbose=None):
        if spin == 0:
            return self.nr_rks_fxc(cell, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, kpts, max_memory, verbose)
        else:
            return self.nr_uks_fxc(cell, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, kpts, max_memory, verbose)
    get_fxc = nr_fxc

    def block_loop(self, cell, grids, nao=None, deriv=0, kpts=None,
                   kpts_band=None, max_memory=2000, non0tab=None, blksize=None):
        '''Define this macro to loop over grids by blocks.
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = cell.nao
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6

        if kpts is None:
            kpts = numpy.zeros((1,3))

        kpts_all = kpts
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            dk = abs(kpts[:,None] - kpts_band).max(axis=2)
            k_idx, kband_idx = numpy.where(dk < KPT_DIFF_TOL)
            where = numpy.empty(len(kpts_band), dtype=int)
            where[kband_idx] = k_idx
            kband_mask = numpy.ones(len(kpts_band), dtype=bool)
            kband_mask[kband_idx] = False
            kpts_band_uniq = kpts_band[kband_mask]
            if kpts_band_uniq.size > 0:
                kpts_all = numpy.vstack([kpts,kpts_band_uniq])
                where[kband_mask] = len(kpts) + numpy.arange(kpts_band_uniq.shape[0])

# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*len(kpts_all)*nao*16*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*2400))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = 0xff

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao_kall = self.eval_ao(cell, coords, kpts_all, deriv=deriv, non0tab=non0)
            if kpts_band is None:
                ao_k1 = ao_k2 = ao_kall
            else:
                ao_k1 = [ao_kall[k] for k in where]
                ao_k2 = ao_kall[:len(kpts)]
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

    def _gen_rho_evaluator(self, cell, dms, hermi=0, with_lapl=False, grids=None):
        if getattr(dms, 'mo_coeff', None) is not None:
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms[0], numpy.ndarray) and dms[0].ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
            nao = dms[0].shape[-1]
            ndms = len(mo_occ)
            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(cell, ao, mo_coeff[idm], mo_occ[idm],
                                      non0tab, xctype, with_lapl)
        else:
            if isinstance(dms[0], numpy.ndarray):
                if dms[0].ndim == 2:
                    dms = [numpy.stack(dms)]
                elif dms[0].ndim != 3:
                    raise RuntimeError(f'dm has incorrect dimension {numpy.ndim(dms)}')
            nao = dms[0].shape[-1]
            ndms = len(dms)
            def make_rho(idm, ao_kpts, non0tab, xctype):
                return self.eval_rho(cell, ao_kpts, dms[idm], non0tab, xctype,
                                     hermi, with_lapl)
        return make_rho, ndms, nao

    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    cache_xc_kernel  = cache_xc_kernel
    cache_xc_kernel1 = cache_xc_kernel1
    get_rho = get_rho

    def to_gpu(self):
        from gpu4pyscf.pbc.dft.numint import KNumInt # type: ignore
        return KNumInt()

_KNumInt = KNumInt
