#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.symm import symmetry as symm
from pyscf.pbc.lib.kpts_helper import member, KPT_DIFF_TOL, KptsHelper
from numpy.linalg import inv

libpbc = lib.load_library('libpbc')

def make_kpts_ibz(kpts):
    """
    Locate k-points in IBZ.

    Parameters
    ----------
        kpts : :class:`KPoints` object

    Notes
    -----
        This function modifies the :obj:`kpts` object.
    """
    cell = kpts.cell
    nkpts = kpts.nkpts
    nop = kpts.nop
    op_rot = np.asarray([op.a2b(cell).rot for op in kpts.ops])
    if kpts.time_reversal:
        op_rot = np.concatenate([op_rot, -op_rot])

    bz2bz_ks = map_k_points_fast(kpts.kpts_scaled, op_rot, KPT_DIFF_TOL)
    kpts.k2opk = bz2bz_ks.copy()
    if -1 in bz2bz_ks:
        bz2bz_ks[:, np.unique(np.where(bz2bz_ks == -1)[1])] = -1
        if kpts.verbose >= logger.WARN:
            logger.warn(kpts, 'k-points have lower symmetry than lattice.')

    bz2bz_k = -np.ones(nkpts+1, dtype=int)
    ibz2bz_k = []
    for k in range(nkpts-1, -1, -1):
        if bz2bz_k[k] == -1:
            bz2bz_k[bz2bz_ks[k]] = k
            ibz2bz_k.append(k)
    ibz2bz_k = np.array(ibz2bz_k[::-1])
    bz2bz_k = bz2bz_k[:-1].copy()

    bz2ibz_k = np.empty(nkpts, int)
    bz2ibz_k[ibz2bz_k] = np.arange(len(ibz2bz_k))
    bz2ibz_k = bz2ibz_k[bz2bz_k]

    kpts.bz2ibz = bz2ibz_k
    kpts.ibz2bz = ibz2bz_k
    kpts.weights_ibz = np.bincount(bz2ibz_k) * (1.0 / nkpts)
    kpts.kpts_scaled_ibz = kpts.kpts_scaled[kpts.ibz2bz]
    kpts.kpts_ibz = kpts.cell.get_abs_kpts(kpts.kpts_scaled_ibz)
    kpts.nkpts_ibz = len(kpts.kpts_ibz)

    for k in range(nkpts):
        bz_k_scaled = kpts.kpts_scaled[k]
        ibz_idx = kpts.bz2ibz[k]
        ibz_k_scaled = kpts.kpts_scaled_ibz[ibz_idx]
        for io, op in enumerate(op_rot):
            if -1 in bz2bz_ks[:,io]:
                #This rotation is not in the subgroup
                #that the k-mesh belongs to; only happens
                #when k-mesh has lower symmetry than lattice.
                continue
            diff = bz_k_scaled - np.dot(ibz_k_scaled, op.T)
            diff = diff - diff.round()
            if (np.absolute(diff) < KPT_DIFF_TOL).all():
                kpts.time_reversal_symm_bz[k] = io // nop
                kpts.stars_ops_bz[k] = io % nop
                break

    for i in range(kpts.nkpts_ibz):
        ibz_k_scaled = kpts.kpts_scaled_ibz[i]
        idx = np.where(kpts.bz2ibz == i)[0]
        kpts.stars.append(idx)
        kpts.stars_ops.append(kpts.stars_ops_bz[idx])

def make_ktuples_ibz(kpts, kpts_scaled=None, ntuple=2, tol=KPT_DIFF_TOL):
    """
    Constructe k-point tuples in IBZ.

    Parameters
    ----------
        kpts : :class:`KPoints` object
        kpts_scaled : (nkpts, ntuple, 3) ndarray
            Input k-points among which symmetry relations are seeked.
            Default is None, meaning all the k-points in :obj:`kpts` are considered.
        ntuple : int
             Dimension of the tuples. Default is 2.
        tol : float
            K-points differ by ``tol`` are considered as different.
            Default is 1e-6.

    Returns
    -------
        ibz2bz_kk : (nibz,) ndarray of int
            Mapping table from IBZ to full BZ.
        ibz_kk_weight : (nibz,) ndarray of int
            Weights of each k-point tuple in the IBZ.
        bz2ibz_kk : (nkpts**ntuple,) ndarray of int
            Mapping table from full BZ to IBZ.
        kk_group : list of (nk,) ndarrays of int
            Similar as :attr:`.stars`.
        kk_sym_group : list of (nk,) ndarrays of int
            Similar as :attr:`.stars_ops`.
    """
    if kpts_scaled is not None:
        cell = kpts.cell
        op_rot = np.asarray([op.a2b(cell).rot for op in kpts.ops])
        if kpts.time_reversal:
            op_rot = np.concatenate([op_rot, -op_rot])
        bz2bz_ksks = map_k_tuples(kpts_scaled, op_rot, ntuple=ntuple, tol=tol)
        bz2bz_ksks[:, np.unique(np.where(bz2bz_ksks == -1)[1])] = -1
        nbzk2 = bz2bz_ksks.shape[0]
    else:
        bz2bz_ks = kpts.k2opk
        nop = bz2bz_ks.shape[-1]
        nbzk = kpts.nkpts
        nbzk2 = nbzk**ntuple
        bz2bz_T = np.zeros([nop, nbzk2], dtype=int)
        for iop in range(nop):
            tmp = lib.cartesian_prod([bz2bz_ks[:,iop]]*ntuple)
            if -1 in tmp:
                bz2bz_T[iop,:] = -1
            else:
                #idx_throw = np.unique(np.where(tmp == -1)[0])
                for i in range(ntuple):
                    bz2bz_T[iop] += tmp[:,i] * nbzk**(ntuple-i-1)
                #bz2bz_T[iop, idx_throw] = -1
        bz2bz_ksks = bz2bz_T.T

    if kpts.verbose >= logger.INFO:
        logger.info(kpts, 'Number of k-point %d-tuples: %d', ntuple, nbzk2)

    bz2bz_kk = -np.ones(nbzk2+1, dtype=int)
    ibz2bz_kk = []
    k_group = []
    sym_group = []
    for k in range(nbzk2-1, -1, -1):
        if bz2bz_kk[k] == -1:
            bz2bz_kk[bz2bz_ksks[k]] = k
            ibz2bz_kk.append(k)
            k_idx, op_idx = np.unique(bz2bz_ksks[k], return_index=True)
            if k_idx[0] == -1:
                k_idx = k_idx[1:]
                op_idx = op_idx[1:]
            k_group.append(k_idx)
            sym_group.append(op_idx)

    ibz2bz_kk = np.array(ibz2bz_kk[::-1])
    if kpts.verbose >= logger.INFO:
        logger.info(kpts, 'Number of k %d-tuples in IBZ: %s', ntuple, len(ibz2bz_kk))

    bz2bz_kk = bz2bz_kk[:-1].copy()
    bz2ibz_kk = np.empty(nbzk2,dtype=int)
    bz2ibz_kk[ibz2bz_kk] = np.arange(len(ibz2bz_kk))
    bz2ibz_kk = bz2ibz_kk[bz2bz_kk]

    kk_group = k_group[::-1]
    kk_sym_group = sym_group[::-1]
    ibz_kk_weight = np.bincount(bz2ibz_kk) * (1.0 / nbzk2)
    return ibz2bz_kk, ibz_kk_weight, bz2ibz_kk, kk_group, kk_sym_group

def make_k4_ibz(kpts, sym='s1'):
    #physicist's notation
    ibz2bz, weight, bz2ibz, group, _ = kpts.make_ktuples_ibz(ntuple=3)
    khelper = KptsHelper(kpts.cell, kpts.kpts)
    k4 = []
    for ki, kj, ka in kpts.loop_ktuples(ibz2bz, 3):
        kb = khelper.kconserv[ki,ka,kj]
        k4.append([ki,kj,ka,kb])

    if sym == "s1":
        return np.asarray(k4), np.asarray(weight), np.asarray(bz2ibz)
    elif sym == "s2" or sym == "s4":
        ibz2ibz_s2 = np.arange(len(k4))
        k4_s2 = []
        weight_s2 = []
        for i, k in enumerate(k4):
            ki,kj,ka,kb = k
            k_sym = [kj,ki,kb,ka] #interchange dummy indices
            if k not in k4_s2 and k_sym not in k4_s2:
                k4_s2.append(k)
                ibz2ibz_s2[i] = len(k4_s2) - 1
                w = weight[i]
                if k != k_sym and k_sym in k4:
                    idx = k4.index(k_sym)
                    ibz2ibz_s2[idx] = ibz2ibz_s2[i]
                    w += weight[idx]
                weight_s2.append(w)
        #refine s2 symmetry
        k4_s2_refine = []
        weight_s2_refine = []
        skip = np.zeros([len(k4_s2)], dtype=int)
        ibz_s22ibz_s2_refine = np.arange(len(k4_s2))
        for i, k in enumerate(k4_s2):
            if skip[i]: continue
            ki,kj,ka,kb = k
            k_sym = [kj,ki,kb,ka]
            if ki==kj and ka==kb:
                k4_s2_refine.append(k)
                ibz_s22ibz_s2_refine[i] = len(k4_s2_refine) - 1
                weight_s2_refine.append(weight_s2[i])
                continue
            idx_sym = None
            for j in range(i+1, len(k4_s2)):
                if skip[j]: continue
                k_tmp = k4_s2[j]
                if ki in k_tmp and kj in k_tmp and ka in k_tmp and kb in k_tmp:
                    idx = k4.index(k_tmp)
                    for kii,kjj,kaa in kpts.loop_ktuples(group[idx], 3):
                        kbb = khelper.kconserv[kii,kaa,kjj]
                        if k_sym == [kii,kjj,kaa,kbb]:
                            idx_sym = j
                            break
                    if idx_sym is not None:
                        break
            w = weight_s2[i]
            if idx_sym is not None:
                skip[idx_sym] = 1
                w += weight_s2[idx_sym]
                ibz_s22ibz_s2_refine[idx_sym] = len(k4_s2_refine)
            k4_s2_refine.append(k)
            ibz_s22ibz_s2_refine[i] = len(k4_s2_refine) - 1
            weight_s2_refine.append(w)
        k4_s2 = k4_s2_refine
        weight_s2 = weight_s2_refine
        #end refine
        if sym == "s2":
            k4_s2 = np.asarray(k4_s2)
            weight_s2 = np.asarray(weight_s2)
            idx = np.lexsort(k4_s2.T[::-1,:])
            bz2ibz_s2 = np.arange(len(bz2ibz))
            for i in range(len(bz2ibz)):
                bz2ibz_s2[i] = np.where(idx == ibz_s22ibz_s2_refine[ibz2ibz_s2[bz2ibz[i]]])[0]
            return k4_s2[idx], weight_s2[idx], bz2ibz_s2
        else:
            k4_s4 = []
            weight_s4 = []
            for i, k in enumerate(k4_s2):
                ki,kj,ka,kb = k
                k_sym = [ka,kb,ki,kj] #complex conjugate
                if k not in k4_s4 and k_sym not in k4_s4:
                    k4_s4.append(k)
                    w = weight_s2[i]
                    if k != k_sym and k_sym in k4_s2:
                        idx = k4_s2.index(k_sym)
                        w += weight_s2[idx]
                    weight_s4.append(w)
            k4_s4 = np.asarray(k4_s4)
            weight_s4 = np.asarray(weight_s4)
            idx = np.lexsort(k4_s4.T[::-1,:])
            return k4_s4[idx], weight_s4[idx], None
    else:
        raise NotImplementedError("Unsupported symmetry.")
    return

def map_k_points_fast(kpts_scaled, ops, tol=KPT_DIFF_TOL):
    #This routine is modified from GPAW
    """
    Find symmetry-related k-points.

    Parameters
    ----------
        kpts_scaled : (nkpts, 3) ndarray
            Scaled k-points.
        ops : (nop, 3, 3) ndarray of int
            Rotation operators.
        tol : float
            K-points differ by ``tol`` are considered as different.
            Default is 1e-6.

    Returns
    -------
        bz2bz_ks : (nkpts, nop) ndarray of int
            mapping table between k and op*k.
            bz2bz_ks[k1,s] = k2 if ops[s] * kpts_scaled[k1] = kpts_scaled[k2] + K,
            where K is a reciprocal lattice vector.
    """
    nkpts = len(kpts_scaled)
    nop = len(ops)
    bz2bz_ks = -np.ones((nkpts, nop), dtype=int)
    for s, op in enumerate(ops):
        # Find mapped kpoints
        op_kpts_scaled = np.dot(kpts_scaled, op.T)

        # Do some work on the input
        k_kc = np.concatenate([kpts_scaled, op_kpts_scaled])
        k_kc = np.mod(np.mod(k_kc, 1), 1)
        k_kc = aglomerate_points(k_kc, tol)
        k_kc = k_kc.round(-np.log10(tol).astype(int))
        k_kc = np.mod(k_kc, 1)

        # Find the lexicographical order
        order = np.lexsort(k_kc.T)
        k_kc = k_kc[order]
        diff_kc = np.diff(k_kc, axis=0)
        equivalentpairs_k = np.array((diff_kc == 0).all(1), dtype=bool)

        # Mapping array.
        orders = np.array([order[:-1][equivalentpairs_k],
                           order[1:][equivalentpairs_k]])

        # This has to be true.
        assert (orders[0] < nkpts).all()
        assert (orders[1] >= nkpts).all()
        bz2bz_ks[orders[1] - nkpts, s] = orders[0]
    return bz2bz_ks

def map_k_tuples(kpts_scaled, ops, ntuple=2, tol=KPT_DIFF_TOL):
    """
    Find symmetry-related k-point tuples.

    Parameters
    ----------
        kpts_scaled : (nkpts, ntuple, 3) ndarray
            Scaled k-point tuples.
        ops : (nop, 3, 3) ndarray of int
            Rotation operators.
        ntuple : int
            Dimension of tuples. Default is 2.
        tol : float
            K-points differ by ``tol`` are considered as different.
            Default is 1e-6.

    Returns
    -------
        bz2bz_ks : (nkpts, nop) ndarray of int
            mapping table between k and op*k.
            bz2bz_ks[k1,s] = k2 if ops[s] * kpts_scaled[k1] = kpts_scaled[k2] + K,
            where K is a reciprocal lattice vector.
    """
    nkpts = len(kpts_scaled)
    nop = len(ops)
    bz2bz_ks = -np.ones((nkpts, nop), dtype=int)
    for s, op in enumerate(ops):
        # Find mapped kpoints
        op_kpts_scaled = np.empty((nkpts, ntuple, 3), dtype=float)
        for i in range(ntuple):
            op_kpts_scaled[:,i] = np.dot(kpts_scaled[:,i], op.T)
        op_kpts_scaled = op_kpts_scaled.reshape((nkpts,-1))

        # Do some work on the input
        k_kc = np.concatenate([kpts_scaled.reshape((nkpts,-1)), op_kpts_scaled])
        k_kc = np.mod(np.mod(k_kc, 1), 1)
        k_kc = aglomerate_points(k_kc, tol)
        k_kc = k_kc.round(-np.log10(tol).astype(int))
        k_kc = np.mod(k_kc, 1)

        # Find the lexicographical order
        order = np.lexsort(k_kc.T)
        k_kc = k_kc[order]
        diff_kc = np.diff(k_kc, axis=0)
        equivalentpairs_k = np.array((diff_kc == 0).all(1), dtype=bool)

        # Mapping array.
        orders = np.array([order[:-1][equivalentpairs_k],
                           order[1:][equivalentpairs_k]])

        # This has to be true.
        assert (orders[0] < nkpts).all()
        assert (orders[1] >= nkpts).all()
        bz2bz_ks[orders[1] - nkpts, s] = orders[0]
    return bz2bz_ks

def aglomerate_points(k_kc, tol=KPT_DIFF_TOL):
    #This routine is adopted from GPAW
    '''
    Remove numerical error
    '''
    nd = k_kc.shape[1]
    nbzkpts = len(k_kc)

    inds_kc = np.argsort(k_kc, axis=0)

    for c in range(nd):
        sk_k = k_kc[inds_kc[:, c], c]
        dk_k = np.diff(sk_k)

        pt_K = np.argwhere(dk_k > tol)[:, 0]
        pt_K = np.append(np.append(0, pt_K + 1), nbzkpts*2)
        for i in range(len(pt_K) - 1):
            k_kc[inds_kc[pt_K[i]:pt_K[i + 1], c], c] = k_kc[inds_kc[pt_K[i], c], c]
    return k_kc

def symmetrize_density(kpts, rhoR_k, ibz_k_idx, mesh):
    '''
    Transform real-space densities from IBZ to full BZ
    '''
    rhoR_k = np.asarray(rhoR_k, order='C')
    rhoR = np.zeros_like(rhoR_k, order='C')

    dtype = rhoR_k.dtype
    if dtype == np.double:
        symmetrize = libpbc.symmetrize
        symmetrize_ft = libpbc.symmetrize_ft
    elif dtype == np.complex128:
        symmetrize = libpbc.symmetrize_complex
        symmetrize_ft = libpbc.symmetrize_ft_complex
    else:
        raise RuntimeError("Unsupported data type %s" % dtype)

    c_rhoR = rhoR.ctypes.data_as(ctypes.c_void_p)
    c_rhoR_k = rhoR_k.ctypes.data_as(ctypes.c_void_p)

    mesh = np.asarray(mesh, dtype=np.int32, order='C')
    c_mesh = mesh.ctypes.data_as(ctypes.c_void_p)
    for iop in kpts.stars_ops[ibz_k_idx]:
        op = kpts.ops[iop]
        if op.is_eye: #or op.is_inversion:
            rhoR += rhoR_k
        else:
            inv_op = op.inv()
            op_rot = np.asarray(inv_op.rot, dtype=np.int32, order='C')
            c_op_rot = op_rot.ctypes.data_as(ctypes.c_void_p)
            if inv_op.trans_is_zero:
                symmetrize(c_rhoR, c_rhoR_k, c_op_rot, c_mesh)
            else:
                trans = np.asarray(inv_op.trans, dtype=np.double, order='C')
                c_trans = trans.ctypes.data_as(ctypes.c_void_p)
                symmetrize_ft(c_rhoR, c_rhoR_k, c_op_rot, c_trans, c_mesh)
    return rhoR

def symmetrize_wavefunction(kpts, psiR_k, mesh):
    '''
    transform real-space wavefunctions from IBZ to full BZ
    '''
    raise RuntimeError('need verification')
    psiR_k = np.asarray(psiR_k, order='C')
    is_complex = psiR_k.dtype == np.complex
    nao = psiR_k.shape[1]
    nG = psiR_k.shape[2]
    psiR = np.zeros([kpts.nkpts,nao,nG], dtype = psiR_k.dtype, order='C')

    mesh = np.asarray(mesh, dtype=np.int32, order='C')
    c_mesh = mesh.ctypes.data_as(ctypes.c_void_p)

    for ibz_k_idx in range(kpts.nkpts_ibz):
        for idx, iop in enumerate(kpts.stars_ops[ibz_k_idx]):
            bz_k_idx = kpts.stars[ibz_k_idx][idx]
            op = kpts.ops[iop].b2a(kpts.cell).rot
            op = np.asarray(op, dtype=np.int32, order='C')
            time_reversal = kpts.time_reversal_symm_bz[bz_k_idx]
            if symm.is_eye(op): #or symm.is_inversion(op):
                psiR[bz_k_idx] = psiR_k[ibz_k_idx]
            else:
                c_psiR = psiR[bz_k_idx].ctypes.data_as(ctypes.c_void_p)
                c_psiR_k = psiR_k[ibz_k_idx].ctypes.data_as(ctypes.c_void_p)
                c_op = op.ctypes.data_as(ctypes.c_void_p)
                if is_complex:
                    libpbc.symmetrize_complex(c_psiR, c_psiR_k, c_op, c_mesh)
                else:
                    libpbc.symmetrize(c_psiR, c_psiR_k, c_op, c_mesh)
            if time_reversal and is_complex:
                psiR[bz_k_idx] = psiR[bz_k_idx].conj()
    return psiR

def transform_mo_coeff(kpts, mo_coeff_ibz):
    """
    Transform MO coefficients from IBZ to full BZ.

    Parameters
    ----------
        kpts : :class:`KPoints` object
        mo_coeff_ibz : ([2,] nkpts_ibz, nao, nmo) ndarray
            MO coefficients for k-points in IBZ.

    Returns
    -------
        mo_coeff_bz : ([2,] nkpts, nao, nmo) ndarray
            MO coefficients for k-points in full BZ.
    """
    mos = []
    is_uhf = False
    if isinstance(mo_coeff_ibz[0][0], np.ndarray) and mo_coeff_ibz[0][0].ndim == 2:
        is_uhf = True
        mos = [[],[]]
    for k in range(kpts.nkpts):
        ibz_k_idx = kpts.bz2ibz[k]
        ibz_k_scaled = kpts.kpts_scaled_ibz[ibz_k_idx]
        iop = kpts.stars_ops_bz[k]
        op = kpts.ops[iop]
        time_reversal = kpts.time_reversal_symm_bz[k]

        def _transform(mo_ibz, iop, op):
            if op.is_eye:
                mo_bz = mo_ibz
            else:
                mo_bz = symm.transform_mo_coeff(kpts.cell, ibz_k_scaled, mo_ibz, op, kpts.Dmats[iop])
            if time_reversal:
                mo_bz = mo_bz.conj()
            return mo_bz

        if is_uhf:
            mo_coeff_a = mo_coeff_ibz[0][ibz_k_idx]
            mos[0].append(_transform(mo_coeff_a, iop, op))
            mo_coeff_b = mo_coeff_ibz[1][ibz_k_idx]
            mos[1].append(_transform(mo_coeff_b, iop, op))
        else:
            mo_coeff = mo_coeff_ibz[ibz_k_idx]
            mos.append(_transform(mo_coeff, iop, op))
    return mos

def transform_mo_coeff_k(kpts, mo_coeff_ibz, k):
    """
    Get MO coefficients for a single k-point in full BZ.

    Parameters
    ----------
        kpts : :class:`KPoints` object
        mo_coeff_ibz : (nkpts_ibz, nao, nmo) ndarray
            MO coefficients for k-points in IBZ.
        k : int
            Index of the k-point in full BZ.

    Returns
    -------
        mo_coeff_bz : (nao, nmo) ndarray
            MO coefficients for the ``k``-th k-point in full BZ.
    """
    ibz_k_idx = kpts.bz2ibz[k]
    ibz_k_scaled = kpts.kpts_scaled_ibz[ibz_k_idx]
    iop = kpts.stars_ops_bz[k]
    op = kpts.ops[iop]
    time_reversal = kpts.time_reversal_symm_bz[k]

    mo_ibz = mo_coeff_ibz[ibz_k_idx]
    if op.is_eye:
        mo_bz = mo_ibz
    else:
        mo_bz = symm.transform_mo_coeff(kpts.cell, ibz_k_scaled, mo_ibz, op, kpts.Dmats[iop])
    if time_reversal:
        mo_bz = mo_bz.conj()
    return mo_bz

transform_single_mo_coeff = transform_mo_coeff_k

def transform_mo_occ(kpts, mo_occ_ibz):
    '''
    Transform MO occupations from IBZ to full BZ
    '''
    occ = []
    is_uhf = False
    if isinstance(mo_occ_ibz[0][0], np.ndarray) and mo_occ_ibz[0][0].ndim == 1:
        is_uhf = True
        occ = [[],[]]
    for k in range(kpts.nkpts):
        ibz_k_idx = kpts.bz2ibz[k]
        if is_uhf:
            occ[0].append(mo_occ_ibz[0][ibz_k_idx])
            occ[1].append(mo_occ_ibz[1][ibz_k_idx])
        else:
            occ.append(mo_occ_ibz[ibz_k_idx])
    return occ

def transform_dm(kpts, dm_ibz):
    """
    Transform density matrices from IBZ to full BZ.

    Parameters
    ----------
        kpts : :class:`KPoints` object
        dm_ibz : ([2,] nkpts_ibz, nao, nao) ndarray
            Density matrices for k-points in IBZ.

    Returns
    -------
        dm_bz : ([2,] nkpts, nao, nao) ndarray
            Density matrices for k-points in full BZ.
    """
    mo_occ = mo_coeff = None
    if getattr(dm_ibz, "mo_coeff", None) is not None:
        mo_coeff = kpts.transform_mo_coeff(dm_ibz.mo_coeff)
        mo_occ = kpts.transform_mo_occ(dm_ibz.mo_occ)

    dms = []
    is_uhf = False
    if (isinstance(dm_ibz, np.ndarray) and dm_ibz.ndim == 4) or \
       (isinstance(dm_ibz[0][0], np.ndarray) and dm_ibz[0][0].ndim == 2):
        is_uhf = True
        dms = [[],[]]
    for k in range(kpts.nkpts):
        ibz_k_idx = kpts.bz2ibz[k]
        ibz_kpt_scaled = kpts.kpts_scaled_ibz[ibz_k_idx]
        iop = kpts.stars_ops_bz[k]
        op = kpts.ops[iop]
        time_reversal = kpts.time_reversal_symm_bz[k]

        def _transform(dm_ibz, iop, op):
            if op.is_eye:
                dm_bz = dm_ibz
            else:
                dm_bz = symm.transform_dm(kpts.cell, ibz_kpt_scaled, dm_ibz, op, kpts.Dmats[iop])
            if time_reversal:
                dm_bz = dm_bz.conj()
            return dm_bz

        if is_uhf:
            dm_a = dm_ibz[0][ibz_k_idx]
            dms[0].append(_transform(dm_a, iop, op))
            dm_b = dm_ibz[1][ibz_k_idx]
            dms[1].append(_transform(dm_b, iop, op))
        else:
            dm = dm_ibz[ibz_k_idx]
            dms.append(_transform(dm, iop, op))
    if is_uhf:
        nkpts = len(dms[0])
        nao = dms[0][0].shape[0]
        dms = lib.asarray(dms).reshape(2,nkpts,nao,nao)
    else:
        dms = lib.asarray(dms)
    return lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)

def dm_at_ref_cell(kpts, dm_ibz):
    """
    Given the density matrices in IBZ, compute the reference cell density matrix.

    Parameters
    ----------
        kpts : :class:`KPoints` object
        dm_ibz : ([2,] nkpts_ibz, nao, nao) ndarray
            Density matrices for k-points in IBZ.

    Returns
    -------
        dm0 : ([2,] nao, nao) ndarray
            Density matrix at reference cell.
    """
    dm_bz = transform_dm(kpts, dm_ibz)
    dm0 = np.sum(dm_bz, axis=-3) / kpts.nkpts
    if abs(dm0.imag).max() > 1e-10:
        logger.warn(kpts, 'Imaginary density matrix found at reference cell: \
                    abs(dm0.imag).max() = %g', abs(dm0.imag).max())
    return dm0

def transform_mo_energy(kpts, mo_energy_ibz):
    '''
    Transform MO energies from IBZ to full BZ
    '''
    is_uhf = False
    if isinstance(mo_energy_ibz[0][0], np.ndarray):
        is_uhf = True
    mo_energy_bz = []
    if is_uhf:
        mo_energy_bz = [[],[]]
    for k in range(kpts.nkpts):
        ibz_k_idx = kpts.bz2ibz[k]
        if is_uhf:
            mo_energy_bz[0].append(mo_energy_ibz[0][ibz_k_idx])
            mo_energy_bz[1].append(mo_energy_ibz[1][ibz_k_idx])
        else:
            mo_energy_bz.append(mo_energy_ibz[ibz_k_idx])
    return mo_energy_bz

def transform_1e_operator(kpts, fock_ibz):
    """
    Transform 1-electron operator from IBZ to full BZ.

    Parameters
    ----------
        kpts : :class:`KPoints` object
        fock_ibz : ([2,] nkpts_ibz, nao, nao) ndarray
            Fock-like matrices for k-points in IBZ.

    Returns
    -------
        fock_bz : ([2,] nkpts, nao, nao) ndarray
            Fock-like matrices for k-points in full BZ.
    """
    fock = []
    is_uhf = False
    if isinstance(fock_ibz[0][0], np.ndarray) and fock_ibz[0][0].ndim == 2:
        is_uhf = True
        fock = [[],[]]

    for k in range(kpts.nkpts):
        ibz_k_idx = kpts.bz2ibz[k]
        ibz_kpt_scaled = kpts.kpts_scaled_ibz[ibz_k_idx]
        iop = kpts.stars_ops_bz[k]
        op = kpts.ops[iop]
        time_reversal = kpts.time_reversal_symm_bz[k]

        def _transform(fock_ibz, iop, op):
            if op.is_eye:
                fock_bz = fock_ibz
            else:
                fock_bz = symm.transform_1e_operator(kpts.cell, ibz_kpt_scaled, fock_ibz, op, kpts.Dmats[iop])
            if time_reversal:
                fock_bz = fock_bz.conj()
            return fock_bz

        if is_uhf:
            fock_a = fock_ibz[0][ibz_k_idx]
            fock[0].append(_transform(fock_a, iop, op))
            fock_b = fock_ibz[1][ibz_k_idx]
            fock[1].append(_transform(fock_b, iop, op))
        else:
            fock.append(_transform(fock_ibz[ibz_k_idx], iop, op))
    if is_uhf:
        nkpts = len(fock[0])
        nao = fock[0][0].shape[0]
        fock = lib.asarray(fock).reshape(2,nkpts,nao,nao)
    else:
        fock = lib.asarray(fock)
    return fock

transform_fock = transform_1e_operator

def check_mo_occ_symmetry(kpts, mo_occ, tol=1e-5):
    """
    Check if MO occupations in full BZ have the correct symmetry.
    If not, raise error; else, return MO occupations in IBZ.

    Parameters
    ----------
        kpts : :class:`KPoints` object
        mo_occ : list of (nmo,) ndarray
            MO occupations for k-points in full BZ.
            len(mo_occ) = nkpts
        tol : float
            Occupations differ less than ``tol`` are considered as the same.
            Default is 1e-5.

    Returns
    -------
        mo_occ_ibz : list of (nmo,) ndarray
            MO occupations for k-points in IBZ.
            len(mo_occ_ibz) = nkpts_ibz

    Raises
    ------
        RuntimeError
            If symmetry is broken.
    """
    for bz_k in kpts.stars:
        nbzk = len(bz_k)
        for i in range(nbzk):
            for j in range(i+1,nbzk):
                if not (np.absolute(mo_occ[bz_k[i]] - mo_occ[bz_k[j]]) < tol).all():
                    raise RuntimeError("Symmetry broken solution found. \
                                        This is probably due to KUHF calculations \
                                        with integer occupation numbers. \
                                        Try use smearing or turn off symmetry.")
    mo_occ_ibz = []
    for k in range(kpts.nkpts_ibz):
        mo_occ_ibz.append(mo_occ[kpts.ibz2bz[k]])
    return mo_occ_ibz

def make_kpts(cell, kpts=np.zeros((1,3)),
              space_group_symmetry=False, time_reversal_symmetry=False,
              symmorphic=True):
    """
    A wrapper function to build the :class:`KPoints` object.

    Parameters
    ----------
        cell : :class:`Cell` instance
            Unit cell information.
        kpts : (nkpts,3) ndarray
            K-points in full BZ.
        space_group_symmetry : bool
            Whether to consider space group symmetry. Default is False.
        time_reversal_symmetry : bool
            Whether to consider time reversal symmetry. Default is False.
        symmorphic : bool
            Whether to consider only the symmorphic subgroup. Default is True.

    Examples
    --------
    >>> cell = gto.M(
    ...     atom = '''He 0. 0. 0.''',
    ...     a = numpy.eye(3)*2.0).build()
    >>> kpts = make_kpts(cell,
    ...                  numpy.array([[0.,0.,0.],[0.5,0.,0.],[0.,0.5,0.],[0.,0.,0.5]])
    ...                  space_group_symmetry=True)
    >>> print(kpts.kpts_ibz)
    [[0.  0.  0. ]
     [0.  0.  0.5]]
    """
    if isinstance(kpts, KPoints):
        return kpts.build(space_group_symmetry, time_reversal_symmetry, symmorphic)
    else:
        return KPoints(cell, kpts).build(space_group_symmetry, time_reversal_symmetry, symmorphic)

class KPoints(symm.Symmetry, lib.StreamObject):
    """
    A symmetry object which handles k-point symmetry.

    Parameters
    ----------
        cell : :class:`Cell` instance
            Unit cell information.
        kpts : (nkpts,3) ndarray
            K-points in full BZ.

    Examples
    --------
    >>> cell = gto.M(
    ...     atom = '''He 0. 0. 0.''',
    ...     a = numpy.eye(3)*2.0).build()
    >>> kpts = KPoints(cell, numpy.array([[0.,0.,0.],[0.5,0.,0.],[0.,0.5,0.],[0.,0.,0.5]]))
    >>> kpts.build(space_group_symmetry=True)
    >>> print(kpts.kpts_ibz)
    [[0.  0.  0. ]
     [0.  0.  0.5]]

    Attributes
    ----------
        cell : :class:`Cell` instance
            Unit cell information.
        verbose : int
            Print level. Default value is ``cell.verbose``.
        time_reversal : bool
            Whether to consider time-reversal symmetry.
            For systems with inversion symmetry, time-reversal symmetry is not considered
            unless spin-orbit coupling is present.
        kpts : (nkpts,3) ndarray
            K-points in full BZ.
        kpts_scaled : (nkpts,3) ndarray
            Scaled k-points in full BZ.
        weights : (nkpts,) ndarray
            Weights of k-points in full BZ.
        bz2ibz : (nkpts,) ndarray of int
            Mapping table from full BZ to IBZ.
        kpts_ibz : (nkpts_ibz,3) ndarray
            K-points in IBZ.
        kpts_scaled_ibz : (nkpts_ibz,3) ndarray
            Scaled k-points in IBZ.
        weights_ibz : (nkpts_ibz,) ndarray
            Weights of k-points in IBZ.
        ibz2bz : (nkpts_ibz,) ndarray of int
            Mapping table from IBZ to full BZ.
        k2opk (bz2bz_ks) : (nkpts, nop*(time_reversal+1)) ndarray of int
            Mapping table between kpts and ops.rot * kpts.
        stars : list of (nk,) ndarrays of int
            Stars of k-points in full BZ with len(stars) = nkpts_ibz
            and ``nk`` is the number of symmetry-related k-points for each k-point in IBZ.
        stars_ops : same as ``stars``
            Indices of rotation operators connecting k points in full BZ and in IBZ.
        stars_ops_bz : (nkpts,) ndarray of int
            Same as stars_ops but arranged in the sequence of k-points in full BZ.
        time_reversal_symm_bz : (nkpts,) ndarray of int
            Whether k-points in BZ and IBZ are related in addition by time-reversal symmetry.
    """
    def __init__(self, cell=None, kpts=np.zeros((1,3))):
        symm.Symmetry.__init__(self, cell)
        self.verbose = getattr(cell, 'verbose', logger.NOTE)
        self.stdout = getattr(cell, 'stdout', None)
        self.time_reversal = False

        self.kpts_ibz = self.kpts = kpts
        self.kpts_scaled_ibz = self.kpts_scaled = None
        nkpts = len(self.kpts)
        self.weights_ibz = self.weights = np.asarray([1./nkpts] * nkpts)
        self.ibz2bz = self.bz2ibz = np.arange(nkpts, dtype=int)

        self.k2opk = None
        self.stars = []
        self.stars_ops = []
        self.stars_ops_bz = np.zeros(nkpts, dtype=int)
        self.time_reversal_symm_bz = np.zeros(nkpts, dtype=int)

        #private variables
        self._nkpts = len(self.kpts)
        self._nkpts_ibz = len(self.kpts_ibz)

    @property
    def nkpts(self):
        return self._nkpts

    @nkpts.setter
    def nkpts(self, n):
        self._nkpts = n

    @property
    def nkpts_ibz(self):
        return self._nkpts_ibz

    @nkpts_ibz.setter
    def nkpts_ibz(self, n):
        self._nkpts_ibz = n

    def __str__(self):
        s = ''
        s += self.__repr__()

        s += '\nk-points (scaled) in BZ                        weights\n'
        bzk = self.kpts_scaled
        for k in range(self.nkpts):
            s += '%d:  %11.8f, %11.8f, %11.8f    %9.6f\n' % \
                (k, bzk[k][0], bzk[k][1], bzk[k][2], self.weights[k])

        s += 'k-points (scaled) in IBZ                       weights\n'
        ibzk = self.kpts_scaled_ibz
        for k in range(self.nkpts_ibz):
            s += '%d:  %11.8f, %11.8f, %11.8f    %9.6f\n' % \
                (k, ibzk[k][0], ibzk[k][1], ibzk[k][2], self.weights_ibz[k])

        s += 'mapping from BZ to IBZ\n'
        s += "%s" % self.bz2ibz
        s += '\nmapping from IBZ to BZ\n'
        s += "%s" % self.ibz2bz
        return s

    def __len__(self):
        return self.nkpts_ibz

    def build(self, space_group_symmetry=True, time_reversal_symmetry=True,
              symmorphic=True, *args, **kwargs):
        symm.Symmetry.build(self, space_group_symmetry, symmorphic, *args, **kwargs)
        if not getattr(self.cell, '_built', None): return

        self.time_reversal = time_reversal_symmetry and not self.has_inversion
        self.kpts_scaled_ibz = self.kpts_scaled = self.cell.get_scaled_kpts(self.kpts)
        self.make_kpts_ibz()
        self.dump_info()
        return self

    def dump_info(self):
        if self.verbose >= logger.INFO:
            logger.info(self, 'time reversal: %s', self.time_reversal)
            logger.info(self, 'k-points in IBZ                           weights')
            ibzk = self.kpts_scaled_ibz
            for k in range(self.nkpts_ibz):
                logger.info(self, '%d:  %11.8f, %11.8f, %11.8f    %d/%d',
                            k, ibzk[k][0], ibzk[k][1], ibzk[k][2],
                            np.floor(self.weights_ibz[k]*self.nkpts), self.nkpts)

    def make_gdf_kptij_lst_jk(self):
        '''
        Build GDF k-point-pair list for get_jk
        All combinations:
            k_ibz != k_bz
            k_bz  == k_bz
        '''
        kptij_lst = [(self.kpts[i], self.kpts[i]) for i in range(self.nkpts)]
        for i in range(self.nkpts_ibz):
            ki = self.kpts_ibz[i]
            where = member(ki, self.kpts)
            for j in range(self.nkpts):
                kj = self.kpts[j]
                if j not in where:
                    kptij_lst.extend([(ki,kj)])
        kptij_lst = np.asarray(kptij_lst)
        return kptij_lst

    def loop_ktuples(self, ibz2bz, ntuple):
        nkpts = self.nkpts
        for k in ibz2bz:
            res = []
            for i in range(ntuple-1, -1, -1):
                ki = k // nkpts**(i)
                k = k - ki * nkpts**(i)
                res.append(ki)
            yield tuple(res)

    make_kpts_ibz = make_kpts_ibz
    make_ktuples_ibz = make_ktuples_ibz
    make_k4_ibz = make_k4_ibz
    loop_ktuples = loop_ktuples
    symmetrize_density = symmetrize_density
    symmetrize_wavefunction = symmetrize_wavefunction
    transform_mo_coeff = transform_mo_coeff
    transform_single_mo_coeff = transform_single_mo_coeff
    transform_dm = transform_dm
    transform_mo_energy = transform_mo_energy
    transform_mo_occ = transform_mo_occ
    check_mo_occ_symmetry = check_mo_occ_symmetry
    transform_fock = transform_fock
    transform_1e_operator = transform_1e_operator
    dm_at_ref_cell = dm_at_ref_cell

if __name__ == "__main__":
    import numpy
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = """
        Si  0.0 0.0 0.0
        Si  1.3467560987 1.3467560987 1.3467560987
    """
    cell.a = [[0.0, 2.6935121974, 2.6935121974],
              [2.6935121974, 0.0, 2.6935121974],
              [2.6935121974, 2.6935121974, 0.0]]
    cell.verbose = 4
    cell.build()
    nk = [3,3,3]
    kpts_bz = cell.make_kpts(nk)
    kpts0 = cell.make_kpts(nk, space_group_symmetry=True, time_reversal_symmetry=True)
    kpts1 = KPoints(cell, kpts_bz).build(space_group_symmetry=True, time_reversal_symmetry=True)
    print(numpy.allclose(kpts0.kpts_ibz, kpts1.kpts_ibz))

    kpts = KPoints()
    print(kpts.kpts)
