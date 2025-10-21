#!/usr/bin/env python
# Copyright 2020-2023 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
import numpy as np
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.symm import symmetry as symm
from pyscf.pbc.symm.group import PGElement, PointGroup, Representation
from pyscf.pbc.lib.kpts_helper import member, round_to_fbz, KPT_DIFF_TOL
from numpy.linalg import inv

libpbc = lib.load_library('libpbc')

def make_kpts_ibz(kpts, tol=KPT_DIFF_TOL):
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

    bz2bz_ks = map_k_points_fast(kpts.kpts_scaled, op_rot, tol)
    kpts.k2opk = bz2bz_ks.copy()
    if -1 in bz2bz_ks:
        bz2bz_ks[:, np.unique(np.where(bz2bz_ks == -1)[1])] = -1
        if kpts.verbose >= logger.WARN:
            logger.warn(kpts, 'k-points have lower symmetry than lattice.')

    bz2bz_k = -np.ones(nkpts+1, dtype=int)
    ibz2bz_k = []
    for k in range(nkpts-1, -1, -1):
        if bz2bz_k[k] == -1:
            # Note:, bz2bz_ks[k] has duplicated index
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
            if (np.absolute(diff) < tol).all():
                kpts.time_reversal_symm_bz[k] = io // nop
                kpts.stars_ops_bz[k] = io % nop
                break

    for i in range(kpts.nkpts_ibz):
        idx = np.where(kpts.bz2ibz == i)[0]
        kpts.stars.append(idx)
        kpts.stars_ops.append(kpts.stars_ops_bz[idx])

    little_cogroup_ops = []
    for ki_ibz in range(kpts.nkpts_ibz):
        ki = kpts.ibz2bz[ki_ibz]
        ops_id = np.where(kpts.k2opk[ki] == ki)[0]
        little_cogroup_ops.append(ops_id)
    kpts.little_cogroup_ops = little_cogroup_ops

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
        ibz2bz : (nibz,) ndarray of int
            Mapping table from IBZ to full BZ.
        weight_ibz : (nibz,) ndarray of int
            Weights of each k-point tuple in the IBZ.
        bz2ibz : (nkpts**ntuple,) ndarray of int
            Mapping table from full BZ to IBZ.
        stars : list of (nibz,) ndarrays of int
            Similar as :attr:`.stars`.
        stars_ops : list of (nibz,) ndarrays of int
            Similar as :attr:`.stars_ops`.
        stars_ops_bz : (nkpts**ntuple,) ndarray of int
            Similar as :attr:`.stars_ops_bz`.
    """
    if kpts_scaled is not None:
        cell = kpts.cell
        op_rot = np.asarray([op.a2b(cell).rot for op in kpts.ops])
        if kpts.time_reversal:
            op_rot = np.concatenate([op_rot, -op_rot])
        kt2opkt = map_kpts_tuples(kpts_scaled, op_rot, ntuple=ntuple, tol=tol)
        kt2opkt[:, np.unique(np.where(kt2opkt == -1)[1])] = -1
        nktuple = kt2opkt.shape[0]
    else:
        k2opk = kpts.k2opk
        nkpts, nop = k2opk.shape
        nktuple = nkpts ** ntuple
        kt2opkt_T = np.empty([nop, nktuple], dtype=np.int64)
        for iop in range(nop):
            if -1 in k2opk[:,iop]:
                kt2opkt_T[iop,:] = -1
            else:
                tmp = lib.cartesian_prod([k2opk[:,iop],] * ntuple)
                kt2opkt_T[iop] = lib.inv_base_repr_int(tmp, nkpts)
                tmp = None
        kt2opkt = kt2opkt_T.T

    if kpts.verbose >= logger.INFO:
        logger.info(kpts, 'Number of k-point %d-tuples: %d', ntuple, nktuple)

    bz2bz = -np.ones(nktuple+1, dtype=np.int64)
    ibz2bz = []
    stars = []
    stars_ops = []
    stars_ops_bz = np.empty(nktuple, dtype=np.int32)
    for k in range(nktuple-1, -1, -1):
        if bz2bz[k] == -1:
            bz2bz[kt2opkt[k]] = k
            ibz2bz.append(k)
            k_idx, op_idx = np.unique(kt2opkt[k], return_index=True)
            if k_idx[0] == -1:
                k_idx = k_idx[1:]
                op_idx = op_idx[1:]
            stars.append(k_idx)
            stars_ops.append(op_idx)
            stars_ops_bz[k_idx] = op_idx

    kt2opkt =None

    ibz2bz = np.array(ibz2bz, dtype=np.int64)[::-1]
    if kpts.verbose >= logger.INFO:
        logger.info(kpts, f'Number of k {ntuple}-tuples in IBZ: {len(ibz2bz)}')

    bz2bz = bz2bz[:-1]
    bz2ibz = np.empty(nktuple, dtype=np.int64)
    bz2ibz[ibz2bz] = np.arange(len(ibz2bz))
    bz2ibz = bz2ibz[bz2bz]

    stars = stars[::-1]
    stars_ops = stars_ops[::-1]
    weight_ibz = np.bincount(bz2ibz) * (1.0 / nktuple)
    return ibz2bz, weight_ibz, bz2ibz, stars, stars_ops, stars_ops_bz

def make_k4_ibz(kpts, sym='s1', return_ops=False):
    #physicist's notation
    ibz2bz, weight, bz2ibz, stars, stars_ops, stars_ops_bz = \
            kpts.make_ktuples_ibz(ntuple=3)
    kconserv = kpts.get_kconserv()

    kija = kpts.index_to_ktuple(ibz2bz, 3)
    kb = kconserv[kija[:,0], kija[:,2], kija[:,1]]
    k4 = np.concatenate((kija, kb[:,None]), axis=1)

    if sym == "s1":
        if return_ops:
            return k4, weight, bz2ibz, ibz2bz, stars_ops, stars_ops_bz
        else:
            return k4, weight, bz2ibz
    elif sym == "s2" or sym == "s4":
        ibz2ibz_s2 = np.arange(len(k4))
        k4_s2 = []
        weight_s2 = []
        for i, kijab in enumerate(k4):
            ki, kj, ka, kb = kijab
            k = list(kijab)
            k_sym = [kj, ki, kb, ka] #interchange dummy indices
            if k not in k4_s2 and k_sym not in k4_s2:
                k4_s2.append(k)
                ibz2ibz_s2[i] = len(k4_s2) - 1
                w = weight[i]
                if k != k_sym:
                    k_sym_in_k4, idx = lib.isin_1d(k_sym, k4, True)
                    if k_sym_in_k4:
                        ibz2ibz_s2[idx] = ibz2ibz_s2[i]
                        w += weight[idx]
                weight_s2.append(w)
        #refine s2 symmetry
        k4_s2_refine = []
        weight_s2_refine = []
        skip = np.zeros([len(k4_s2)], dtype=int)
        ibz_s22ibz_s2_refine = np.arange(len(k4_s2))
        for i, k in enumerate(k4_s2):
            if skip[i]:
                continue
            ki, kj, ka, kb = k
            k_sym = [kj, ki, kb, ka]
            if ki == kj and ka == kb:
                k4_s2_refine.append(k)
                ibz_s22ibz_s2_refine[i] = len(k4_s2_refine) - 1
                weight_s2_refine.append(weight_s2[i])
                continue
            idx_sym = None
            for j in range(i+1, len(k4_s2)):
                if skip[j]:
                    continue
                k_tmp = k4_s2[j]
                if ki in k_tmp and kj in k_tmp and ka in k_tmp and kb in k_tmp:
                    _, idx = lib.isin_1d(k_tmp, k4, True)
                    for kii,kjj,kaa in kpts.loop_ktuples(stars[idx], 3):
                        kbb = kconserv[kii,kaa,kjj]
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
                bz2ibz_s2[i] = np.where(idx == ibz_s22ibz_s2_refine[ibz2ibz_s2[bz2ibz[i]]])[0][0]
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
    return map_kpts_tuples(kpts_scaled.reshape(len(kpts_scaled),-1,3),
                           ops, ntuple=1, tol=tol)

def map_kpts_tuples(kpts_scaled, ops, ntuple=2, tol=KPT_DIFF_TOL):
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
            K-points differ less than ``tol`` are considered
            as the same. Default is 1e-6.

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
        op_kpts_scaled = np.einsum('kix,xy->kiy', kpts_scaled, op.T)
        op_kpts_scaled = op_kpts_scaled.reshape((nkpts,-1))

        k_opk = np.concatenate([kpts_scaled.reshape((nkpts,-1)), op_kpts_scaled])
        k_opk = round_to_fbz(k_opk, tol=tol)
        order = np.lexsort(k_opk.T)
        k_opk = k_opk[order]
        diff = np.diff(k_opk, axis=0)
        equivalent_pairs = np.array((diff == 0).all(1), dtype=bool)

        maps = np.array([order[:-1][equivalent_pairs],
                         order[ 1:][equivalent_pairs]])

        assert (maps[0] < nkpts).all()
        assert (maps[1] >= nkpts).all()
        bz2bz_ks[maps[1] - nkpts, s] = maps[0]
    return bz2bz_ks

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
    is_complex = np.iscomplexobj(psiR_k.dtype)
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
    def _transform(kpts, mo_coeff_ibz, k):
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

    if isinstance(mo_coeff_ibz[0][0], np.ndarray) and mo_coeff_ibz[0][0].ndim == 2:
        # KUHF
        if kpts.nkpts_ibz != len(mo_coeff_ibz[0]):
            raise KeyError('Shape of mo_coeff does not match the number of IBZ k-points: '
                           f'{len(mo_coeff_ibz[0])} vs {kpts.nkpts_ibz}.')
        mos = [[_transform(kpts, mo_coeff_ibz[0], k) for k in range(kpts.nkpts)],
               [_transform(kpts, mo_coeff_ibz[1], k) for k in range(kpts.nkpts)]]
    else:
        if kpts.nkpts_ibz != len(mo_coeff_ibz):
            raise KeyError('Shape of mo_coeff does not match the number of IBZ k-points: '
                           f'{len(mo_coeff_ibz)} vs {kpts.nkpts_ibz}.')
        mos = [_transform(kpts, mo_coeff_ibz, k) for k in range(kpts.nkpts)]
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

    if is_uhf:
        if kpts.nkpts_ibz != len(mo_occ_ibz[0]):
            raise KeyError('Shape of mo_occ does not match the number of IBZ k-points: '
                           f'{len(mo_occ_ibz[0])} vs {kpts.nkpts_ibz}.')
    else:
        if kpts.nkpts_ibz != len(mo_occ_ibz):
            raise KeyError('Shape of mo_occ does not match the number of IBZ k-points: '
                           f'{len(mo_occ_ibz)} vs {kpts.nkpts_ibz}.')

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

    def _transform(dm_ibz, k):
        ibz_k_idx = kpts.bz2ibz[k]
        ibz_kpt_scaled = kpts.kpts_scaled_ibz[ibz_k_idx]
        iop = kpts.stars_ops_bz[k]
        op = kpts.ops[iop]
        time_reversal = kpts.time_reversal_symm_bz[k]
        if op.is_eye:
            dm_bz = dm_ibz[ibz_k_idx]
        else:
            dm_bz = symm.transform_dm(kpts.cell, ibz_kpt_scaled,
                                      dm_ibz[ibz_k_idx], op, kpts.Dmats[iop])
        if time_reversal:
            dm_bz = dm_bz.conj()
        return dm_bz

    dms = []
    is_uhf = False
    if (isinstance(dm_ibz, np.ndarray) and dm_ibz.ndim == 4) or \
       (isinstance(dm_ibz[0][0], np.ndarray) and dm_ibz[0][0].ndim == 2):
        is_uhf = True
        dms = [[],[]]

    if is_uhf:
        if kpts.nkpts_ibz != len(dm_ibz[0]):
            raise KeyError('Shape of the input density matrix does not '
                           'match the number of IBZ k-points: '
                           f'{len(dm_ibz[0])} vs {kpts.nkpts_ibz}.')
        for k in range(kpts.nkpts):
            dms[0].append(_transform(dm_ibz[0], k))
            dms[1].append(_transform(dm_ibz[1], k))
        nkpts = len(dms[0])
        nao = dms[0][0].shape[0]
        dms = lib.asarray(dms).reshape(2,nkpts,nao,nao)
    else:
        if kpts.nkpts_ibz != len(dm_ibz):
            raise KeyError('Shape of the input density matrix does not '
                           'match the number of IBZ k-points: '
                           f'{len(dm_ibz)} vs {kpts.nkpts_ibz}.')
        for k in range(kpts.nkpts):
            dms.append(_transform(dm_ibz, k))
        dms = lib.asarray(dms)

    if getattr(dm_ibz, "mo_coeff", None) is not None:
        dms = lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)
    return dms

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

def get_rotation_mat_for_mos(kpts, mo_coeff, ovlp, k1, k2, ops_id=None):
    '''Rotation matrices for rotating MO[k1] to MO[k2].

    Args:
        kpts : :class:`KPoints` instance
            K-point object.
        mo_coeff : array
            MO coefficients.
        ovlp : array
            Overlap matrix in AOs.
        k1 : array like
            Indices of the original k-points in the BZ.
        k2 : array like
            Indices of the target k-points in the BZ.
        ops_id : list, optional
            Indices of space group operations.
            If not given, all the rotations are considered.

    Returns:
        out : list
            Rotation matrices.
    '''
    from pyscf.pbc.symm.symmetry import _get_rotation_mat
    cell = kpts.cell
    nk = len(k1)
    assert nk == len(k2)
    if ops_id is not None:
        assert nk == len(ops_id)

    out = []
    for k, (k_orig, k_target) in enumerate(zip(k1, k2)):
        mats = []
        if ops_id is None:
            ids = np.arange(kpts.nop)
        else:
            ids = np.asarray(ops_id[k]).reshape(-1)
        for iop in ids:
            mat_ao = _get_rotation_mat(cell, kpts.kpts_scaled[k_orig],
                                       mo_coeff[k_orig], kpts.ops[iop],
                                       kpts.Dmats[iop])
            mat = reduce(np.dot,(mo_coeff[k_orig].conj().T, ovlp[k_orig],
                                 mat_ao.conj().T, mo_coeff[k_target]))
            mats.append(mat)
        out.append(np.asarray(mats))
    return out


def make_kpts(cell, kpts=np.zeros((1,3)),
              space_group_symmetry=False, time_reversal_symmetry=False,
              **kwargs):
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

    Examples
    --------
    >>> cell = gto.M(
    ...     atom = '''He 0. 0. 0.''',
    ...     a = numpy.eye(3)*2.0).build()
    >>> kpts = make_kpts(cell,
    ...                  numpy.array([[0.,0.,0.],
    ...                               [0.5,0.,0.],
    ...                               [0.,0.5,0.],
    ...                               [0.,0.,0.5]]),
    ...                  space_group_symmetry=True)
    >>> print(kpts.kpts_ibz)
    [[0.  0.  0. ]
     [0.  0.  0.5]]
    """
    if isinstance(kpts, KPoints):
        return kpts.build(space_group_symmetry,
                          time_reversal_symmetry,
                          **kwargs)
    else:
        kpts_symm = KPoints(cell, kpts)
        kpts_symm.build(space_group_symmetry,
                        time_reversal_symmetry,
                        **kwargs)
        return kpts_symm

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
        self.little_cogroup_ops = []

        #private variables
        self._nkpts = len(self.kpts)
        self._nkpts_ibz = len(self.kpts_ibz)
        self._addition_table = None
        self._inverse_table = None
        self._copgs = None
        self._copg_ops_map_ibz2bz = None

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

    def build(self, space_group_symmetry=False, time_reversal_symmetry=False,
              *args, **kwargs):
        if space_group_symmetry:
            _lattice_symm = getattr(self.cell, 'lattice_symmetry', None)
            if isinstance(_lattice_symm, symm.Symmetry):
                self.__dict__.update(_lattice_symm.__dict__)
        if not self._built:
            symm.Symmetry.build(self, space_group_symmetry, *args, **kwargs)
        if not getattr(self.cell, '_built', None):
            return self

        self.time_reversal = time_reversal_symmetry and not self.has_inversion
        self.kpts_scaled_ibz = self.kpts_scaled = self.cell.get_scaled_kpts(self.kpts)
        self.make_kpts_ibz()
        self.dump_info()
        return self

    def reset(self, cell=None):
        '''
        Update the absolute k-points of an object wrt the input cell,
        while preserving the same fractional (scaled) k-point coordinates.
        '''
        if cell is None:
            return self

        self.cell = cell
        self._built = False
        self.kpts = self.kpts_ibz = cell.get_abs_kpts(self.kpts_scaled)
        self.build(space_group_symmetry=cell.space_group_symmetry,
                   time_reversal_symmetry=self.time_reversal)
        return self

    def dump_info(self):
        if self.verbose >= logger.INFO:
            logger.info(self, 'time reversal: %s', self.time_reversal)
            logger.info(self, 'k-points in IBZ                         weights')
            ibzk = self.kpts_scaled_ibz
            for k in range(self.nkpts_ibz):
                logger.info(self, '%3d: %9.6f, %9.6f, %9.6f    %d/%d',
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
        for k in ibz2bz:
            res = self.index_to_ktuple(k, ntuple)
            yield tuple(res)

    def ktuple_to_index(self, kk):
        return lib.inv_base_repr_int(kk, self.nkpts)

    def index_to_ktuple(self, k, ntuple):
        return lib.base_repr_int(k, self.nkpts, ntuple)

    @property
    def addition_table(self, tol=KPT_DIFF_TOL):
        if self._addition_table is None:
            kptsi = kptsj = self.kpts_scaled
            kptsij = kptsi[:,None,:] + kptsj[None,:,:]
            diff = kptsij[:,:,None,:] - self.kpts_scaled[None,None,:,:]
            diff = round_to_fbz(diff, tol=tol)
            idx = np.where(abs(diff).max(axis=-1) < tol)

            nk = self.nkpts
            table= -np.ones((nk,nk), dtype=np.int32)
            table[idx[0],idx[1]] = idx[2]
            assert (table > -1).all()
            self._addition_table = table
        return self._addition_table

    @property
    def inverse_table(self, tol=KPT_DIFF_TOL):
        if self._inverse_table is None:
            diff = -self.kpts_scaled[:,None,:] - self.kpts_scaled[None,:,:]
            diff = round_to_fbz(diff, tol=tol)
            idx = np.where(abs(diff).max(axis=-1) < tol)

            table = -np.ones((self.nkpts,), dtype=np.int32)
            table[idx[0]] = idx[1]
            assert (table > -1).all()
            self._inverse_table = table
        return self._inverse_table

    def get_kconserv(self):
        '''Equivalent to `kpts_helper.get_kconserv`,
           but with better performance.
        '''
        add_tab = self.addition_table
        inv_tab = self.inverse_table
        kconserv = add_tab[add_tab[:, inv_tab[:]],:]
        return kconserv

    def little_cogroups(self, return_indices=True):
        if self._copgs is not None:
            return self._copgs, self._copg_ops_map_ibz2bz
        copgs = []
        indices = []
        for ki in range(self.nkpts):
            ki_ibz = self.bz2ibz[ki]
            ops_ibz = self.little_cogroup_ops[ki_ibz]
            elements = np.sort([PGElement(self.ops[i].rot) for i in ops_ibz])
            iop = self.stars_ops_bz[ki]
            op_i = PGElement(self.ops[iop].rot)
            elements_i = np.asarray([op_i @ g @ op_i.inv() for g in elements])
            idx = np.argsort(elements_i)
            indices.append(idx)
            copgs.append(PointGroup(elements_i[idx]))
        self._copgs, self._copg_ops_map_ibz2bz = copgs, indices
        return self._copgs, self._copg_ops_map_ibz2bz

    def little_cogroup_rep(self, ki, ir):
        copgs, indices = self.little_cogroups()
        ki_ibz = self.bz2ibz[ki]
        pg_ibz = copgs[self.ibz2bz[ki_ibz]]
        chi = pg_ibz.get_irrep_chi(ir)
        chi_ki = chi[indices[ki]]
        return Representation(copgs[ki], chi=chi_ki)

    make_kpts_ibz = make_kpts_ibz
    make_ktuples_ibz = make_ktuples_ibz
    make_k4_ibz = make_k4_ibz
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
    get_rotation_mat_for_mos = get_rotation_mat_for_mos


class MORotationMatrix:
    '''
    The class holding the rotation matrices that transform
    MOs from one k-point to another.
    '''
    def __init__(self, kpts, mo_coeff, ovlp, nocc, nmo):
        self.kpts = kpts
        self.mo_coeff = mo_coeff
        self.ovlp = ovlp
        assert nmo >= nocc
        self.nocc = nocc
        self.nmo = nmo
        self.oo = None
        self.vv = None

    def build(self):
        kpts = self.kpts
        mo_coeff = self.mo_coeff
        ovlp = self.ovlp
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        orb_occ = [mo[:,:nocc] for mo in mo_coeff]
        orb_vir = [mo[:,nocc:] for mo in mo_coeff]
        nkpts = kpts.nkpts
        nop = kpts.nop
        k2opk = kpts.k2opk

        self.oo = []
        self.vv = []
        for ki in range(nkpts):
            k1 = np.repeat(ki, nop)
            k2 = k2opk[ki]
            rot_oo = kpts.get_rotation_mat_for_mos(orb_occ, ovlp, k1, k2,
                                                   ops_id=np.arange(nop))
            rot_oo = np.asarray(rot_oo).reshape(nop,nocc,nocc)
            rot_vv = kpts.get_rotation_mat_for_mos(orb_vir, ovlp, k1, k2,
                                                   ops_id=np.arange(nop))
            rot_vv = np.asarray(rot_vv).reshape(nop,nvir,nvir)
            self.oo.append(rot_oo)
            self.vv.append(rot_vv)

        self.oo = np.asarray(self.oo)
        self.vv = np.asarray(self.vv)
        return self


class KQuartets:
    '''
    The class holding the symmetry relations between k-quartets.
    '''
    def __init__(self, kpts):
        assert isinstance(kpts, KPoints)
        self.kpts = kpts
        self.kqrts_ibz = None
        self.weights_ibz = None
        self.ibz2bz = None
        self.bz2ibz = None
        self.stars_ops = None
        self.stars_ops_bz = None
        self._kqrts_stab = None
        self._ops_stab = None

    def build(self):
        kpts = self.kpts
        (self.kqrts_ibz, self.weights_ibz, self.bz2ibz,
         self.ibz2bz, self.stars_ops, self.stars_ops_bz) = \
                kpts.make_k4_ibz(sym='s1', return_ops=True)

        # Sanity check
        #assert -1 not in self.stars_ops_bz
        #for ki, ki_ibz in enumerate(self.bz2ibz):
        #    iop = self.stars_ops_bz[ki]
        #    iops = self.stars_ops[ki_ibz]
        #    assert iop in iops
        #    i,j,a,b = self.kqrts_ibz[ki_ibz]
        #    k,l,c = kpts.k2opk[[i,j,a,], iop]
        #    kk, ll, cc = lib.base_repr_int(ki, kpts.nkpts, 3)
        #    assert (k,l,c) == (kk,ll,cc)
        return self

    def cache_stabilizer(self):
        self._kqrts_stab = []
        self._ops_stab = []
        for i, kq in enumerate(self.kqrts_ibz):
            op_group = self.stars_ops[i]
            ks = self.kpts.k2opk[kq[0], op_group]
            idx = np.where(ks == kq[0])[0]
            op_group_small = op_group[idx]
            klcd = self.kpts.k2opk[kq[:,None], op_group_small].T
            self._kqrts_stab.append(klcd)
            self._ops_stab.append(op_group_small)

    def loop_stabilizer(self, index):
        if self._kqrts_stab is None or self._ops_stab is None:
            self.cache_stabilizer()
        yield from zip(self._kqrts_stab[index], self._ops_stab[index])
