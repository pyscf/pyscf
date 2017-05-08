#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
JK with discrete Fourier transformation
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.dft import numint
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpt_misc import is_zero, gamma_point


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    gs = mydf.gs

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, gs=gs)
    ngs = len(coulG)

    vR = rhoR = np.zeros((nset,ngs))
    for k, aoR in mydf.aoR_loop(gs, kpts):
        for i in range(nset):
            rhoR[i] += numint.eval_rho(cell, aoR, dms[i,k])
    for i in range(nset):
        rhoR[i] *= 1./nkpts
        rhoG = tools.fft(rhoR[i], gs)
        vG = coulG * rhoG
        vR[i] = tools.ifft(vG, gs).real

    kpts_band, single_kpt_band = _format_kpts_band(kpts_band, kpts)
    nband = len(kpts_band)
    weight = cell.vol / ngs
    if gamma_point(kpts_band):
        vj_kpts = np.empty((nset,nband,nao,nao))
    else:
        vj_kpts = np.empty((nset,nband,nao,nao), dtype=np.complex128)
    for k, aoR in mydf.aoR_loop(gs, kpts_band):
        for i in range(nset):
            vj_kpts[i,k] = weight * lib.dot(aoR.T.conj()*vR[i], aoR)

    return _format_jks(vj_kpts, dm_kpts, kpts_band, kpts, single_kpt_band)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    gs = mydf.gs
    coords = cell.gen_uniform_grids(gs)
    ngs = coords.shape[0]

    kpts = np.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngs)

    input_band = kpts_band
    kpts_band, single_kpt_band = _format_kpts_band(kpts_band, kpts)
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    if input_band is None:
        ao_kpts = mydf._numint.eval_ao(cell, coords, kpts, non0tab=mydf.non0tab)
        for k2, ao_k2 in enumerate(ao_kpts):
            kpt2 = kpts[k2]
            aoR_dms = [lib.dot(ao_k2, dms[i,k2]) for i in range(nset)]
            for k1, ao_k1 in enumerate(ao_kpts):
                kpt1 = kpts_band[k1]
                vkR_k1k2 = get_vkR(mydf, cell, ao_k1, ao_k2, kpt1, kpt2,
                                   coords, gs, exxdiv)
                for i in range(nset):
                    tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dms[i])
                    vk_kpts[i,k1] += weight * lib.dot(ao_k1.T.conj(), tmp_Rq)
            vkR_k1k2 = aoR_dms = tmp_Rq = None
    else:
        for k2, ao_k2 in mydf.aoR_loop(gs, kpts):
            kpt2 = kpts[k2]
            aoR_dms = [lib.dot(ao_k2, dms[i,k2]) for i in range(nset)]
            for k1, ao_k1 in mydf.aoR_loop(gs, kpts_band):
                kpt1 = kpts_band[k1]
                vkR_k1k2 = get_vkR(mydf, cell, ao_k1, ao_k2, kpt1, kpt2,
                                   coords, gs, exxdiv)
                for i in range(nset):
                    tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dms[i])
                    vk_kpts[i,k1] += weight * lib.dot(ao_k1.T.conj(), tmp_Rq)
            vkR_k1k2 = aoR_dms = tmp_Rq = None

    return _format_jks(vk_kpts, dm_kpts, kpts_band, kpts, single_kpt_band)


def get_jk(mydf, dm, hermi=1, kpt=np.zeros(3), kpt_band=None,
           with_j=True, with_k=True, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpt_band : (3,) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = np.asarray(dm, order='C')
    vj = vk = None
    if with_j:
        vj = get_j(mydf, dm, hermi, kpt, kpt_band)
    if with_k:
        vk = get_k(mydf, dm, hermi, kpt, kpt_band, exxdiv)
    return vj, vk

def get_j(mydf, dm, hermi=1, kpt=np.zeros(3), kpt_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpt_band : (3,) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = np.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vj = get_j_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpt_band)
    return vj.reshape(dm.shape)

def get_k(mydf, dm, hermi=1, kpt=np.zeros(3), kpt_band=None, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpt_band : (3,) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = np.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vk = get_k_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpt_band, exxdiv)
    return vk.reshape(dm.shape)


def get_vkR(mydf, cell, aoR_k1, aoR_k2, kpt1, kpt2, coords, gs, exxdiv):
    '''Get the real-space 2-index "exchange" potential V_{i,k1; j,k2}(r)
    where {i,k1} = exp^{i k1 r) |i> , {j,k2} = exp^{-i k2 r) <j|

    Kwargs:
        kpt1, kpt2 : (3,) ndarray
            The sampled k-points; may be required for G=0 correction.

    Returns:
        vR : (ngs, nao, nao) ndarray
            The real-space "exchange" potential at every grid point, for all
            AO pairs.

    Note:
        This is essentially a density-fitting or resolution-of-the-identity.
        The returned object is of size ngs*nao**2
    '''
    ngs, nao = aoR_k1.shape
    expmikr = np.exp(-1j*np.dot(kpt1-kpt2,coords.T))
    mydf.exxdiv = exxdiv
    coulG = tools.get_coulG(cell, kpt1-kpt2, True, mydf, gs)

    aoR_k1 = np.asarray(aoR_k1.T, order='C')
    aoR_k2 = np.asarray(aoR_k2.T, order='C')
    vR = np.empty((nao,nao,ngs), dtype=np.complex128)
    for i in range(nao):
        rhoR = aoR_k1 * aoR_k2[i].conj()
        rhoG = tools.fftk(rhoR, gs, expmikr)
        vG = rhoG * coulG
        vR[:,i] = tools.ifftk(vG, gs, expmikr.conj())
    vR = vR.transpose(2,0,1)

    if aoR_k1.dtype == np.double and aoR_k2.dtype == np.double:
        return vR.real
    else:
        return vR

