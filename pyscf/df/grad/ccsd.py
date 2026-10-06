#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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

'''
Density-fitting CCSD analytical nuclear gradients

Copied from pyscf.grad.ccsd.py.  The four-index ERI derivatives are replaced by
the derivatives of the three-index integrals and of the fitting metric, so that
the gradient differentiates the DF energy expression that the calculation
actually minimised.
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf import df
from pyscf.lib import logger
from pyscf.cc import ccsd_rdm
from pyscf.ao2mo.outcore import balance_partition
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad.mp2 import _index_frozen_active, has_frozen_orbitals
from pyscf.df.grad.rhf import _int3c_wrapper, _gen_metric_solver


def _auxmol(mycc):
    with_df = getattr(mycc, 'with_df', None)
    if with_df is None:
        with_df = mycc._scf.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    return with_df, auxmol

def _pair_weights(nao):
    # Undo the double counting of the off-diagonal elements of a symmetric
    # matrix stored in the lower-triangular packed layout.
    w = numpy.ones(nao*(nao+1)//2)
    diag = numpy.arange(nao)
    w[diag*(diag+1)//2 + diag] = .5
    return w

def solve_df_rdm2(cc_grad, fdm2, max_memory=None):
    '''Solve (P|Q) d_Quv = (P|kl) Gamma_uv,kl for the AO-basis DF 2-PDM.

    Args:
        cc_grad : DF-CCSD gradients method object
        fdm2 : HDF5 group holding the AO 2-PDM 'dm2' written by
            :func:`pyscf.grad.ccsd._rdm2_mo2ao`

    Returns:
        (int3c, dferi, dfdm2), each of shape (naux, nao_pair) in the
        lower-triangular AO pair layout:
            int3c[P,uv] = (P|uv)
            dferi[P,uv] = (P|Q)^-1 (Q|uv)
            dfdm2[P,uv] = (P|Q)^-1 (Q|kl) Gamma_uv,kl
    '''
    mol = cc_grad.mol
    auxmol = _auxmol(cc_grad.base)[1]
    nao, nbas = mol.nao, mol.nbas
    naux = auxmol.nao
    nao_pair = nao * (nao+1) // 2
    aux_loc = auxmol.ao_loc
    if max_memory is None:
        max_memory = cc_grad.max_memory

    int3c = numpy.empty((naux, nao_pair))
    get_int3c = _int3c_wrapper(mol, auxmol, 'int3c2e', 's2ij')
    blksize = max(1, min(int(max_memory*.2e6/8/max(nao_pair, 1)), naux))
    for shl0, shl1, nL in balance_partition(aux_loc, blksize):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c[p0:p1] = get_int3c((0, nbas, 0, nbas, shl0, shl1)).T

    solve_j2c = _gen_metric_solver(auxmol.intor('int2c2e', aosym='s1'))
    dferi = solve_j2c(int3c.copy())

    dm2 = fdm2['dm2']
    dfdm2 = numpy.empty((naux, nao_pair))
    dferi_w = dferi * _pair_weights(nao)
    blksize = max(1, min(int(max_memory*.2e6/8/max(naux, 1)), nao_pair))
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        dfdm2[:,p0:p1] = lib.dot(dferi_w, numpy.asarray(dm2[p0:p1]).T)
    return int3c, dferi, dfdm2

def _contract_dfdm2(cc_grad, fdm2, atmlst, max_memory=None):
    '''DF counterpart of the four-index 2-PDM contraction in
    :func:`pyscf.grad.ccsd.grad_elec`.

    Returns:
        de : ndarray of shape (len(atmlst),3)
            d/dX of 0.5 sum_pqrs Gamma_pq,rs (pq|rs)_DF
        Imat : ndarray of shape (nao,nao)
            Imat[p,q] = sum_{i,kl} (ip|kl)_DF Gamma_iq,kl
    '''
    mol = cc_grad.mol
    auxmol = _auxmol(cc_grad.base)[1]
    nao, nbas = mol.nao, mol.nbas
    naux = auxmol.nao
    nao_pair = nao * (nao+1) // 2
    aux_loc = auxmol.ao_loc
    wpair = _pair_weights(nao)
    if max_memory is None:
        max_memory = cc_grad.max_memory

    int3c, dferi, dfdm2 = solve_df_rdm2(cc_grad, fdm2, max_memory)

    Imat = numpy.zeros((nao, nao))
    de = numpy.zeros((len(atmlst), 3))
    dvec = numpy.zeros((3, nao))
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    blksize = max(1, min(int(max_memory*.3e6/8/max(nao**2*6, 1)), naux))
    for shl0, shl1, nL in balance_partition(aux_loc, blksize):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        d_blk = lib.unpack_tril(dfdm2[p0:p1])
        c_blk = lib.unpack_tril(int3c[p0:p1])
        Imat += lib.dot(c_blk.reshape(-1,nao).T, d_blk.reshape(-1,nao))
        c_blk = None
        int3c_ip1 = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1))
        dvec -= numpy.einsum('xijp,ijp->xi', int3c_ip1,
                             numpy.ascontiguousarray(d_blk.transpose(1,2,0))) * 2
        int3c_ip1 = d_blk = None

    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        de[k] += dvec[:,p0:p1].sum(axis=1)

    if cc_grad.auxbasis_response:
        aux_de = numpy.zeros((3, naux))
        dfdm2_w = dfdm2 * wpair
        get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's2ij')
        blksize = max(1, min(int(max_memory*.3e6/8/max(nao_pair*4, 1)), naux))
        for shl0, shl1, nL in balance_partition(aux_loc, blksize):
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            int3c_ip2 = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))
            aux_de[:,p0:p1] -= numpy.einsum('xyp,py->xp', int3c_ip2,
                                            dfdm2_w[p0:p1]) * 2
            int3c_ip2 = None
        metric = lib.dot(dferi*wpair, dfdm2.T) * 2
        aux_de += lib.einsum('xpq,pq->xp', auxmol.intor('int2c2e_ip1'), metric)
        auxslices = auxmol.aoslice_by_atom()
        for k, ia in enumerate(atmlst):
            p0, p1 = auxslices[ia,2:]
            de[k] += aux_de[:,p0:p1].sum(axis=1)
    return de, Imat


def grad_elec(cc_grad, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
              d1=None, d2=None, verbose=logger.INFO):
    mycc = cc_grad.base
    if eris is not None:
        if abs(eris.fock - numpy.diag(eris.fock.diagonal())).max() > 1e-3:
            raise RuntimeError('CCSD gradients does not support NHF (non-canonical HF)')

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2

    log = logger.new_logger(mycc, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    log.debug('Build ccsd rdm1 intermediates')
    if d1 is None:
        d1 = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    time1 = log.timer_debug1('rdm1 intermediates', *time0)
    log.debug('Build ccsd rdm2 intermediates')
    fdm2 = lib.H5TmpFile()
    if d2 is None:
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)
    time1 = log.timer_debug1('rdm2 intermediates', *time1)

    mol = cc_grad.mol
    mo_coeff = mycc.mo_coeff
    mo_energy = mycc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(mycc.mo_occ > 0)
    with_frozen = has_frozen_orbitals(mycc)
    OA, VA, OF, VF = _index_frozen_active(mycc.get_frozen_mask(), mycc.mo_occ)

    log.debug('symmetrized rdm2 and MO->AO transformation')
# Roughly, dm2*2 is computed in _rdm2_mo2ao
    mo_active = mo_coeff[:,numpy.hstack((OA,VA))]
    ccsd_grad._rdm2_mo2ao(mycc, d2, mo_active, fdm2)  # transform the active orbitals
    time1 = log.timer_debug1('MO->AO transformation', *time1)
    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

# 2pdm contracted with the derivatives of the three-index integrals and of the
# fitting metric
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    de, Imat = _contract_dfdm2(cc_grad, fdm2, atmlst, max_memory)
    time1 = log.timer_debug1('2e-part grad', *time1)

    Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mycc._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = numpy.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mycc._scf.get_veff(mycc.mol, dm1) * 2
    Xvo = reduce(numpy.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += ccsd_grad._response_dm1(mycc, Xvo, eris)
    time1 = log.timer_debug1('response_rdm1 intermediates', *time1)

    Imat[nocc:,:nocc] = Imat[:nocc,nocc:].T
    im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))
    time1 = log.timer_debug1('response_rdm1', *time1)

    log.debug('h1 and JK1')
    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = mycc._scf.nuc_grad_method()
    mf_grad.auxbasis_response = cc_grad.auxbasis_response
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf_s1occ = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1+dm1.T), p1))
    time1 = log.timer_debug1('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    dm1p = hf_dm1 + dm1*2
    dm1 += hf_dm1
    zeta += rhf_grad.make_rdm1e(mo_energy, mo_coeff, mycc.mo_occ)

    # The derivative of the separable part of the 2-PDM.  In the conventional
    # algorithm this is the vhf1 term; written through get_jk it picks up the
    # DF auxiliary-basis response for free.
    vj, vk = mf_grad.get_jk(mol, (hf_dm1, dm1p))
    vhf0 = vj[0] - vk[0] * .5
    vhfp = vj[1] - vk[1] * .5
    if cc_grad.auxbasis_response:
        vaux = vj.aux - vk.aux * .5
        de_aux = (vaux[0,1] + vaux[1,0]) * .5
    vj = vk = None
    time1 = log.timer_debug1('vhf1', *time1)

    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1])
# h[1] \dot DM, contribute to f1
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, dm1)
# -s[1]*e \dot DM,  contribute to f1
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
        de[k] -= numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1])
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhfp[:,p0:p1], hf_dm1[p0:p1])
        de[k] += numpy.einsum('xij,ij->x', vhf0[:,p0:p1], dm1p[p0:p1])
        if cc_grad.auxbasis_response:
            de[k] += de_aux[ia]

    log.timer('%s gradients' % mycc.__class__.__name__, *time0)
    return de


class Gradients(ccsd_grad.Gradients):
    '''Restricted density-fitting CCSD gradients'''

    _keys = {'with_df', 'auxbasis_response'}

    # Whether to include the response of the DF auxiliary basis
    auxbasis_response = True

    def __init__(self, mycc):
        # The correlation part of this assembly is density fitted, while the
        # HF reference part is delegated to mycc._scf (get_veff for the
        # Z-vector equation, nuc_grad_method for h1, S1 and the separable
        # 2-PDM).  Both halves have to differentiate the same energy
        # expression, so the reference must be a density-fitting SCF object.
        # dfccsd.RCCSD accepts a conventional-ERI mean field and builds its own
        # with_df, which would mix the two.
        assert isinstance(mycc._scf, df.df_jk._DFHF), (
            'DF-CCSD gradients require a density-fitting SCF reference. '
            'Use cc.CCSD(mf.density_fit()) rather than dfccsd.RCCSD(mf).')
        ccsd_grad.Gradients.__init__(self, mycc)
        self.with_df = getattr(mycc, 'with_df', None) or mycc._scf.with_df

    def check_sanity(self):
        ccsd_grad.Gradients.check_sanity(self)
        assert isinstance(self.base._scf, df.df_jk._DFHF)

    grad_elec = grad_elec

Grad = Gradients

from pyscf.cc import dfccsd
dfccsd.RCCSD.Gradients = lib.class_as_method(Gradients)
