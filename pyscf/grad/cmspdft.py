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
import numpy as np
from scipy import linalg
from pyscf import ao2mo, lib
from pyscf.lib import logger
from pyscf.mcscf import newton_casscf
import copy
from functools import reduce
from pyscf.mcpdft.cmspdft import coulomb_tensor

# TODO: docstring?
def diab_response (mc_grad, Lis, mo=None, ci=None, eris=None, **kwargs):
    '''Computes the Hessian-vector product of

    Q_a-a = 1/2 sum_I g_pqrs <I|p'q|I> <I|r's|I>

    where the vector is a vector of intermediate-state rotations and the
    external derivatives are with respect to orbital rotations and CI
    transfers.

    Args:
        mc_grad : object of class Gradients (CASSCF or CASCI)
        Lis : ndarray of shape (nroots*(nroots-1)/2,)
            Contains step vector for intermediate state rotations

    Kwargs:
        mo : ndarray of shape (nao,nmo)
            Contains MO coefficients
        ci : ndarray or list of length (nroots)
            Contains intermediate-state CI vectors
        eris : object of class ERIS (CASSCF or CASCI)
            Contains (true) ERIs in the MO basis

    Returns:
        R : ndarray of shape (mc_grad.ngorb+mc_grad.nci)
            Contains Hessian-vector product
    '''

    mc = mc_grad.base
    if mo is None: mo = mc.mo_coeff
    if ci is None: ci = mc.ci
    if eris is None: eris = mc.ao2mo (mo)
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nroots, nocc = mc_grad.nroots, ncore + ncas
    nmo = mo.shape[1]

    # CI vector shift
    L = np.zeros ((nroots, nroots), dtype=Lis.dtype)
    L[np.tril_indices (nroots, k=-1)] = Lis[:]
    L -= L.T
    ci_arr = np.asarray (ci)
    Lci = np.tensordot (L, ci_arr, axes=1)

    # Density matrices
    tril_idx = np.tril_indices (nroots)
    diag_idx = np.arange (nroots)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    tdm1 = np.stack (mc.fcisolver.states_trans_rdm12 (ci_arr[tril_idx[0]],
        ci_arr[tril_idx[1]], ncas, nelecas)[0], axis=0)
    dm1 = tdm1[diag_idx,:,:]
    edm1 = np.stack (mc.fcisolver.states_trans_rdm12 (Lci, ci, ncas,
        nelecas)[0], axis=0)
    edm1 += edm1.transpose (0,2,1)

    # Potentials
    aapa = np.zeros ([ncas,ncas,nmo,ncas], dtype=dm1.dtype)
    for i in range (ncas):
        j = i + ncore
        aapa[i,:,:,:] = eris.papa[j][:,:,:]
    vj = np.tensordot (dm1, aapa, axes=2)
    evj = np.tensordot (edm1, aapa, axes=2)

    # Orbital degree of freedom
    Rorb = np.zeros ((nmo,nmo), dtype=vj[0].dtype)
    Rorb[:,ncore:nocc] = sum ([np.dot (v, ed) + np.dot (ev, d)
        for v, d, ev, ed in zip (vj, dm1, evj, edm1)])
    Rorb -= Rorb.T

    # CI degree of freedom
    w = coulomb_tensor (mc, mo_coeff=mo, ci=ci, h2eff=aapa[:,:,ncore:nocc,:])
    const_IJ = -4*np.einsum ('jiik,ik->ij', w, L)
    const_IJ -= 2*np.einsum ('iijk,ik->ij', w, L)
    const_IJ += 2*np.einsum ('jkkk,ik->ij', w, L)
    Rci = np.tensordot (const_IJ, ci_arr, axes=1) # Delta_IJ |J> term

    def contract (v,c):
        return mc.fcisolver.contract_1e (v, c, ncas, nelecas)

    vj, evj = vj[:,ncore:nocc,:], evj[:,ncore:nocc,:]
    vci = np.stack ([contract (v,c) for v, c in zip (vj, ci)], axis=0)
    Rci -= 2 * np.tensordot (L, vci, axes=1) # -2 |zW_I> term
    for I in range (nroots):
        Rci[I] += 2 * contract (vj[I], Lci[I]) # 2 W^I_I |z_I> term
        Rci[I] += 2 * contract (evj[I], ci[I]) # 4 W^zI_I |I> z_IJ term
        # (*2 in def. of evj)

    return mc_grad.pack_uniq_var (2*Rorb, Rci)

# TODO: get rid?? Fix?? Unittest???
# BROKEN FOR CI AND IS; DO NOT USE
def diab_response_o0 (mc_grad, Lis, mo=None, ci=None, eris=None, **kwargs):
    '''Alternate implementation: monkeypatch everything but
    active-active Coulomb part of the Hamiltonian and call
    newton_casscf.gen_g_hop ()[2].
    '''

    mc = mc_grad.base
    if mo is None: mo = mc.mo_coeff
    if ci is None: ci = mc.ci
    if eris is None: eris = mc.ao2mo (mo)
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nroots, nocc, nmo = mc_grad.nroots, ncore + ncas, mo.shape[1]
    moH = mo.conj ().T

    # CI vector shift
    L = np.zeros ((nroots, nroots), dtype=Lis.dtype)
    L[np.tril_indices (nroots, k=-1)] = Lis[:]
    L -= L.T
    ci_arr = np.asarray (ci)
    Lci = list (np.tensordot (L, ci_arr, axes=1))
    x = mc_grad.pack_uniq_var (np.zeros ((nmo,nmo)), Lci)

    # Fake Hamiltonian!
    h1e_mo = moH @ mc.get_hcore () @ mo
    feris = mc.ao2mo (mo)
    for i in range (nmo):
        feris.papa[i][:,:,:] = 0.0
        feris.ppaa[i][:ncore,:,:] = 0.0
        feris.ppaa[i][nocc:,:,:] = 0.0
    feris.vhf_c[:,:] = -h1e_mo.copy ()
    from pyscf.mcscf.newton_casscf import _pack_ci_get_H as getH
    from pyscf.mcscf import addons
    def _pack_ci_get_H (mc1, mo1, ci1):
        ci1, _, _Hdiag, linkstrl, linkstr, _pack_ci, _unpack_ci = getH (mc1,
            mo1, ci1)
        dm1 = mc.fcisolver.states_make_rdm1 (ci1, ncas, nelecas)
        _state_arg = addons.StateAverageMixFCISolver_state_args
        def _Hci_kernel (s, op1, op2, ci1, ci2, ne, my_L):
            h1ci2, h1ci1, h2ci1, c1c2 = [], [], [], []
            for o1, o2, c2, c1 in zip (op1, op2, ci2, ci1):
                h1ci2.append (s.contract_1e (o1, c2, ncas, ne))
                h2ci1.append (s.contract_1e (o2, c1, ncas, ne))
                h1ci1.append (s.contract_1e (o1, c1, ncas, ne))
                c1c2.append (c1.dot (c2))
            if np.all (np.asarray (c1c2) < 0.5): # chain rule
                h1ci2, h2ci1 = np.asarray (h1ci2), np.asarray (h2ci1)
                h1ci1 = np.tensordot (my_L, np.asarray (h1ci1), axes=1)
                return list (h1ci2 + h2ci1 - h1ci1)
            else:
                return h1ci2

        if isinstance (mc.fcisolver, addons.StateAverageMixFCISolver):
            full_idx = np.arange (nroots)
            def _Hci (h1, h2, ci2):
                hci = []
                tm1 = mc.fcisolver.states_trans_rdm12 (ci2, ci1, ncas,
                    nelecas)[9]
                for s, args, kwargs in enumerate (mc.fcisolver._loop_solver (
                        _state_arg (ci2), _state_arg (ci1), _state_arg (dm1),
                        _state_arg (tm1), _state_arg (full_idx))):
                    ci2i, ci1i, dm1i, tm1i, idx = args[0:5]
                    Lsec = L[np.ix_(idx,idx)]
                    nelec = mc.fcisolver._get_nelec (s, nelecas)
                    op1 = h1[None,:,:] + np.tensordot (dm1i, h2, axes=2)
                    op2 = np.tensordot (tm1i + tm1i.transpose (0,2,1), h2,
                        axes=2)
                    hci.extend (_Hci_kernel (s, op1, op2, ci1i, ci2i, nelec,
                        Lsec))
                return hci
        else:
            def _Hci (h1, h2, ci2):
                tm1 = np.asarray (mc.fcisolver.states_trans_rdm12 (ci2, ci1,
                    ncas, nelecas)[0])
                op1 = h1[None,:,:] + np.tensordot (dm1, h2, axes=2)
                op2 = np.tensordot (tm1 + tm1.transpose (0,2,1), h2, axes=2)
                return _Hci_kernel (mc.fcisolver, op1, op2, ci1, ci2, nelecas,
                    L)
        return ci1, _Hci, _Hdiag, linkstrl, linkstr, _pack_ci, _unpack_ci

    # Fake 2TDM!
    dm1 = mc.fcisolver.states_make_rdm1 (ci, ncas, nelecas)
    def trans_rdm12 (ci1, ci0, *args, **kwargs):
        tm1, tm2 = mc.fcisolver.states_trans_rdm12 (ci1, ci0, *args, **kwargs)
        for t1, t2, d1, w in zip (tm1, tm2, dm1, mc.weights):
            t2[:,:,:,:] = w * (np.multiply.outer (t1, d1)
                             + np.multiply.outer (d1, t1))
            t1[:,:] *= w
        return sum (tm1), sum (tm2)

    # Fake Newton CASSCF!
    with lib.temporary_env (newton_casscf, _pack_ci_get_H=_pack_ci_get_H):
        with lib.temporary_env (mc.fcisolver, trans_rdm12=trans_rdm12):
            hx = newton_casscf.gen_g_hop (mc, mo, ci, feris, verbose=0)[2](x)
    hx *= nroots
    hx_orb, hx_ci = mc_grad.unpack_uniq_var (hx)
    hx_ci = np.asarray (hx_ci)
    hx_is = lib.einsum ('pab,qab->pq', hx_ci, ci_arr.conj ())
    hx_ci -= np.tensordot(hx_is, ci_arr, axes=1)

    return mc_grad.pack_uniq_var (hx_orb, hx_ci)

def diab_grad (mc_grad, Lis, atmlst=None, mo=None, ci=None, eris=None,
        mf_grad=None, **kwargs):
    '''Computes the partial first derivatives of

    Q_a-a = 1/2 sum_I g_pqrs <I|p'q|I> <I|r's|I>

    with respect to geometry perturbation.

    Args:
        mc_grad : object of class Gradients (CASSCF or CASCI)
        Lis : ndarray of shape (nroots*(nroots-1)/2,)
            Contains step vector for intermediate state rotations

    Kwargs:
        atmlst : list
            List of atoms whose geometries are perturbed. Defaults
            to all atoms in mc_grad.mol.
        mo : ndarray of shape (nao,nmo)
            Contains MO coefficients
        ci : ndarray or list of length (nroots)
            Contains intermediate-state CI vectors
        eris : object of class ERIS (CASSCF or CASCI)
            Contains (true) ERIs in the MO basis
        mf_grad: object of class Gradients (RHF)
            Defaults to mc_grad.base.get_rhf_base ().nuc_grad_method ()

    Returns:
        de : ndarray of shape (len (atmlst), 3)
            Contains gradient vector
    '''

    mc = mc_grad.base
    mol = mc_grad.mol
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nroots, nocc, nmo = mc_grad.nroots, ncore + ncas, mo.shape[1]
    moH = mo.conj ().T
    mo_cas = mo[:,ncore:nocc]
    moH_cas = moH[ncore:nocc,:]
    if mf_grad is None: mf_grad = mc.get_rhf_base ().nuc_grad_method()
    if atmlst is None: atmlst = list (range(mol.natm))

    # CI vector shift
    L = np.zeros ((nroots, nroots), dtype=Lis.dtype)
    L[np.tril_indices (nroots, k=-1)] = Lis[:]
    L -= L.T
    ci_arr = np.asarray (ci)
    Lci = np.tensordot (L, ci_arr, axes=1)

    # Density matrices
    dm1 = np.stack (mc.fcisolver.states_make_rdm1 (ci, ncas, nelecas), axis=0)
    edm1 = np.stack (mc.fcisolver.states_trans_rdm12 (Lci, ci, ncas,
        nelecas)[0], axis=0)
    edm1 += edm1.transpose (0,2,1)
    dm1_ao = reduce (np.dot, (mo_cas, dm1, moH_cas)).transpose (1,0,2)
    edm1_ao = reduce (np.dot, (mo_cas, edm1, moH_cas)).transpose (1,0,2)

    # Potentials and operators
    aapa = np.zeros ([nmo,]+[ncas,]*3, dtype=dm1.dtype)
    for i in range (nmo): aapa[i] = eris.ppaa[i][ncore:nocc,:,:]
    aapa = aapa.transpose (2,3,0,1)
    vj = np.tensordot (dm1, aapa, axes=2)
    evj = np.tensordot (edm1, aapa, axes=2)
    dvj_all = mf_grad.get_j (mc.mol, list(dm1_ao) + list(edm1_ao))
    dvj_aux = getattr (dvj_all, 'aux', np.zeros ((nroots, nroots, mol.natm,
        3)))
    dvj = np.stack (dvj_all[:nroots], axis=1)
    devj = np.stack (dvj_all[nroots:], axis=1)

    # Generalized Fock and overlap operator
    gfock = np.zeros ([nmo,nmo], dtype=vj.dtype)
    gfock[:,ncore:nocc] = sum ([np.dot (v, ed) + np.dot (ev, d)
        for v, d, ev, ed in zip (vj, dm1, evj, edm1)])
    dme0 = reduce (np.dot, (mo, (gfock+gfock.T)*.5, moH))
    s1 = mf_grad.get_ovlp (mc.mol)

    # Crunch
    de_direct = np.zeros ((len (atmlst), 3))
    de_renorm = np.zeros ((len (atmlst), 3))
    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        de_renorm[k] -= lib.einsum('xpq,pq->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        de_direct[k] += lib.einsum('xipq,ipq->x', dvj[:,:,p0:p1],
            edm1_ao[:,p0:p1]) * 2
        de_direct[k] += lib.einsum('xipq,ipq->x', devj[:,:,p0:p1],
            dm1_ao[:,p0:p1]) * 2
    dvj_aux = dvj_aux[:,:,atmlst,:]
    de_aux = (np.trace (dvj_aux, offset=nroots, axis1=0, axis2=1)
            + np.trace (dvj_aux, offset=-nroots, axis1=0, axis2=1))

    logger.debug (mc, "CMS-PDFT Lis lagrange direct component:\n{}".format (
        de_direct))
    logger.debug (mc, "CMS-PDFT Lis lagrange renorm component:\n{}".format (
        de_renorm))
    logger.debug (mc, "CMS-PDFT Lis lagrange auxbasis component:\n{}".format (
        de_aux))
    de = de_direct + de_aux + de_renorm
    return de

# TODO: get rid? Unittest?
def diab_grad_o0 (mc_grad, Lis, atmlst=None, mo=None, ci=None, eris=None,
        mf_grad=None, **kwargs):
    ''' Monkeypatch version of diab_grad '''
    mc = mc_grad.base
    ncas, nelecas, nroots = mc.ncas, mc.nelecas, mc_grad.nroots
    if mf_grad is None: mf_grad = mc.get_rhf_base ().nuc_grad_method()

    # CI vector shift
    L = np.zeros ((nroots, nroots), dtype=Lis.dtype)
    L[np.tril_indices (nroots, k=-1)] = Lis[:]
    L -= L.T
    ci_arr = np.asarray (ci)
    Lci = list (np.tensordot (L, ci_arr, axes=1))

    # Fake dms!
    dm1 = mc.fcisolver.states_make_rdm1 (ci, ncas, nelecas)
    def trans_rdm12 (ci1, ci0, *args, **kwargs):
        tm1, tm2 = mc.fcisolver.states_trans_rdm12 (ci1, ci0, *args, **kwargs)
        for t1, t2, d1, w in zip (tm1, tm2, dm1, mc.weights):
            t2[:,:,:,:] = (np.multiply.outer (t1, d1)
                         + np.multiply.outer (d1, t1))
            t1[:,:] = 0.0
        return sum (tm1), sum (tm2)

    from pyscf.grad.sacasscf import Lci_dot_dgci_dx
    with lib.temporary_env (mc.fcisolver, trans_rdm12=trans_rdm12):
        de = Lci_dot_dgci_dx (Lci, mc.weights, mc, mo_coeff=mo, ci=ci,
            atmlst=atmlst, eris=eris, mf_grad=mf_grad)
    return de

if __name__ == '__main__':
    import math
    from pyscf import scf, gto, mcscf
    from pyscf.fci import csf_solver
    from pyscf.mcpdft.pdft_feff import vector_error
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=False, output='cmspdft.log',
        verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcscf.CASSCF (mf, 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.state_average ([1.0/3,]*3).run ()
    ci_arr = np.asarray (mc.ci)

    mc_grad = mc.nuc_grad_method ()
    Lis = math.pi * (np.random.rand (3) - 0.5)
    eris = mc.ao2mo (mc.mo_coeff)

    dw_test = diab_response (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci,
        eris=eris)
    dworb_test, dwci_test = mc_grad.unpack_uniq_var (dw_test)
    dwci_test = np.asarray (dwci_test)
    dwis_test = lib.einsum ('pab,qab->pq', dwci_test, ci_arr.conj ())
    dwci_test -= lib.einsum ('pq,qab->pab', dwis_test, ci_arr)
    dw_ref = diab_response_o0 (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci,
        eris=eris)
    dworb_ref, dwci_ref = mc_grad.unpack_uniq_var (dw_ref)
    dwci_ref = np.asarray (dwci_ref)
    dwis_ref = lib.einsum ('pab,qab->pq', dwci_ref, ci_arr.conj ())
    dwci_ref -= lib.einsum ('pq,qab->pab', dwis_ref, ci_arr)
    dh_test = diab_grad (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)
    dh_ref = diab_grad_o0 (mc_grad, Lis, mo=mc.mo_coeff, ci=mc.ci, eris=eris)

    print ("dworb:", vector_error (dworb_test, dworb_ref), linalg.norm (
        dworb_ref))
    print ("dwci:", vector_error (dwci_test, dwci_ref), linalg.norm (dwci_ref))
    print ("dwis:", vector_error (dwis_test, dwis_ref), linalg.norm (dwis_ref))
    print ("dh:", vector_error (dh_test, dh_ref), linalg.norm (dh_ref))

    print (dh_test, '\n', dh_ref)
