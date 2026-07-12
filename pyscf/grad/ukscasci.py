#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
UCASCI analytical nuclear gradients with UKS orbitals

The orbital response is the CPKS response of the underlying UKS reference, with
the associated spin-resolved XC first-derivative terms in the KS orbital-source
constraints.

Supported functionals include:
  LDA: lda, lda,vwn, slater,vwn, svwn
  GGA: pbe, pbe,pbe, b88,p86, bp86, b88,lyp, blyp, pw91,pw91, b97-d
  global hybrid GGA: b3lyp, b3lyp5, b3p86, b3pw91, pbe0, pbe1pbe, o3lyp, x3lyp

Unsupported functionals include:
  meta-GGA: tpss, scan, m06-l, m06, m06-2x, mn15
  range-separated hybrids: cam-b3lyp, wb97x
  NLC: vv10
'''

from functools import reduce

from pyscf.lib import logger
import numpy
from pyscf.scf import hf
from pyscf.scf import ucphf
from pyscf import scf
from pyscf import mcscf
from pyscf import lib
from pyscf.mcscf import umc1step
from pyscf.grad import ucasci as ucasci_grad
from pyscf.grad import kscasci as kscasci_grad
from pyscf.grad.mp2 import _shell_prange


def _check_supported_functional(mc):
    ni = mc._scf._numint
    xctype = ni._xc_type(mc._scf.xc)
    if xctype == 'MGGA':
        raise NotImplementedError('UKS-CASCI gradients for meta-GGA '
                                  'functionals')
    if xctype not in ('LDA', 'GGA'):
        raise NotImplementedError('UKS-CASCI gradients for %s functionals' %
                                  xctype)
    if mc._scf.do_nlc():
        raise NotImplementedError('UKS-CASCI gradients for NLC functionals')
    omega, _, _ = ni.rsh_and_hybrid_coeff(mc._scf.xc, mc.mol.spin)
    if omega != 0:
        raise NotImplementedError('UKS-CASCI gradients for range-separated '
                                  'hybrid functionals')


def _check_supported_orbital_source(mc):
    if not isinstance(mc._scf, hf.KohnShamDFT):
        raise RuntimeError('UKS-CASCI gradients require a KohnShamDFT '
                           'reference')
    if getattr(mc, '_scf_df_source', False):
        raise NotImplementedError(
            'UKS-CASCI gradients with density-fitted KS orbitals require '
            'DF-UKS orbital-response terms')
    if getattr(mc._scf, 'with_df', None):
        raise NotImplementedError(
            'UKS-CASCI gradients with density-fitted KS orbitals require '
            'DF-UKS orbital-response terms')
    _check_supported_functional(mc)


def _is_restricted_collapse(mc, mo_coeff):
    if mc.mol.spin != 0:
        return False
    ncorea, ncoreb = mc.ncore
    neleca, nelecb = mc.nelecas
    return (ncorea == ncoreb and neleca == nelecb and
            numpy.allclose(mo_coeff[0], mo_coeff[1], atol=1e-8, rtol=1e-8))


def _restricted_collapse_grad(mc_grad, mo_coeff, ci, atmlst, verbose):
    mc = mc_grad.base
    mf = scf.addons.convert_to_rhf(mc._scf)
    mf.mo_coeff = mo_coeff[0]
    mf.mo_occ = mc._scf.mo_occ[0] * 2
    mf.mo_energy = mc._scf.mo_energy[0]
    mf.converged = mc._scf.converged
    rmc = mcscf.CASCI(mf, mc.ncas, sum(mc.nelecas), ncore=mc.ncore[0])
    rmc.mo_coeff = mo_coeff[0]
    rmc.ci = ci
    rmc.e_tot = mc.e_tot
    rmc.e_cas = mc.e_cas
    rmc.converged = getattr(mc, 'converged', True)
    rmc.fcisolver = mc.fcisolver
    with lib.temporary_env(mc_grad, base=rmc):
        return kscasci_grad.grad_elec(mc_grad, mo_coeff[0], ci, atmlst,
                                      verbose)


def _ao_density_components(casdm1s, mo_coeff, ncore, ncas):
    mo_core = (mo_coeff[0][:,:ncore[0]], mo_coeff[1][:,:ncore[1]])
    mo_act = (mo_coeff[0][:,ncore[0]:ncore[0]+ncas],
              mo_coeff[1][:,ncore[1]:ncore[1]+ncas])
    dm_core = [numpy.dot(mo_core[s], mo_core[s].T) for s in range(2)]
    dm_act = [reduce(numpy.dot, (mo_act[s], casdm1s[s], mo_act[s].T))
              for s in range(2)]
    return mo_act, dm_core, dm_act


def _same_spin_dm2ao_slice(dm_core, dm_act, mo_act, casdm2,
                           p0, p1, q0, q1):
    # Same-spin CASCI 2-RDM = core determinant + core-active terms + active
    # CAS 2-RDM.  Only the AO slice needed for the current derivative-ERI block
    # is transformed.
    dm2 = lib.einsum('pq,rs->pqrs', dm_core[p0:p1,q0:q1], dm_core)
    dm2 -= lib.einsum('ps,qr->pqrs', dm_core[p0:p1], dm_core[q0:q1])
    dm2 += lib.einsum('pq,rs->pqrs', dm_core[p0:p1,q0:q1], dm_act)
    dm2 += lib.einsum('pq,rs->pqrs', dm_act[p0:p1,q0:q1], dm_core)
    dm2 -= lib.einsum('ps,qr->pqrs', dm_core[p0:p1], dm_act[q0:q1])
    dm2 -= lib.einsum('ps,qr->pqrs', dm_act[p0:p1], dm_core[q0:q1])
    dm2 += lib.einsum('pi,qj,rk,sl,ijkl->pqrs',
                      mo_act[p0:p1], mo_act[q0:q1], mo_act, mo_act,
                      casdm2)
    return dm2


def _mixed_spin_dm2ao_slice(dm_core0, dm_act0, mo_act0,
                            dm_core1, dm_act1, mo_act1, casdm2,
                            p0, p1, q0, q1):
    # Mixed-spin 2-RDM has no exchange terms.  The first spin occupies the
    # derivative index pair (p,q); the second spin occupies (r,s).
    dm2 = lib.einsum('pq,rs->pqrs', dm_core0[p0:p1,q0:q1],
                     dm_core1 + dm_act1)
    dm2 += lib.einsum('pq,rs->pqrs', dm_act0[p0:p1,q0:q1], dm_core1)
    dm2 += lib.einsum('pi,qj,rk,sl,ijkl->pqrs',
                      mo_act0[p0:p1], mo_act0[q0:q1], mo_act1, mo_act1,
                      casdm2)
    return dm2


def _casci_orbital_intermediates(mc, mo_coeff, ci, casdm1s, casdm2s):
    casscf = mcscf.UCASSCF(mc._scf, mc.ncas, mc.nelecas, ncore=mc.ncore)
    casscf.mo_coeff = mo_coeff
    casscf.ci = ci
    casscf.fcisolver = mc.fcisolver
    eris = casscf.ao2mo(mo_coeff)

    # UCASSCF uses the antisymmetric orbital-gradient vector as the orbital
    # optimization RHS.  The same object is Delta X in the Sherrill's notes
    # for CI-gradient orbital-response term
    # https://vergil.chemistry.gatech.edu/static/content/cigrad.pdf
    g_orb = umc1step.gen_g_hop(casscf, mo_coeff, None,
                               casdm1s, casdm2s, eris)[0]
    dx = casscf.unpack_uniq_var(g_orb)

    # The overlap/Lagrangian term needs the unsymmetrized generalized Fock X,
    # not the antisymmetric orbital-gradient matrix returned by gen_g_hop.
    ncas = mc.ncas
    ncore = mc.ncore
    nocc = (ncore[0] + ncas, ncore[1] + ncas)
    nmo = mo_coeff[0].shape[1]
    dm1 = numpy.zeros((2,nmo,nmo))
    dm1[0,numpy.arange(ncore[0]),numpy.arange(ncore[0])] = 1
    dm1[1,numpy.arange(ncore[1]),numpy.arange(ncore[1])] = 1
    dm1[0,ncore[0]:nocc[0],ncore[0]:nocc[0]] = casdm1s[0]
    dm1[1,ncore[1]:nocc[1],ncore[1]:nocc[1]] = casdm1s[1]

    vhf_c = eris.vhf_c
    vhf_ca = (
        vhf_c[0]
        + numpy.einsum('uvpq,uv->pq', eris.aapp, casdm1s[0])
        - numpy.einsum('upqv,uv->pq', eris.appa, casdm1s[0])
        + numpy.einsum('uvpq,uv->pq', eris.AApp, casdm1s[1]),
        vhf_c[1]
        + numpy.einsum('uvpq,uv->pq', eris.aaPP, casdm1s[0])
        + numpy.einsum('uvpq,uv->pq', eris.AAPP, casdm1s[1])
        - numpy.einsum('upqv,uv->pq', eris.APPA, casdm1s[1]))
    hdm2 = [
        numpy.einsum('tuvw,vwpq->tupq', casdm2s[0], eris.aapp)
        + numpy.einsum('tuvw,vwpq->tupq', casdm2s[1], eris.AApp),
        numpy.einsum('vwtu,vwpq->tupq', casdm2s[1], eris.aaPP)
        + numpy.einsum('tuvw,vwpq->tupq', casdm2s[2], eris.AAPP)]
    hcore = casscf.get_hcore()
    h1e_mo = (reduce(numpy.dot, (mo_coeff[0].T, hcore[0], mo_coeff[0])),
              reduce(numpy.dot, (mo_coeff[1].T, hcore[1], mo_coeff[1])))
    xmat = [numpy.dot(h1e_mo[0], dm1[0]),
            numpy.dot(h1e_mo[1], dm1[1])]
    for s in range(2):
        xmat[s][:,:ncore[s]] += vhf_ca[s][:,:ncore[s]]
        xmat[s][:,ncore[s]:nocc[s]] += (
            numpy.einsum('vuuq->qv', hdm2[s][:,:,ncore[s]:nocc[s]])
            + numpy.dot(vhf_c[s][:,ncore[s]:nocc[s]], casdm1s[s]))
    return dx, tuple(xmat)


def _reference_occ(mc, mo_coeff):
    ncorea, ncoreb = mc.ncore
    neleca, nelecb = mc.nelecas
    nmoa = mo_coeff[0].shape[1]
    nmob = mo_coeff[1].shape[1]
    mo_occ = (numpy.zeros(nmoa), numpy.zeros(nmob))
    mo_occ[0][:ncorea+neleca] = 1
    mo_occ[1][:ncoreb+nelecb] = 1
    return mo_occ


def _orbital_response_grad(mc_grad, mo_coeff, dx, atmlst):
    mc = mc_grad.base
    mol = mc_grad.mol
    atmlst = list(atmlst)
    mo_occ = _reference_occ(mc, mo_coeff)
    nocc = (numpy.count_nonzero(mo_occ[0] > 0),
            numpy.count_nonzero(mo_occ[1] > 0))
    nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
    nao = mo_coeff[0].shape[0]
    h1ao = mc._scf.Hessian().make_h1(mo_coeff, mo_occ)
    s1a = mc_grad.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    vresp = mc._scf.gen_response(mo_coeff, mo_occ, hermi=1)
    viridx = (mo_occ[0] == 0, mo_occ[1] == 0)

    def occ_response_dm(mo1):
        nset = mo1[0].shape[0]
        dm = numpy.empty((2,nset,nao,nao))
        for s in range(2):
            dm[s] = numpy.einsum('pi,xij,qj->xpq',
                                 mo_coeff[s], mo1[s],
                                 mo_coeff[s][:,:nocc[s]])
        return dm + dm.transpose(0,1,3,2)

    def full_response_dm(u):
        dm = numpy.empty((2,nao,nao))
        for s in range(2):
            dm[s] = reduce(numpy.dot, (mo_coeff[s], u[s], mo_coeff[s].T))
        return dm + dm.transpose(0,2,1)

    # Put CASCI-independent rotations that are redundant for the UKS
    # determinant directly into zvec with canonical-orbital denominators.  Their
    # induced KS response potential contributes to the occupied-virtual adjoint
    # RHS below, matching the restricted CASCI Z-vector strategy.
    zvec = [numpy.zeros((nmo[s], nmo[s])) for s in range(2)]
    for s in range(2):
        eps = mc._scf.mo_energy[s]
        ncore = mc.ncore[s]
        nref = ncore + mc.nelecas[s]
        ncasocc = ncore + mc.ncas
        for p in range(ncore, nref):
            for q in range(ncore):
                zvec[s][p,q] = dx[s][p,q] / (eps[q] - eps[p])
        for p in range(ncasocc, nmo[s]):
            for q in range(nref, ncasocc):
                zvec[s][p,q] = dx[s][p,q] / (eps[q] - eps[p])

    vden = vresp(full_response_dm(zvec))
    wvo = []
    for s in range(2):
        w = dx[s] + reduce(numpy.dot, (mo_coeff[s].T, vden[s],
                                       mo_coeff[s]))
        wvo.append(w[viridx[s]][:,:nocc[s]])

    def fvind(x):
        # UCPHF may ask for several right-hand sides at once.  Keep the leading
        # dimension through the AO density build so the UKS response kernel can
        # contract all corresponding XC potentials in one call.
        x = x.reshape(-1, wvo[0].size + wvo[1].size)
        nset = x.shape[0]
        mo1 = []
        p0 = 0
        for s in range(2):
            p1 = p0 + wvo[s].size
            z = numpy.zeros((nset, nmo[s], nocc[s]))
            z[:,viridx[s]] = x[:,p0:p1].reshape((nset,) + wvo[s].shape)
            mo1.append(z)
            p0 = p1
        dm = occ_response_dm(mo1)
        v1 = vresp(dm)
        vout = []
        for s in range(2):
            v = numpy.einsum('pi,xpq,qj->xij', mo_coeff[s], v1[s],
                             mo_coeff[s][:,:nocc[s]])
            vout.append(v[:,viridx[s]].reshape(nset,-1))
        return numpy.hstack(vout)

    # This is the single perturbation-independent adjoint CPKS solve.  The
    # nuclear derivative enters only later through B0-like contractions.
    zvo = ucphf.solve(fvind, mc._scf.mo_energy, mo_occ, tuple(wvo),
                      max_cycle=50)[0]
    for s in range(2):
        zvec[s][viridx[s],:nocc[s]] = zvo[s]

    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        s1 = numpy.zeros_like(s1a)
        s1[:,p0:p1] += s1a[:,p0:p1]
        s1[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        # ucphf.solve(with s1) fixes the occupied-occupied response gauge to
        # -S1/2.  In the adjoint form this gauge is a direct source term and
        # its induced UKS response potential contributes to the B0 contraction.
        smo = []
        ug = []
        for s in range(2):
            sm = numpy.einsum('xij,ip,jq->xpq', s1, mo_coeff[s],
                              mo_coeff[s])
            u = numpy.zeros((3,nmo[s],nmo[s]))
            u[:,:nocc[s],:nocc[s]] = -.5 * sm[:,:nocc[s],:nocc[s]]
            smo.append(sm)
            ug.append(u)

        dm_g = numpy.empty((2,3,nao,nao))
        for x in range(3):
            dm_g[:,x] = full_response_dm((ug[0][x], ug[1][x]))
        vg = vresp(dm_g)
        for s in range(2):
            b0 = numpy.einsum('pi,xpq,qj->xij', mo_coeff[s],
                              h1ao[s][ia] + vg[s], mo_coeff[s])
            b0 -= smo[s] * mc._scf.mo_energy[s][None,None,:]
            de[k] += 2 * numpy.einsum('pq,xpq->x', zvec[s], b0)
            de[k] += 2 * numpy.einsum('xpq,pq->x',
                                      ug[s][:,:nocc[s],:nocc[s]],
                                      dx[s][:nocc[s],:nocc[s]])
    return de


def _skeleton_grad(mc_grad, mo_coeff, xmo, casdm1s, casdm2s, atmlst):
    mc = mc_grad.base
    mol = mc_grad.mol
    nao = mol.nao
    mo_act, dm_core, dm_act = _ao_density_components(
        casdm1s, mo_coeff, mc.ncore, mc.ncas)
    dm1ao = (dm_core[0] + dm_act[0], dm_core[1] + dm_act[1])

    hcore_deriv = mc_grad.hcore_generator(mol)
    s1a = mc_grad.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    xtilde = []
    for s in range(2):
        x = xmo[s].copy()
        idx = numpy.tril_indices(x.shape[0], -1)
        x[idx] = xmo[s].T[idx]
        xtilde.append(x)

    de = numpy.zeros((len(atmlst),3))
    max_memory = max(2000, mc_grad.max_memory*.9 - lib.current_memory()[0])
    max_atom_nao = max(aoslices[:,3] - aoslices[:,2])
    blksize = int(max_memory*1e6/8 / (max_atom_nao * nao * nao * 6))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        de[k] += numpy.einsum('xij,ij->x', hcore_deriv(ia),
                              dm1ao[0] + dm1ao[1])

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s1',
                             shls_slice=shls_slice)
            eri1 = eri1.reshape(3, p1-p0, nf, nao, nao)

            # Same-spin FCI RDMs already carry the convention needed by
            # PySCF's one-index derivative integral contraction.  Do not halve
            # them here as in the spin-free energy expression.
            dm2a = _same_spin_dm2ao_slice(
                dm_core[0], dm_act[0], mo_act[0], casdm2s[0],
                p0, p1, q0, q1)
            dm2a += _mixed_spin_dm2ao_slice(
                dm_core[0], dm_act[0], mo_act[0],
                dm_core[1], dm_act[1], mo_act[1], casdm2s[1],
                p0, p1, q0, q1)
            dm2b = _same_spin_dm2ao_slice(
                dm_core[1], dm_act[1], mo_act[1], casdm2s[2],
                p0, p1, q0, q1)
            dm2b += _mixed_spin_dm2ao_slice(
                dm_core[1], dm_act[1], mo_act[1],
                dm_core[0], dm_act[0], mo_act[0],
                casdm2s[1].transpose(2,3,0,1), p0, p1, q0, q1)

            de[k] -= numpy.einsum('xpqrs,pqrs->x', eri1, dm2a) * 2
            de[k] -= numpy.einsum('xpqrs,pqrs->x', eri1, dm2b) * 2

        s1 = numpy.zeros_like(s1a)
        s1[:,p0:p1] += s1a[:,p0:p1]
        s1[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        for s in range(2):
            s1mo = numpy.einsum('xij,ip,jq->xpq',
                                s1, mo_coeff[s], mo_coeff[s])
            de[k] -= numpy.einsum('xij,ij->x', s1mo, xtilde[s])
    return de


def _direct_grad_elec(mc_grad, mo_coeff, ci, atmlst):
    mc = mc_grad.base
    casdm1s, casdm2s = mc.fcisolver.make_rdm12s(ci, mc.ncas, mc.nelecas)
    dx, xmo = _casci_orbital_intermediates(
        mc, mo_coeff, ci, casdm1s, casdm2s)
    de = _skeleton_grad(mc_grad, mo_coeff, xmo, casdm1s, casdm2s, atmlst)
    de += _orbital_response_grad(mc_grad, mo_coeff, dx, atmlst)
    return de


def grad_elec(mc_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    mc = mc_grad.base
    _check_supported_orbital_source(mc)
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if _is_restricted_collapse(mc, mo_coeff):
        return _restricted_collapse_grad(mc_grad, mo_coeff, ci, atmlst,
                                         verbose)
    if atmlst is None:
        atmlst = range(mc_grad.mol.natm)
    return _direct_grad_elec(mc_grad, mo_coeff, ci, list(atmlst))


class Gradients(ucasci_grad.Gradients):
    '''Non-relativistic UKS-CASCI gradients'''

    grad_elec = grad_elec

    def dump_flags(self, verbose=None):
        ucasci_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'KS orbital source = %s', self.base._scf.xc)
        return self


Grad = Gradients
