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
UCASCI analytical nuclear gradients with unrestricted KS orbitals

This module implements the UKS-orbital UCASCI gradient formulation.  The
orbital response is the CPKS response of the underlying UKS reference, with
the associated spin-resolved XC first-derivative terms in the KS orbital-source
constraints.
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


def _full_rdm12s(casdm1s, casdm2s, nmo, ncore, ncas):
    ncorea, ncoreb = ncore
    nacca = ncorea + ncas
    naccb = ncoreb + ncas
    casdm1a, casdm1b = casdm1s
    casdm2aa, casdm2ab, casdm2bb = casdm2s

    dm1a = numpy.zeros((nmo,nmo))
    dm1b = numpy.zeros((nmo,nmo))
    dm1a[numpy.arange(ncorea),numpy.arange(ncorea)] = 1
    dm1b[numpy.arange(ncoreb),numpy.arange(ncoreb)] = 1
    dm1a[ncorea:nacca,ncorea:nacca] = casdm1a
    dm1b[ncoreb:naccb,ncoreb:naccb] = casdm1b

    dm2aa = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2ab = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2bb = numpy.zeros((nmo,nmo,nmo,nmo))
    for i in range(ncorea):
        for j in range(ncorea):
            dm2aa[i,i,j,j] += 1
            dm2aa[i,j,j,i] -= 1
        dm2aa[i,i,ncorea:nacca,ncorea:nacca] += casdm1a
        dm2aa[ncorea:nacca,ncorea:nacca,i,i] += casdm1a
        dm2aa[i,ncorea:nacca,ncorea:nacca,i] -= casdm1a
        dm2aa[ncorea:nacca,i,i,ncorea:nacca] -= casdm1a
    for i in range(ncoreb):
        for j in range(ncoreb):
            dm2bb[i,i,j,j] += 1
            dm2bb[i,j,j,i] -= 1
        dm2bb[i,i,ncoreb:naccb,ncoreb:naccb] += casdm1b
        dm2bb[ncoreb:naccb,ncoreb:naccb,i,i] += casdm1b
        dm2bb[i,ncoreb:naccb,ncoreb:naccb,i] -= casdm1b
        dm2bb[ncoreb:naccb,i,i,ncoreb:naccb] -= casdm1b
    for i in range(ncorea):
        for j in range(ncoreb):
            dm2ab[i,i,j,j] += 1
        dm2ab[i,i,ncoreb:naccb,ncoreb:naccb] += casdm1b
    for j in range(ncoreb):
        dm2ab[ncorea:nacca,ncorea:nacca,j,j] += casdm1a

    dm2aa[ncorea:nacca,ncorea:nacca,ncorea:nacca,ncorea:nacca] += casdm2aa
    dm2ab[ncorea:nacca,ncorea:nacca,ncoreb:naccb,ncoreb:naccb] += casdm2ab
    dm2bb[ncoreb:naccb,ncoreb:naccb,ncoreb:naccb,ncoreb:naccb] += casdm2bb
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)


def _casci_orbital_gradient(mc, mo_coeff, ci, casdm1s, casdm2s):
    # UCASSCF uses the antisymmetric orbital-gradient vector as the orbital
    # optimization RHS.  The same object is Delta X in the Sherrill CI-gradient
    # orbital-response term.
    casscf = mcscf.UCASSCF(mc._scf, mc.ncas, mc.nelecas, ncore=mc.ncore)
    casscf.mo_coeff = mo_coeff
    casscf.ci = ci
    casscf.fcisolver = mc.fcisolver
    eris = casscf.ao2mo(mo_coeff)
    g_orb = umc1step.gen_g_hop(casscf, mo_coeff, None,
                               casdm1s, casdm2s, eris)[0]
    return casscf.unpack_uniq_var(g_orb)


def _casci_generalized_fock(mc, mo_coeff, ci, casdm1s, casdm2s):
    # The overlap/Lagrangian term needs the unsymmetrized generalized Fock X,
    # not the antisymmetric orbital-gradient matrix returned by gen_g_hop.
    casscf = mcscf.UCASSCF(mc._scf, mc.ncas, mc.nelecas, ncore=mc.ncore)
    casscf.mo_coeff = mo_coeff
    casscf.ci = ci
    casscf.fcisolver = mc.fcisolver
    eris = casscf.ao2mo(mo_coeff)
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
    return tuple(xmat)


def _reference_occ(mc, mo_coeff):
    ncorea, ncoreb = mc.ncore
    neleca, nelecb = mc.nelecas
    nmoa = mo_coeff[0].shape[1]
    nmob = mo_coeff[1].shape[1]
    mo_occ = (numpy.zeros(nmoa), numpy.zeros(nmob))
    mo_occ[0][:ncorea+neleca] = 1
    mo_occ[1][:ncoreb+nelecb] = 1
    return mo_occ


def _orbital_response_grad(mc_grad, mo_coeff, ci, casdm1s, casdm2s, atmlst):
    mc = mc_grad.base
    mol = mc_grad.mol
    mo_occ = _reference_occ(mc, mo_coeff)
    nocc = (numpy.count_nonzero(mo_occ[0] > 0),
            numpy.count_nonzero(mo_occ[1] > 0))
    nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
    nao = mo_coeff[0].shape[0]
    dx = _casci_orbital_gradient(mc, mo_coeff, ci, casdm1s, casdm2s)
    h1ao = mc._scf.Hessian().make_h1(mo_coeff, mo_occ)
    s1a = mc_grad.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    vresp = mc._scf.gen_response(mo_coeff, mo_occ, hermi=1)

    def fvind(x):
        x = x.reshape(-1, nmo[0]*nocc[0] + nmo[1]*nocc[1])
        vout = []
        for x1 in x:
            xa = x1[:nmo[0]*nocc[0]].reshape(nmo[0], nocc[0])
            xb = x1[nmo[0]*nocc[0]:].reshape(nmo[1], nocc[1])
            dm = numpy.empty((2,nao,nao))
            dm[0] = reduce(numpy.dot, (mo_coeff[0], xa,
                                       mo_coeff[0][:,:nocc[0]].T))
            dm[1] = reduce(numpy.dot, (mo_coeff[1], xb,
                                       mo_coeff[1][:,:nocc[1]].T))
            dm = dm + dm.transpose(0,2,1)
            v1 = vresp(dm)
            va = reduce(numpy.dot, (mo_coeff[0].T, v1[0],
                                    mo_coeff[0][:,:nocc[0]]))
            vb = reduce(numpy.dot, (mo_coeff[1].T, v1[1],
                                    mo_coeff[1][:,:nocc[1]]))
            vout.append(numpy.hstack((va.ravel(), vb.ravel())))
        return numpy.asarray(vout)

    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        s1 = numpy.zeros_like(s1a)
        s1[:,p0:p1] += s1a[:,p0:p1]
        s1[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        h1mo = []
        s1mo = []
        for s in range(2):
            h1mo.append(numpy.einsum('xij,ip,jq->xpq',
                                     h1ao[s][ia], mo_coeff[s],
                                     mo_coeff[s][:,:nocc[s]]))
            s1mo.append(numpy.einsum('xij,ip,jq->xpq',
                                     s1, mo_coeff[s],
                                     mo_coeff[s][:,:nocc[s]]))

        mo1 = ucphf.solve(fvind, mc._scf.mo_energy, mo_occ,
                          (h1mo[0], h1mo[1]), (s1mo[0], s1mo[1]),
                          max_cycle=50)[0]
        for x in range(3):
            dm1 = numpy.empty((2,nao,nao))
            for s in range(2):
                dm1[s] = reduce(numpy.dot, (mo_coeff[s], mo1[s][x],
                                            mo_coeff[s][:,:nocc[s]].T))
            dm1 = dm1 + dm1.transpose(0,2,1)
            v1 = vresp(dm1)

            for s in range(2):
                u = numpy.zeros_like(dx[s])
                u[:,:nocc[s]] = mo1[s][x]

                b0 = reduce(numpy.dot, (mo_coeff[s].T,
                                        h1ao[s][ia][x] + v1[s],
                                        mo_coeff[s]))
                smo = reduce(numpy.dot, (mo_coeff[s].T, s1[x], mo_coeff[s]))
                b0 -= smo * mc._scf.mo_energy[s][None,:]
                eps = mc._scf.mo_energy[s]
                ncore = mc.ncore[s]
                nref = ncore + mc.nelecas[s]
                ncasocc = ncore + mc.ncas

                # ucphf.solve(with s1) fixes the occupied-occupied response
                # gauge to -S1/2.  CASCI also needs the canonical
                # active-occupied/core rotation, so add the off-diagonal
                # first-order Fock correction in the UCASSCF unique-variable
                # orientation.
                for p in range(ncore, nref):
                    for q in range(ncore):
                        u[p,q] += b0[p,q] / (eps[q] - eps[p])
                for p in range(ncasocc, nmo[s]):
                    for q in range(nref, ncasocc):
                        u[p,q] = b0[p,q] / (eps[q] - eps[p])

                de[k,x] += 2 * numpy.einsum('pq,pq->', u, dx[s])
    return de


def _skeleton_grad(mc_grad, mo_coeff, ci, casdm1s, casdm2s, atmlst):
    mc = mc_grad.base
    mol = mc_grad.mol
    nmo = mo_coeff[0].shape[1]
    dm1mo, dm2mo = _full_rdm12s(casdm1s, casdm2s, nmo, mc.ncore, mc.ncas)
    dm1ao = (reduce(numpy.dot, (mo_coeff[0], dm1mo[0], mo_coeff[0].T)),
             reduce(numpy.dot, (mo_coeff[1], dm1mo[1], mo_coeff[1].T)))
    dm2aa = lib.einsum('pi,qj,rk,sl,ijkl->pqrs',
                       mo_coeff[0], mo_coeff[0], mo_coeff[0], mo_coeff[0],
                       dm2mo[0])
    dm2ab = lib.einsum('pi,qj,rk,sl,ijkl->pqrs',
                       mo_coeff[0], mo_coeff[0], mo_coeff[1], mo_coeff[1],
                       dm2mo[1])
    dm2bb = lib.einsum('pi,qj,rk,sl,ijkl->pqrs',
                       mo_coeff[1], mo_coeff[1], mo_coeff[1], mo_coeff[1],
                       dm2mo[2])
    # Same-spin FCI RDMs already carry the convention needed by PySCF's
    # one-index derivative integral contraction.  Do not halve them here as in
    # the spin-free energy expression.
    dm2a = dm2aa + dm2ab
    dm2b = dm2bb + dm2ab.transpose(2,3,0,1)

    hcore_deriv = mc_grad.hcore_generator(mol)
    s1a = mc_grad.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    eri1 = mol.intor('int2e_ip1', comp=3, aosym='s1')
    eri1 = eri1.reshape(3, mol.nao, mol.nao, mol.nao, mol.nao)
    xmo = _casci_generalized_fock(mc, mo_coeff, ci, casdm1s, casdm2s)
    xtilde = []
    for s in range(2):
        x = xmo[s].copy()
        idx = numpy.tril_indices(x.shape[0], -1)
        x[idx] = xmo[s].T[idx]
        xtilde.append(x)

    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        de[k] += numpy.einsum('xij,ij->x', hcore_deriv(ia),
                              dm1ao[0] + dm1ao[1])
        de[k] -= numpy.einsum('xpqrs,pqrs->x',
                              eri1[:,p0:p1], dm2a[p0:p1]) * 2
        de[k] -= numpy.einsum('xpqrs,pqrs->x',
                              eri1[:,p0:p1], dm2b[p0:p1]) * 2
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
    de = _skeleton_grad(mc_grad, mo_coeff, ci, casdm1s, casdm2s, atmlst)
    de += _orbital_response_grad(
        mc_grad, mo_coeff, ci, casdm1s, casdm2s, atmlst)
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
