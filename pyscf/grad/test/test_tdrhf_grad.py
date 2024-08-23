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

import unittest
import numpy
from pyscf import gto, scf, lib
from pyscf import tdscf
from pyscf import ao2mo
from pyscf.scf import cphf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import tdrhf as tdrhf_grad

#
# LR-TDHF TDA gradients
#
def tda_kernel(tdgrad, z):
    mol = tdgrad.mol
    mf = tdgrad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    z = z[0].reshape(nocc,nvir).T * numpy.sqrt(2)
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    def fvind(x):
        #v_mo  = numpy.einsum('iabj,xai->xbj', g[:nocc,nocc:,nocc:,:nocc], x)
        #v_mo += numpy.einsum('aibj,xai->xbj', g[nocc:,:nocc,nocc:,:nocc], x)
        dm = numpy.einsum('pi,xij,qj->xpq', orbv, x, orbo)
        vj, vk = mf.get_jk(mol, (dm+dm.transpose(0,2,1)))
        v_ao = vj * 2 - vk
        v_mo = numpy.einsum('pi,xpq,qj->xij', orbv, v_ao, orbo).reshape(3,-1)
        return v_mo

    h1 = rhf_grad.get_hcore(mol)
    s1 = rhf_grad.get_ovlp(mol)

    eri1 = -mol.intor('int2e_ip1', aosym='s1', comp=3)
    eri1 = eri1.reshape(3,nao,nao,nao,nao)
    eri0 = ao2mo.kernel(mol, mo_coeff)
    eri0 = ao2mo.restore(1, eri0, nmo).reshape(nmo,nmo,nmo,nmo)
    g = eri0 * 2 - eri0.transpose(0,3,2,1)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[nocc:]

    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((mol.natm,3))
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        h1ao = h1ao + h1ao.transpose(0,2,1)
        h1mo = numpy.einsum('pi,xpq,qj->xij', mo_coeff, h1ao, mo_coeff)
        s1mo = numpy.einsum('pi,xpq,qj->xij', mo_coeff[p0:p1], s1[:,p0:p1], mo_coeff)
        s1mo = s1mo + s1mo.transpose(0,2,1)

        f1 = h1mo - numpy.einsum('xpq,pq->xpq', s1mo, zeta)
        f1-= numpy.einsum('klpq,xlk->xpq', g[:nocc,:nocc], s1mo[:,:nocc,:nocc])

        eri1a = eri1.copy()
        eri1a[:,:p0] = 0
        eri1a[:,p1:] = 0
        eri1a = eri1a + eri1a.transpose(0,2,1,3,4)
        eri1a = eri1a + eri1a.transpose(0,3,4,1,2)
        g1 = numpy.einsum('xpjkl,pi->xijkl', eri1a, mo_coeff)
        g1 = numpy.einsum('xipkl,pj->xijkl', g1, mo_coeff)
        g1 = numpy.einsum('xijpl,pk->xijkl', g1, mo_coeff)
        g1 = numpy.einsum('xijkp,pl->xijkl', g1, mo_coeff)
        g1 = g1 * 2 - g1.transpose(0,1,4,3,2)
        f1 += numpy.einsum('xkkpq->xpq', g1[:,:nocc,:nocc])
        f1ai = f1[:,nocc:,:nocc].copy()

        c1 = s1mo * -.5
        c1vo = cphf.solve(fvind, mo_energy, mo_occ, f1ai, max_cycle=50)[0]
        c1[:,nocc:,:nocc] = c1vo
        c1[:,:nocc,nocc:] = -(s1mo[:,nocc:,:nocc]+c1vo).transpose(0,2,1)
        f1 += numpy.einsum('kapq,xak->xpq', g[:nocc,nocc:], c1vo)
        f1 += numpy.einsum('akpq,xak->xpq', g[nocc:,:nocc], c1vo)

        e1  = numpy.einsum('xaijb,ai,bj->x', g1[:,nocc:,:nocc,:nocc,nocc:], z, z)
        e1 += numpy.einsum('xab,ai,bi->x', f1[:,nocc:,nocc:], z, z)
        e1 -= numpy.einsum('xij,ai,aj->x', f1[:,:nocc,:nocc], z, z)

        g1  = numpy.einsum('pjkl,xpi->xijkl', g, c1)
        g1 += numpy.einsum('ipkl,xpj->xijkl', g, c1)
        g1 += numpy.einsum('ijpl,xpk->xijkl', g, c1)
        g1 += numpy.einsum('ijkp,xpl->xijkl', g, c1)
        e1 += numpy.einsum('xaijb,ai,bj->x', g1[:,nocc:,:nocc,:nocc,nocc:], z, z)

        de[ia] = e1

    return de


def setUpModule():
    global mol, pmol, mf, nstates
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.)], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()
    pmol = mol.copy()
    mf = scf.RHF(mol).set(conv_tol=1e-12).run()
    nstates = 5 # to ensure the first 3 TDSCF states are converged

def tearDownModule():
    global mol, pmol, mf
    mol.stdout.close()
    del mol, pmol, mf

class KnownValues(unittest.TestCase):
    def test_tda_singlet(self):
        td = tdscf.TDA(mf).run(nstates=nstates)
        g1ref = tda_kernel(td.nuc_grad_method(), td.xy[2]) + mf.nuc_grad_method().kernel()

        tdg = td.nuc_grad_method().as_scanner()
#[[ 0  0  -2.67023832e-01]
# [ 0  0   2.67023832e-01]]
        self.assertAlmostEqual(lib.fp(tdg.kernel(td.xy[0])), 0.18686561181358813, 6)

        g1 = tdg(mol.atom_coords(), state=3)[1]
        self.assertAlmostEqual(abs(g1-g1ref).max(), 0, 6)
        self.assertAlmostEqual(g1[0,2], -0.23226123352352346, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 6)

        self.assertAlmostEqual(abs(tdg.kernel(state=0) -
                                   mf.nuc_grad_method().kernel()).max(), 0, 8)

    def test_tda_triplet(self):
        td = tdscf.TDA(mf).run(singlet=False, nstates=nstates)
        tdg = td.nuc_grad_method()
# [[ 0  0  -2.81048403e-01]
#  [ 0  0   2.81048403e-01]]
        self.assertAlmostEqual(lib.fp(tdg.kernel(state=1)), 0.19667995802487931, 6)

        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.472965206465775, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tdhf_singlet(self):
        td = tdscf.TDDFT(mf).run(nstates=nstates)
        tdg = td.nuc_grad_method()
# [[ 0  0  -2.71041021e-01]
#  [ 0  0   2.71041021e-01]]
        self.assertAlmostEqual(lib.fp(tdg.kernel(state=1)), 0.18967687762609461, 6)

        g1 = tdg.kernel(td.xy[2])
        self.assertAlmostEqual(g1[0,2], -0.25240005833657309, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tdhf_triplet(self):
        td = tdscf.TDDFT(mf).run(singlet=False, nstates=nstates)
        tdg = td.nuc_grad_method()
# [[ 0  0  -2.86250870e-01]
#  [ 0  0   2.86250870e-01]]
        self.assertAlmostEqual(lib.fp(tdg.kernel(state=1)), 0.20032088639558535, 6)

        g1 = tdg.kernel(td.xy[2])
        self.assertAlmostEqual(g1[0,2], -0.5408133995976914, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_symmetrize(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True)
        g = mol.RHF.run().TDA().run(nstates=1).Gradients().kernel(state=1)
        self.assertAlmostEqual(lib.fp(g), -0.07887074405221786, 6)


if __name__ == "__main__":
    print("Full Tests for TD-RHF gradients")
    unittest.main()
