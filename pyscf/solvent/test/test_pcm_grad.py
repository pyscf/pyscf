# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy
from pyscf import scf, gto, mcscf, solvent, cc, lib, tddft, tdscf
from pyscf.solvent import pcm
from pyscf.solvent.grad import pcm as pcm_grad

def setUpModule():
    global mol, epsilon, lebedev_order
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 35.9
    lebedev_order = 3

    global dx, mol0, mol1, mol2
    dx = 1e-4
    mol0 = gto.M(atom='H 0 0 0; H 0 1 1.2; H 1. .1 0; H .5 .5 1', unit='B', verbose=0)
    mol1 = gto.M(atom='H 0 0 %g; H 0 1 1.2; H 1. .1 0; H .5 .5 1'%(-dx), unit='B', verbose=0)
    mol2 = gto.M(atom='H 0 0 %g; H 0 1 1.2; H 1. .1 0; H .5 .5 1'%dx, unit='B', verbose=0)
    dx = 2.0*dx

def tearDownModule():
    global mol, mol0, mol1, mol2
    mol.stdout.close()
    del mol, mol0, mol1, mol2

class KnownValues(unittest.TestCase):
    def test_dA_dF(self):
        cm = pcm.PCM(mol)
        cm.lebedev_order = 3
        cm.method = 'IEF-PCM'
        cm.build()

        dF, dA = pcm_grad.get_dF_dA(cm.surface)
        dD, dS, dSii = pcm_grad.get_dD_dS(cm.surface, dF, with_S=True, with_D=True)

        def get_FADS(mol):
            mol.build()
            cm = pcm.PCM(mol)
            cm.lebedev_order = 3
            cm.method = 'IEF-PCM'
            cm.build()
            F = cm.surface['switch_fun']
            A = cm._intermediates['A']
            D = cm._intermediates['D']
            S = cm._intermediates['S']
            return F, A, D, S

        eps = 1e-5
        for ia in range(mol.natm):
            p0,p1 = cm.surface['gslice_by_atom'][ia]
            for j in range(3):
                coords = mol.atom_coords(unit='B')
                coords[ia,j] += eps
                mol.set_geom_(coords, unit='B')
                mol.build()
                F0, A0, D0, S0 = get_FADS(mol)

                coords[ia,j] -= 2.0*eps
                mol.set_geom_(coords, unit='B')
                mol.build()
                F1, A1, D1, S1 = get_FADS(mol)

                coords[ia,j] += eps
                mol.set_geom_(coords, unit='B')
                dF0 = (F0 - F1)/(2.0*eps)
                dA0 = (A0 - A1)/(2.0*eps)
                dD0 = (D0 - D1)/(2.0*eps)
                dS0 = (S0 - S1)/(2.0*eps)

                assert numpy.linalg.norm(dF0 - dF[:,ia,j]) < 1e-8
                assert numpy.linalg.norm(dA0 - dA[:,ia,j]) < 1e-8

                # the diagonal entries are calcualted separately
                assert numpy.linalg.norm(dSii[:,ia,j] - numpy.diag(dS0)) < 1e-8
                numpy.fill_diagonal(dS0, 0)

                dS_ia = numpy.zeros_like(dS0)
                dS_ia[p0:p1] = dS[p0:p1,:,j]
                dS_ia[:,p0:p1] -= dS[:,p0:p1,j]
                assert numpy.linalg.norm(dS0 - dS_ia) < 1e-8

                dD_ia = numpy.zeros_like(dD0)
                dD_ia[p0:p1] = dD[p0:p1,:,j]
                dD_ia[:,p0:p1] -= dD[:,p0:p1,j]
                assert numpy.linalg.norm(dD0 - dD_ia) < 1e-8

    def test_grad_CPCM(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 3
        cm.method = 'C-PCM'
        mf = scf.RHF(mol).PCM(cm)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        e_tot = mf.kernel()

        g = mf.nuc_grad_method()
        grad = g.kernel()

        g0 = numpy.asarray([
             [0.49773047433563E-15,  -0.12128126037559E-15,  -0.58936988992306E-01],
             [0.22810111996954E-01,  -0.68951901317025E-17,   0.29468494708267E-01],
            [-0.22810111996957E-01,   0.12949813945902E-15,   0.29468494708266E-01]])

        print(f"Gradient error in CPCM: {numpy.linalg.norm(g0 - grad)}")
        assert numpy.linalg.norm(g0 - grad) < 1e-9

    def test_grad_COSMO(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 3
        cm.method = 'COSMO'
        mf = scf.RHF(mol).PCM(cm)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        e_tot = mf.kernel()

        g = mf.nuc_grad_method()
        grad = g.kernel()

        g0 = numpy.asarray(
            [[-1.33560836e-16,  8.70874355e-17, -5.89638726e-02],
             [ 2.28202396e-02,  2.63784344e-17,  2.94819363e-02],
             [-2.28202396e-02, -1.08799896e-16,  2.94819363e-02]])

        print(f"Gradient error in COSMO: {numpy.linalg.norm(g0 - grad)}")
        assert numpy.linalg.norm(g0 - grad) < 1e-9

    def test_grad_IEFPCM(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 3
        cm.method = 'IEF-PCM'
        mf = scf.RHF(mol).PCM(cm)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        e_tot = mf.kernel()

        g = mf.nuc_grad_method()
        grad = g.kernel()

        g0 = numpy.asarray([
             [0.18357915015649E-14,   0.14192681822347E-15,  -0.58988087999658E-01],
             [0.22822709179063E-01,  -0.10002010417168E-15,   0.29494044211805E-01],
            [-0.22822709179066E-01,  -0.31051364515588E-16,   0.29494044211806E-01]])
        print(f"Gradient error in IEFPCM: {numpy.linalg.norm(g0 - grad)}")
        assert numpy.linalg.norm(g0 - grad) < 1e-9

    def test_grad_SSVPE(self):
        cm = pcm.PCM(mol)
        cm.eps = epsilon
        cm.verbose = 0
        cm.lebedev_order = 3
        cm.method = 'SS(V)PE'
        mf = scf.RHF(mol).PCM(cm)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        e_tot = mf.kernel()

        g = mf.nuc_grad_method()
        grad = g.kernel()

        g0 = numpy.asarray([
             [0.76104817971710E-15,   0.11185701540547E-15,  -0.58909172879217E-01],
             [0.22862990009767E-01,  -0.13861633974903E-15,   0.29454586651678E-01],
            [-0.22862990009769E-01,   0.34988765678591E-16,   0.29454586651679E-01]])
        print(f"Gradient error in SS(V)PE: {numpy.linalg.norm(g0 - grad)}")
        assert numpy.linalg.norm(g0 - grad) < 1e-9

    def test_casci_grad(self):
        mf = scf.RHF(mol0).PCM().run()
        mc = solvent.PCM(mcscf.CASCI(mf, 2, 2))
        e, de = mc.nuc_grad_method().as_scanner()(mol0)

        mf = scf.RHF(mol1).run()
        mc1 = solvent.PCM(mcscf.CASCI(mf, 2, 2)).run()
        e1 = mc1.e_tot
        mf = scf.RHF(mol2).run()
        mc2 = solvent.PCM(mcscf.CASCI(mf, 2, 2)).run()
        e2 = mc2.e_tot
        self.assertAlmostEqual((e2-e1)/dx, de[0,2], 3)


    def test_casscf_grad(self):
        mf = scf.RHF(mol0).PCM().run()
        mc = solvent.PCM(mcscf.CASSCF(mf, 2, 2)).set(conv_tol=1e-9)
        mc_g = mc.nuc_grad_method().as_scanner()
        e, de = mc_g(mol0)

        mf = scf.RHF(mol1).run()
        mc1 = solvent.PCM(mcscf.CASSCF(mf, 2, 2)).run(conv_tol=1e-9)
        e1 = mc1.e_tot
        mf = scf.RHF(mol2).run()
        mc2 = solvent.PCM(mcscf.CASSCF(mf, 2, 2)).run(conv_tol=1e-9)
        e2 = mc2.e_tot

        self.assertAlmostEqual((e2-e1)/dx, de[0,2], 2)

    def test_ccsd_grad(self):
        mf = scf.RHF(mol0).PCM()
        mf.conv_tol = 1e-12
        mf.kernel()
        mycc = cc.CCSD(mf).PCM()
        mycc.with_solvent.conv_tol = 1e-12
        mycc.conv_tol = 1e-12
        mycc.kernel()
        e, de = mycc.nuc_grad_method().as_scanner()(mol0)

        mf = scf.RHF(mol1)
        mf.conv_tol = 1e-12
        mf.kernel()
        mycc1 = solvent.PCM(cc.CCSD(mf))
        mycc1.with_solvent.conv_tol = 1e-12
        mycc1.conv_tol = 1e-12
        mycc1.kernel()
        e1 = mycc1.e_tot

        mf = scf.RHF(mol2)
        mf.conv_tol = 1e-12
        mf.run()
        mycc2 = solvent.PCM(cc.CCSD(mf))
        mycc2.with_solvent.conv_tol = 1e-12
        mycc2.conv_tol = 1e-12
        mycc2.kernel()
        e2 = mycc2.e_tot

        self.assertAlmostEqual((e2-e1)/dx, de[0,2], 3)

if __name__ == "__main__":
    print("Full Tests for Gradient of PCMs")
    unittest.main()
