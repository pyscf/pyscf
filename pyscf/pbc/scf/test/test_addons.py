#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
cell = pbcgto.Cell()
cell.atom = '''
He 0 0 1
He 1 0 1
'''
cell.basis = '3-21g'
cell.a = numpy.eye(3) * 2
cell.gs = [7] * 3
cell.verbose = 5
cell.output = '/dev/null'
cell.build()
nao = cell.nao_nr()

class KnowValues(unittest.TestCase):
    def test_krhf_smearing(self):
        mf = pscf.KRHF(cell, cell.make_kpts([2,1,1]))
        nkpts = len(mf.kpts)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy_kpts = numpy.array([numpy.arange(nao)*.2+numpy.cos(i+.5)*.1
                                      for i in range(nkpts)])
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1656394960533021, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 2.4500185794942135, 9)

    def test_kuhf_smearing(self):
        mf = pscf.KUHF(cell, cell.make_kpts([2,1,1]))
        nkpts = len(mf.kpts)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy_kpts = numpy.array([numpy.arange(nao)*.2+numpy.cos(i+.5)*.1
                                      for i in range(nkpts)])
        mo_energy_kpts = numpy.array([mo_energy_kpts,
                                      mo_energy_kpts+numpy.cos(mo_energy_kpts)*.02])
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1803390081500869, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 2.4646236868793121, 9)

    def test_rhf_smearing(self):
        mf = pscf.RHF(cell)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy = numpy.arange(nao)*.2+numpy.cos(.5)*.1
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.0922723199786408, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 1.1023835704293432, 9)

    def test_uhf_smearing(self):
        mf = pscf.UHF(cell)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy = numpy.arange(nao)*.2+numpy.cos(.5)*.1
        mo_energy = numpy.array([mo_energy, mo_energy+numpy.cos(mo_energy)*.02])
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.1007387905421022, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 1.1173540623523119, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.addons")
    unittest.main()
