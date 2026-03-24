#!/usr/bin/env python
import unittest
import numpy
from pyscf import gto, scf
from pyscf.lo import quao


def setUpModule():
    global mol, mf
    mol = gto.M(
        atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        basis='sto-3g',
        verbose=0,
    )
    mf = scf.RHF(mol).run()


def tearDownModule():
    global mol, mf
    del mol, mf


class TestQUAO(unittest.TestCase):

    def test_quao_construction(self):
        '''QUAOs can be constructed and have expected count.'''
        q, labels = quao.quao(mol, mf.mo_coeff, orient=False, verbose=0)
        # STO-3G: O has 1s,2s,2px,2py,2pz (5 AOs), each H has 1s (1 AO)
        # Total AOs = 7. QUAOs should be <= 7.
        self.assertGreater(q.shape[1], 0)
        self.assertEqual(q.shape[0], mol.nao_nr())
        self.assertEqual(len(labels), q.shape[1])

    def test_quao_orthonormality(self):
        '''QUAOs should be orthonormal w.r.t. AO overlap.'''
        s = mol.intor_symmetric('int1e_ovlp')
        q, _ = quao.quao(mol, mf.mo_coeff, orient=False, s=s, verbose=0)
        metric = q.T @ s @ q
        numpy.testing.assert_allclose(metric, numpy.eye(q.shape[1]), atol=1e-10)

    def test_occupations_sum(self):
        '''QUAO occupations should sum to the number of electrons.'''
        dm = mf.make_rdm1()
        s = mol.intor_symmetric('int1e_ovlp')
        q, _ = quao.quao(mol, mf.mo_coeff, orient=False, s=s, verbose=0)
        occ = quao.occupations(q, dm, s=s)
        numpy.testing.assert_allclose(numpy.sum(occ), mol.nelectron, atol=0.1)

    def test_kei_bo_symmetry(self):
        '''KEI-BO matrix should be symmetric.'''
        dm = mf.make_rdm1()
        s = mol.intor_symmetric('int1e_ovlp')
        q, _ = quao.quao(mol, mf.mo_coeff, orient=False, s=s, verbose=0)
        kei = quao.kei_bo(mol, q, dm, s=s)
        numpy.testing.assert_allclose(kei, kei.T, atol=1e-10)

    def test_hybridization_sum(self):
        '''Hybridization percentages should sum to ~1 for each QUAO.'''
        s = mol.intor_symmetric('int1e_ovlp')
        q, labels = quao.quao(mol, mf.mo_coeff, orient=False, s=s, verbose=0)
        hyb = quao.hybridization(mol, q, s=s)
        for i in range(hyb.shape[0]):
            self.assertAlmostEqual(numpy.sum(hyb[i]), 1.0, places=2)

    def test_analyze(self):
        '''Full analyze() should run without errors.'''
        dm = mf.make_rdm1()
        result = quao.analyze(mol, mf.mo_coeff, dm, orient=False, verbose=0)
        self.assertIn('quao_coeff', result)
        self.assertIn('labels', result)
        self.assertIn('occupations', result)
        self.assertIn('kei_bo', result)
        self.assertIn('hybridization', result)

    def test_with_mo_occ_filter(self):
        '''QUAOs with mo_occ filter should use only occupied MOs.'''
        q_all, _ = quao.quao(mol, mf.mo_coeff, orient=False, verbose=0)
        q_occ, _ = quao.quao(mol, mf.mo_coeff, mo_occ=mf.mo_occ,
                             orient=False, verbose=0)
        # With occ filter we use fewer MOs, might get fewer QUAOs
        self.assertGreater(q_occ.shape[1], 0)

    def test_orient(self):
        '''Orientation should not break orthonormality.'''
        s = mol.intor_symmetric('int1e_ovlp')
        q, _ = quao.quao(mol, mf.mo_coeff, orient=True, s=s, verbose=0)
        metric = q.T @ s @ q
        numpy.testing.assert_allclose(metric, numpy.eye(q.shape[1]), atol=1e-10)

    def test_hydrogen_s_character(self):
        '''H atom QUAOs should be predominantly s-character.'''
        s = mol.intor_symmetric('int1e_ovlp')
        q, labels = quao.quao(mol, mf.mo_coeff, orient=False, s=s, verbose=0)
        hyb = quao.hybridization(mol, q, s=s)
        for i, (ia, elem, l, m) in enumerate(labels):
            if elem == 'H':
                # H should have dominant s character
                self.assertGreater(hyb[i, 0], 0.5,
                                   f'H QUAO {i} has low s-character: {hyb[i]}')


class TestQUAOLargerBasis(unittest.TestCase):

    def test_631g(self):
        '''QUAO should work with a larger basis set.'''
        mol2 = gto.M(
            atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='6-31g',
            verbose=0,
        )
        mf2 = scf.RHF(mol2).run()
        dm = mf2.make_rdm1()
        result = quao.analyze(mol2, mf2.mo_coeff, dm, orient=False, verbose=0)
        occ = result['occupations']
        numpy.testing.assert_allclose(numpy.sum(occ), mol2.nelectron, atol=0.2)


if __name__ == '__main__':
    unittest.main()
