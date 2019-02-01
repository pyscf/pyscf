from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RKS, KRKS
from pyscf.pbc.tdscf import KTDDFT
from pyscf.pbc.tools.pbc import super_cell
from pyscf.tdscf.rhf_slow import eig

from test_common import retrieve_m, test_m_complex_support as cstest, adjust_mf_phase, ov_order, assert_vectors_close, tdhf_frozen_mask

import unittest
from numpy import testing
import numpy


class DiamondTestGamma(unittest.TestCase):
    """--- Testing TDHF--- TBD:replace this"""
    k = [2, 1, 1]

    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        # cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        cls.model_rks = model_rks = KRKS(super_cell(cell, cls.k))
        model_rks.kernel()

        cls.model_krks = model_krks = KRKS(cell, cell.make_kpts(cls.k))
        model_krks.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krks
        del cls.model_rks
        del cls.cell

    def test_eig(self):
        """Tests eigvals."""
        nroots = 5

        td_model_rks = KTDDFT(self.model_rks)
        td_model_rks.nroots = 2 * nroots
        td_model_rks.kernel()
        e1, v1 = eig(retrieve_m(td_model_rks), nroots=2*nroots)

        td_model_krks = KTDDFT(self.model_krks)
        td_model_krks.nroots = nroots
        td_model_krks.kernel()

        print td_model_rks.e
        print e1
        print td_model_krks.e

        cstest(td_model_rks)
        cstest(td_model_krks)
