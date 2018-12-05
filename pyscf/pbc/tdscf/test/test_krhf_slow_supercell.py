from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RHF, KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf.krhf_slow_supercell import PhysERI, PhysERI4, PhysERI8, build_matrix, eig, kernel, k_nocc
from pyscf.pbc.tools.pbc import super_cell
from pyscf.tdscf.rhf_slow import PhysERI4 as PhysERI4_mol, kernel as kernel_mol

import unittest
from numpy import testing
import numpy


def retrieve_m(model, **kwargs):
    vind, hdiag = model.gen_vind(model._scf, **kwargs)
    size = model.init_guess(model._scf, 1).shape[1]
    return vind(numpy.eye(size)).T


def sign(x):
    return x / abs(x)


def make_phase_well_defined(model):
    if "kpts" in dir(model):
        for i in model.mo_coeff:
            i /= sign(i[0])[numpy.newaxis, :]
    else:
        model.mo_coeff /= sign(model.mo_coeff[0])[numpy.newaxis, :]


def ov_order(model):
    nocc = k_nocc(model)
    e_occ = tuple(e[:o] for e, o in zip(model.mo_energy, nocc))
    e_virt = tuple(e[o:] for e, o in zip(model.mo_energy, nocc))
    sort_o = []
    sort_v = []
    for o in e_occ:
        for v in e_virt:
            _v, _o = numpy.meshgrid(v, o)
            sort_o.append(_o.reshape(-1))
            sort_v.append(_v.reshape(-1))
    sort_o, sort_v = numpy.concatenate(sort_o), numpy.concatenate(sort_v)
    vals = numpy.array(
        list(zip(sort_o, sort_v)),
        dtype=[('o', sort_o[0].dtype), ('v', sort_v[0].dtype)]
    )
    result = numpy.argsort(vals, order=('o', 'v'))
    # Double for other blocks
    return numpy.concatenate([result, result + len(result)])


class DiamondTestGamma(unittest.TestCase):
    """Compare this (supercell_slow) @Gamma vs reference."""
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

        cls.model_krhf = model_krhf = KRHF(cell)
        model_krhf.kernel()
        # make_phase_well_defined(model_krhf)

        cls.td_model_krhf = td_model_krhf = KTDHF(model_krhf)
        td_model_krhf.nroots = 5
        td_model_krhf.kernel()

        cls.ref_m_krhf = retrieve_m(td_model_krhf)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (PhysERI, PhysERI4, PhysERI8):
            e = eri(self.model_krhf)
            m = build_matrix(e)
            try:
                testing.assert_allclose(self.ref_m_krhf, m, atol=1e-14)
                vals, vecs = eig(m, nroots=self.td_model_krhf.nroots)
                testing.assert_allclose(vals, self.td_model_krhf.e, atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = kernel(self.model_krhf, driver='eig', nroots=self.td_model_krhf.nroots)
        testing.assert_allclose(vals, self.td_model_krhf.e, atol=1e-5)


class DiamondTestNoGamma(unittest.TestCase):
    """Compare this (supercell_slow) @non-Gamma 1kp vs rhf_slow."""
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

        k = cell.get_abs_kpts((.1, .2, .3))

        # The Gamma-point reference
        cls.model_rhf = model_rhf = RHF(cell, k)
        model_rhf.conv_tol = 1e-14
        model_rhf.kernel()
        make_phase_well_defined(model_rhf)

        cls.ref_m = build_matrix(PhysERI4_mol(model_rhf))
        cls.ref_e, _ = kernel_mol(model_rhf)

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k)
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()
        make_phase_well_defined(model_krhf)

        testing.assert_allclose(model_rhf.mo_energy, model_krhf.mo_energy[0])
        testing.assert_allclose(model_rhf.mo_coeff, model_krhf.mo_coeff[0])

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krhf
        del cls.model_rhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (PhysERI, PhysERI4):
            e = eri(self.model_krhf)
            m = build_matrix(e)

            try:
                testing.assert_allclose(self.ref_m, m, atol=1e-10)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = kernel(self.model_krhf, driver='eig')
        testing.assert_allclose(vals, self.ref_e, atol=1e-7)


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (supercell_slow) @2kp vs rhf_slow (2x1x1 supercell)."""
    k = 2
    k_c = (0, 0, 0)

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

        k = cell.make_kpts([cls.k, 1, 1], scaled_center=cls.k_c)

        # The Gamma-point reference
        cls.model_rhf = model_rhf = RHF(super_cell(cell, [cls.k, 1, 1]), kpt=k[0])
        model_rhf.conv_tol = 1e-14
        model_rhf.kernel()
        make_phase_well_defined(model_rhf)

        cls.ref_m = build_matrix(PhysERI4_mol(model_rhf))
        cls.ref_e, _ = kernel_mol(model_rhf)

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k)
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()
        make_phase_well_defined(model_krhf)
        ke = numpy.concatenate(model_krhf.mo_energy)
        ke.sort()

        # Make sure mo energies are the same
        testing.assert_allclose(model_rhf.mo_energy, ke)

        # Make sure no degeneracies are present
        testing.assert_array_less(1e-4, ke[1:] - ke[:-1])

        cls.ov_order = ov_order(model_krhf)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krhf
        del cls.model_rhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (PhysERI, PhysERI4, PhysERI8):
            e = eri(self.model_krhf)
            m = build_matrix(e)

            try:
                testing.assert_allclose(self.ref_m, m[numpy.ix_(self.ov_order, self.ov_order)], atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = kernel(self.model_krhf, driver='eig')
        testing.assert_allclose(vals, self.ref_e, atol=1e-7)


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (supercell_slow) @3kp vs rhf_slow (3x1x1 supercell)."""
    k = 3
    k_c = (.1, 0, 0)
