from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf.krhf_slow import PhysERIGamma, PhysERI4Gamma, PhysERI8Gamma, PhysERI,\
    PhysERI4, PhysERI8, build_matrix, eig, kernel_gamma, kernel, k_nocc
from pyscf.pbc.tdscf.krhf_slow_supercell import PhysERI4 as PhysERI4_S
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


class DiamondTest(unittest.TestCase):
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

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k)
        model_krhf.kernel()

        cls.td_model_krhf = td_model_krhf = KTDHF(model_krhf)
        td_model_krhf.kernel()

        cls.ref_m_gamma = retrieve_m(td_model_krhf)
        cls.ref_e_gamma = td_model_krhf.e

        cls.ref_e, _ = eig(build_matrix(PhysERI4_S(model_krhf)))

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_eri_gamma(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (PhysERIGamma, PhysERI4Gamma, PhysERI8Gamma, PhysERI, PhysERI4, PhysERI8):
            try:

                if issubclass(eri, PhysERI):
                    e = eri(self.model_krhf, 0)
                else:
                    e = eri(self.model_krhf)
                m = build_matrix(e)

                testing.assert_allclose(self.ref_m_gamma, m, atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel_gamma(self):
        """Tests default eig kernel behavior."""
        vals, vecs = kernel_gamma(self.model_krhf, driver='eig')
        testing.assert_allclose(vals[:len(self.ref_e_gamma)], self.ref_e_gamma, atol=1e-7)

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (PhysERI,):
            try:
                vals = []

                for k in range(self.k):
                    e = eri(self.model_krhf, k)
                    m = build_matrix(e)
                    vals.append(eig(m)[0])

                vals = numpy.concatenate(vals).real
                vals.sort()

                testing.assert_allclose(self.ref_e, vals, atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals = []
        for k in range(self.k):
            vals.append(kernel(self.model_krhf, k, driver='eig')[0])
        vals = numpy.concatenate(vals).real
        vals.sort()
        testing.assert_allclose(vals, self.ref_e, atol=1e-7)

