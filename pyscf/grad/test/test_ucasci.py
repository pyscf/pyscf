#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import mcscf
from pyscf import ci
from pyscf.lib import logger
from pyscf.grad.test.test_casci import _check_fd_grad
from pyscf.grad.test.test_casci import _move_atom


# Helper functions to build an independent reference full spin RDM and
# generic pair-symmetric two-electron tensors. The RDM-intermediate tests
# use them to check the UCASCI RDM -> UCCSD d1/d2 conversion
# with a more literal construction
def _pair_symmetric_eri(nmo, seed):
    rng = numpy.random.default_rng(seed)
    eri = rng.standard_normal((nmo,nmo,nmo,nmo))
    eri = (eri + eri.transpose(1,0,2,3) +
           eri.transpose(0,1,3,2) + eri.transpose(1,0,3,2)) * .25
    return (eri + eri.transpose(2,3,0,1)) * .5


def _pair_symmetric_eri_ab(nmoa, nmob, seed):
    rng = numpy.random.default_rng(seed)
    eri = rng.standard_normal((nmoa,nmoa,nmob,nmob))
    return (eri + eri.transpose(1,0,2,3) +
            eri.transpose(0,1,3,2) + eri.transpose(1,0,3,2)) * .25


def _make_full_rdm12s(casdm1s, casdm2s, nmo, ncore, ncas):
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


class KnownValues(unittest.TestCase):
    # Closed shell UCASCI matches restricted CASCI gradient.
    def test_full_active_matches_restricted(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        rcas = mcscf.CASCI(mf, mf.mo_coeff.shape[1], mol.nelectron).run()
        ref = rcas.nuc_grad_method().kernel()

        mf = scf.UHF(mol).run(conv_tol=1e-12)
        ucas = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec).run()
        grad = ucas.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(ref-grad).max(), 0, 9)

    def test_full_active_one_electron_fd(self):
        atom = '''
        H 0 0 0
        H 0 0 1
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', charge=1, spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec).run()
            return mc

        mc = run()
        grad = mc.nuc_grad_method()
        self.assertIsNone(grad.state)
        with self.assertRaises(AssertionError):
            grad.kernel(state=1)
        _check_fd_grad(self, lambda: grad.kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6)

    # One core electron
    def test_inactive_openshell_fd(self):
        atom = '''
        H 0 0 0
        H 0 0 1
        H 0 1 0
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
            return mc

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 5e-7)

    # UCASCI and UCISD should be the same for a 2-electron case.
    def test_ucasci_matches_ucisd(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        myci = ci.UCISD(mf, frozen=[[0], [2]]).run()
        self.assertAlmostEqual(mc.e_tot, myci.e_tot, 12)
        g_ucas = mc.nuc_grad_method().kernel()
        g_ucis = myci.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g_ucas - g_ucis).max(), 0, 10)


    # Convert UCASCI RDMs into UCCSD d1/d2 intermediates and back.
    def test_rdm_intermediates_round_trip(self):
        from pyscf.cc import uccsd_rdm
        from pyscf.grad import ucasci
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        casdm1s, casdm2s = mc.fcisolver.make_rdm12s(mc.ci, mc.ncas,
                                                     mc.nelecas)
        d1, d2 = ucasci._casci_active_rdm_to_uccsd_intermediates(
            casdm1s, casdm2s, mc.nelecas)
        full1, full2 = _make_full_rdm12s(
            casdm1s, casdm2s, mf.mo_coeff[0].shape[1], mc.ncore, mc.ncas)
        with ucasci._uccsd_env(mc, mc.mo_coeff):
            rdm1 = uccsd_rdm._make_rdm1(mc, d1, with_frozen=True)
            rdm2 = uccsd_rdm._make_rdm2(mc, d1, d2, with_frozen=True)
        self.assertAlmostEqual(max(abs(rdm1[0]-full1[0]).max(),
                                   abs(rdm1[1]-full1[1]).max()), 0, 12)
        self.assertAlmostEqual(max(abs(rdm2[i]-full2[i]).max()
                                   for i in range(3)), 0, 12)

    # Check the larger-space RDM conversion with generic pair-symmetric
    # two-electron tensor contractions.
    def test_rdm_intermediates_contraction(self):
        from pyscf.cc import uccsd_rdm
        from pyscf.grad import ucasci
        mol = gto.M(atom='''
                    C  0.00  0.00  0.00
                    H  0.10  0.02  1.09
                    H  1.02  0.08 -0.35
                    H -0.42  0.96 -0.28''',
                    basis='6-31g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, 7, (3,2), ncore=(2,2)).run()
        casdm1s, casdm2s = mc.fcisolver.make_rdm12s(mc.ci, mc.ncas,
                                                     mc.nelecas)
        d1, d2 = ucasci._casci_active_rdm_to_uccsd_intermediates(
            casdm1s, casdm2s, mc.nelecas)
        full1, full2 = _make_full_rdm12s(
            casdm1s, casdm2s, mf.mo_coeff[0].shape[1], mc.ncore, mc.ncas)
        with ucasci._uccsd_env(mc, mc.mo_coeff):
            rdm1 = uccsd_rdm._make_rdm1(mc, d1, with_frozen=True)
            rdm2 = uccsd_rdm._make_rdm2(mc, d1, d2, with_frozen=True)
        self.assertAlmostEqual(max(abs(rdm1[0]-full1[0]).max(),
                                   abs(rdm1[1]-full1[1]).max()), 0, 12)

        nmo = mf.mo_coeff[0].shape[1]
        eriaa = _pair_symmetric_eri(nmo, 1)
        eriab = _pair_symmetric_eri_ab(nmo, nmo, 2)
        eribb = _pair_symmetric_eri(nmo, 3)
        e_ref = (.5 * numpy.einsum('pqrs,pqrs', eriaa, full2[0]) +
                 numpy.einsum('pqrs,pqrs', eriab, full2[1]) +
                 .5 * numpy.einsum('pqrs,pqrs', eribb, full2[2]))
        e_test = (.5 * numpy.einsum('pqrs,pqrs', eriaa, rdm2[0]) +
                  numpy.einsum('pqrs,pqrs', eriab, rdm2[1]) +
                  .5 * numpy.einsum('pqrs,pqrs', eribb, rdm2[2]))
        self.assertAlmostEqual(e_test, e_ref, 10)

    def test_all_active_fd(self):
        atom = '''
        H 0 0 0
        H 0 0 1
        H 0 1 0
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec).run()
            return mc

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6)

    # CH3 radical against FD
    def test_ch3_fd(self):
        atom = '''
        C  0.00  0.00  0.00
        H  0.10  0.02  1.09
        H  1.02  0.08 -0.35
        H -0.42  0.96 -0.28
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='6-31g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-13)
            return mcscf.UCASCI(mf, 7, (3,2), ncore=(2,2)).run()

        mc = run()
        step = 2.5e-4
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6, step=step)

    # NH2 radical against FD
    def test_nh2_fd(self):
        atom = '''
        N 0.000  0.000 0.000
        H 0.000  0.900 0.650
        H 0.000 -0.850 0.700
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='3-21g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 6, (3,2), ncore=(2,2)).run()

        mc = run()
        step = 5e-4
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6, step=step)

    # CH2 triplet against FD
    def test_ch2_fd(self):
        atom = '''
        C 0.000 0.000  0.000
        H 0.000 0.000  1.080
        H 1.020 0.000 -0.320
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='3-21g', spin=2, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 7, (4,2), ncore=(1,1)).run()

        mc = run()
        step = 5e-4
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6, step=step)

    # CH2 triplet against FD with triple-zeta basis
    def test_ch2_tz_fd(self):
        atom = '''
        C 0.000 0.000  0.000
        H 0.000 0.000  1.080
        H 1.020 0.000 -0.320
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='cc-pVTZ', spin=2, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 7, (4,2), ncore=(1,1)).run()

        mc = run()
        step = 1e-3
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 5e-7, step=step)

    # CH3 excited states against FD
    def test_ch3_excited_state_fd(self):
        atom = '''
        C  0.00  0.00  0.00
        H  0.10  0.02  1.09
        H  1.02  0.08 -0.35
        H -0.42  0.96 -0.28
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='3-21g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, 6, (3,2), ncore=(2,2))
            mc.fcisolver.nroots = 3
            return mc.run()

        mc = run()
        step = 5e-4
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(state=1),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot[1],
                       mc.mol.natm, 2e-6, step=step)

    # CH3 state average against FD
    def test_ch3_state_average_fd(self):
        atom = '''
        C  0.00  0.00  0.00
        H  0.10  0.02  1.09
        H  1.02  0.08 -0.35
        H -0.42  0.96 -0.28
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='3-21g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, 6, (3,2), ncore=(2,2))
            mc.fcisolver.nroots = 2
            return mc.state_average_([.4, .6]).run()

        mc = run()
        step = 5e-4
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 2e-6, step=step)

    # All active excited state against FD
    def test_all_active_excited_state_fd(self):
        atom = '''
        H 0 0 0
        H 0 0 1
        H 0 1 0
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, 3, (2,1))
            mc.fcisolver.nroots = 3
            return mc.run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(state=2),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot[2],
                       mc.mol.natm, 1e-6)

    # UCASCI with symmetry against FD
    def test_symmetry_fd(self):
        atom = '''
        H 0 0 0
        H 0 0 1
        H 0 1 0
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, symmetry=True, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6)

    # X2C on open-shell OH2+ against FD
    def test_x2c_oh2_fd(self):
        atom = '''
        O 0.000000  0.000000 0.000000
        H 0.000000  0.757000 0.587000
        H 0.000000 -0.757000 0.587000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='6-31g', charge=1, spin=1, unit='Angstrom',
                        verbose=0)
            mf = scf.UHF(mol).x2c().run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 4, (2,1), ncore=(3,3)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6)

    # Error out when DF-generated UHF/UKS orbital sources are used
    def test_density_fitted_orbitals_unsupported(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=1, verbose=0)

        mf = scf.UHF(mol).density_fit().run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        self.assertTrue(getattr(mc, '_scf_df_source', False))
        with self.assertRaisesRegex(NotImplementedError, 'DF-UHF'):
            mc.nuc_grad_method().kernel()

        mf = dft.UKS(mol)
        mf.xc = 'lda,vwn'
        mf = mf.density_fit()
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0))
        self.assertTrue(getattr(mc, '_scf_df_source', False))
        with self.assertRaisesRegex(NotImplementedError, 'DF-UKS'):
            mc.nuc_grad_method().grad_elec(ci=0)

    # UKS-CASCI with one core electron
    def test_uks_casci_h3_finite_diff(self):
        def run(ia=None, ix=None, dx=0):
            coords = numpy.asarray(((0., 0., 0.),
                                    (0., 0., 1.),
                                    (0., 1., 0.)))
            if ia is not None:
                coords[ia, ix] += dx
            mol = gto.M(atom=[('H', coords[i]) for i in range(3)],
                        unit='Angstrom', basis='sto-3g', spin=1, verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()

        mc = run()
        self.assertEqual(mc.nuc_grad_method().__class__.__module__,
                         'pyscf.grad.ukscasci')
        self.assertEqual(mc.Gradients().__class__.__module__,
                         'pyscf.grad.ukscasci')
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 2e-7)

    # Error out when requesting UKS-CASCI gradients for unsupported functionals
    def test_uks_casci_unsupported_functionals(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=1, verbose=0)
        mf = dft.UKS(mol)
        mf.xc = 'lda,vwn'
        mf.run(conv_tol=1e-12)

        mf.xc = 'tpss'
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        with self.assertRaisesRegex(NotImplementedError, 'meta-GGA'):
            mc.nuc_grad_method().kernel()

        mf.xc = 'cam-b3lyp'
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        with self.assertRaisesRegex(NotImplementedError, 'range-separated'):
            mc.nuc_grad_method().kernel()

        mf.xc = 'lda,vwn'
        mf.nlc = 'vv10'
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        with self.assertRaisesRegex(NotImplementedError, 'NLC'):
            mc.nuc_grad_method().kernel()

    # CH3 radical against FD with UKS(lda,vwn) orbitals
    def test_uks_ch3_fd(self):
        atom = '''
        C  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.090000
        H  1.026719  0.000000 -0.363333
        H -0.513360  0.889165 -0.363333
        '''

        def run(ia=None, ix=None, dx=0):
            mol0 = gto.M(atom=atom, basis='3-21g', spin=1,
                         unit='Angstrom', verbose=0)
            if ia is not None:
                coords = mol0.atom_coords(unit='Angstrom')
                coords[ia,ix] += dx
                mol0 = gto.M(atom=[(mol0.atom_symbol(i), coords[i])
                                   for i in range(mol0.natm)],
                             basis='3-21g', spin=1, unit='Angstrom',
                             verbose=0)
            mf = dft.UKS(mol0)
            mf.xc = 'lda,vwn'
            mf.grids.level = 1
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 6, (3,2), ncore=(2,2)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 3e-6)

    # NH2 radical against FD with UKS(lda,vwn) orbitals
    def test_uks_nh2_fd(self):
        atom = '''
        N  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.030000
        H  0.968000  0.000000 -0.344000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='3-21g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.grids.level = 1
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 6, (3,2), ncore=(2,2)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 3e-6)

    # CH2 triplet against FD with UKS(pbe,pbe)
    def test_uks_ch2_fd(self):
        atom = '''
        C  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.080000
        H  1.020000  0.000000 -0.360000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='3-21g', spin=2, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'pbe,pbe'
            mf.grids.level = 3
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 7, (4,2), ncore=(1,1)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 3e-6)

    # one-electron UKS-CASCI against FD
    def test_uks_emptybeta_fd(self):
        atom = '''
        Li 0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.650000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, charge=1,
                        unit='Angstrom', verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.grids.level = 4
            mf.run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 1, (1,0), ncore=(1,1)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-7)

    def test_uks_all_active_fd(self):
        atom = '''
        H  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.000000
        H  0.000000  1.050000  0.100000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 3, (2,1), ncore=(0,0)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-8)

    # UKS-CASCI gradient against FD when using B3LYP.
    def test_uks_b3lyp_fd(self):
        def run(ia=None, ix=None, dx=0):
            coords = numpy.asarray(((0., 0., 0.),
                                    (0., 0., 1.),
                                    (0., 1., 0.)))
            if ia is not None:
                coords[ia, ix] += dx
            mol = gto.M(atom=[('H', coords[i]) for i in range(3)],
                        unit='Angstrom', basis='sto-3g', spin=1,
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'b3lyp'
            mf.grids.level = 1
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 3e-6)

    # excited state UKS-CASCI gradients against FD.
    def test_uks_casci_excited_state_fd(self):
        atom = '''
        H  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.000000
        H  0.000000  1.000000  0.150000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-11)
            mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0))
            mc.fcisolver.nroots = 3
            return mc.run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(state=1),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot[1],
                       mc.mol.natm, 5e-7)

    # state-averaged UKS(lda,vwn)-CASCI gradients against FD.
    def test_uks_state_average_fd(self):
        atom = '''
        H  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.000000
        H  0.000000  1.000000  0.150000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-11)
            mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0))
            mc.fcisolver.nroots = 2
            return mc.state_average_([.3, .7]).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 5e-7)

    # Closed shell UKS-CASCI matches RKS-CASCI gradient.
    def test_ukscasci_matches_restricted(self):
        mol = gto.M(atom='''
                    O 0.000  0.000 0.000
                    H 0.000 -0.757 0.587
                    H 0.000  0.757 0.587''',
                    basis='sto-3g', verbose=0)
        rks = dft.RKS(mol)
        rks.xc = 'lda,vwn'
        rks.run(conv_tol=1e-12)
        uks = dft.UKS(mol)
        uks.xc = 'lda,vwn'
        uks.run(conv_tol=1e-12)
        rcas = mcscf.CASCI(rks, 4, 4, ncore=3).run()
        ucas = mcscf.UCASCI(uks, 4, (2,2), ncore=(3,3)).run()
        g_r = rcas.nuc_grad_method().kernel()
        g_u = ucas.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g_r - g_u).max(), 0, 9)


    # Closed shell UCASCI matches restricted CASCI gradient.
    def test_ucasci_matches_restricted(self):
        mol = gto.M(atom='''
                    C  0.00  0.00  0.00
                    H  0.10  0.02  1.09
                    H  1.02  0.08 -0.35
                    H -0.42  0.96 -0.28
                    H -0.62 -0.78 -0.42''',
                    basis='3-21g', verbose=0)
        rhf = scf.RHF(mol).run(conv_tol=1e-12)
        uhf = scf.UHF(mol).run(conv_tol=1e-12)
        rcas = mcscf.CASCI(rhf, 8, 6, ncore=2).run()
        ucas = mcscf.UCASCI(uhf, 8, (3,3), ncore=(2,2)).run()
        self.assertAlmostEqual(rcas.e_tot, ucas.e_tot, 8)
        g_r = rcas.nuc_grad_method().kernel()
        g_u = ucas.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g_r - g_u).max(), 0, 7)

    # State averaged gradient against FD
    def test_state_average_fd(self):
        atom = '''
        H 0 0 0
        H 0 0 1
        H 0 1 0
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, 3, (2,1))
            mc.fcisolver.nroots = 2
            return mc.state_average_([.5, .5]).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6)

    # one-electron against FD
    def test_emptybeta_fd(self):
        atom = '''
        Li 0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.650000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, charge=1,
                        unit='Angstrom', verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 2, (1,0), ncore=(1,1)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6)

    # single orbital single electron (no excitations) edgecase
    def test_single_determinant(self):
        atom = '''
        Li 0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.650000
        '''

        def run(ia=None, ix=None, dx=0):
            mol = gto.M(atom=_move_atom(atom, ia, ix, dx) if ia is not None else atom,
                        basis='sto-3g', spin=1, charge=1,
                        unit='Angstrom', verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 1, (1,0), ncore=(1,1)).run()

        mc = run()
        _check_fd_grad(self, lambda: mc.nuc_grad_method().kernel(),
                       lambda ia, ix, dx: run(ia, ix, dx).e_tot,
                       mc.mol.natm, 1e-6)

    # check UCASCI gradient scanner path against FD
    def test_scanner(self):
        atom = '''
        H 0 0 0
        H 0 0 1
        H 0 1 0
        '''
        mol = gto.M(atom=atom,
                    basis='sto-3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec)
        gs = mc.nuc_grad_method().as_scanner()
        e0 = [None]
        def analytic_grad():
            e0[0], grad = gs(mol)
            return grad
        _check_fd_grad(
            self, analytic_grad,
            lambda ia, ix, dx: gs.base(_move_atom(atom, ia, ix, dx)),
            mol.natm, 1e-5)
        self.assertAlmostEqual(e0[0], -1.5056055923848675, 8)

    # Ensure verbose UCASCI gradient finalization runs without error
    def test_verbose_finalize(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        grad = mc.nuc_grad_method()
        grad.verbose = logger.NOTE
        de = grad.kernel()
        self.assertEqual(de.shape, (3,3))


if __name__ == "__main__":
    print("Tests for UCASCI gradients")
    unittest.main()
