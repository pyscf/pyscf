#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import mcscf
from pyscf import ci
from pyscf import lib
from pyscf.lib import logger


def _move_atom(atom, ia, ix, dx):
    coords = []
    for line in atom.splitlines():
        if line.strip():
            sym, x, y, z = line.split()
            coords.append([sym, float(x), float(y), float(z)])
    coords[ia][ix+1] += dx
    return coords


def _five_point_fd(run, ia, ix, h=1e-3):
    ep2 = run((ia, ix,  2*h)).e_tot
    ep1 = run((ia, ix,    h)).e_tot
    em1 = run((ia, ix,   -h)).e_tot
    em2 = run((ia, ix, -2*h)).e_tot
    return (-ep2 + 8*ep1 - 8*em1 + em2) / (12*h) * lib.param.BOHR


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
    def test_full_active_matches_restricted_casci(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        rcas = mcscf.CASCI(mf, mf.mo_coeff.shape[1], mol.nelectron).run()
        ref = rcas.nuc_grad_method().kernel()

        mf = scf.UHF(mol).run(conv_tol=1e-12)
        ucas = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec).run()
        grad = ucas.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(ref-grad).max(), 0, 9)

    def test_full_active_one_electron_finite_diff(self):
        def run(bond):
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {bond}',
                        basis='sto-3g', charge=1, spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec).run()
            return mc

        mc = run(1.0)
        grad = mc.nuc_grad_method().kernel()
        e1 = run(1.001).e_tot
        e2 = run(0.999).e_tot
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertAlmostEqual(grad[1,2], ref, 5)

    def test_inactive_orbitals_openshell_multicomponent_finite_diff(self):
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
        grad = mc.nuc_grad_method().kernel()
        for ia, ix in ((0,1), (0,2), (1,1), (1,2), (2,1), (2,2)):
            e1 = run(ia, ix, 0.001).e_tot
            e2 = run(ia, ix, -0.001).e_tot
            ref = (e1-e2) / 0.002 * lib.param.BOHR
            self.assertLess(abs(grad[ia,ix] - ref), 5e-7)

    def test_ucasci_matches_ucisd_complete_active_space(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()
        myci = ci.UCISD(mf, frozen=[[0], [2]]).run()
        self.assertAlmostEqual(mc.e_tot, myci.e_tot, 12)
        g_ucas = mc.nuc_grad_method().kernel()
        g_ucis = myci.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g_ucas - g_ucis).max(), 0, 10)

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

    def test_general_rdm_intermediates_contract_symmetric_integrals(self):
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

    def test_full_active_openshell_finite_diff(self):
        def run(bond):
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {bond}; H 0 1 0',
                        basis='sto-3g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec).run()
            return mc

        mc = run(1.0)
        grad = mc.nuc_grad_method().kernel()
        e1 = run(1.001).e_tot
        e2 = run(0.999).e_tot
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertAlmostEqual(grad[1,2], ref, 5)

    def test_general_casci_carbon_multicomponent_finite_diff(self):
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
        grad = mc.nuc_grad_method().kernel()
        step = 2.5e-4
        for ia, ix in ((0,0), (1,2), (2,0), (3,1)):
            e1 = run(ia, ix, step).e_tot
            e2 = run(ia, ix, -step).e_tot
            ref = (e1-e2) / (2*step) * lib.param.BOHR
            self.assertLess(abs(grad[ia,ix] - ref), 1e-6)

    def test_general_casci_nitrogen_hydride_finite_diff(self):
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
        grad = mc.nuc_grad_method().kernel()
        step = 5e-4
        e1 = run(1, 2, step).e_tot
        e2 = run(1, 2, -step).e_tot
        ref = (e1-e2) / (2*step) * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 1e-6)

    def test_general_casci_high_spin_carbon_finite_diff(self):
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
        grad = mc.nuc_grad_method().kernel()
        step = 5e-4
        e1 = run(1, 2, step).e_tot
        e2 = run(1, 2, -step).e_tot
        ref = (e1-e2) / (2*step) * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 1e-6)

    def test_general_casci_ccpvtz_five_point_finite_diff(self):
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
        grad = mc.nuc_grad_method().kernel()
        step = 1e-3
        ep2 = run(1, 2, 2*step).e_tot
        ep1 = run(1, 2, step).e_tot
        em1 = run(1, 2, -step).e_tot
        em2 = run(1, 2, -2*step).e_tot
        ref = (-ep2 + 8*ep1 - 8*em1 + em2) / (12*step) * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 5e-7)

    def test_general_casci_excited_root_finite_diff(self):
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
        grad = mc.nuc_grad_method().kernel(state=1)
        step = 5e-4
        e1 = run(1, 2, step).e_tot[1]
        e2 = run(1, 2, -step).e_tot[1]
        ref = (e1-e2) / (2*step) * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 2e-6)

    def test_general_casci_state_average_finite_diff(self):
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
        grad = mc.nuc_grad_method().kernel()
        step = 5e-4
        e1 = run(1, 2, step).e_tot
        e2 = run(1, 2, -step).e_tot
        ref = (e1-e2) / (2*step) * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 2e-6)

    def test_state_specific_excited_root_finite_diff(self):
        def run(dz=0):
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {1+dz}; H 0 1 0',
                        basis='sto-3g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, 3, (2,1))
            mc.fcisolver.nroots = 3
            return mc.run()

        mc = run()
        grad = mc.nuc_grad_method().kernel(state=2)
        e1 = run(0.001).e_tot[2]
        e2 = run(-0.001).e_tot[2]
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 1e-6)

    def test_symmetry_x2c_df_finite_diff(self):
        def run(dz=0, symmetry=False, x2c=False, density_fit=False):
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {1+dz}; H 0 1 0',
                        basis='sto-3g', spin=1, symmetry=symmetry, verbose=0)
            mf = scf.UHF(mol)
            if x2c:
                mf = mf.x2c()
            if density_fit:
                mf = mf.density_fit()
            mf.run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()

        for kwargs, tol in (((dict(symmetry=True)), 1e-6),
                            ((dict(x2c=True)), 1e-6),
                            ((dict(density_fit=True)), 3e-6)):
            mc = run(**kwargs)
            grad = mc.nuc_grad_method().kernel()
            e1 = run(0.001, **kwargs).e_tot
            e2 = run(-0.001, **kwargs).e_tot
            ref = (e1-e2) / 0.002 * lib.param.BOHR
            self.assertLess(abs(grad[1,2] - ref), tol)

    def test_uks_casci_h3_all_component_finite_diff(self):
        def run(disp=None):
            coords = numpy.asarray(((0., 0., 0.),
                                    (0., 0., 1.),
                                    (0., 1., 0.)))
            if disp is not None:
                ia, ix, dx = disp
                coords[ia, ix] += dx
            mol = gto.M(atom=[('H', coords[i]) for i in range(3)],
                        unit='Angstrom', basis='sto-3g', spin=1, verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0)).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        h = 1e-3
        fd = numpy.zeros_like(grad)
        for ia in range(3):
            for ix in range(3):
                ep2 = run((ia, ix,  2*h)).e_tot
                ep1 = run((ia, ix,    h)).e_tot
                em1 = run((ia, ix,   -h)).e_tot
                em2 = run((ia, ix, -2*h)).e_tot
                fd[ia,ix] = (-ep2 + 8*ep1 - 8*em1 + em2) / (12*h)
                fd[ia,ix] *= lib.param.BOHR
        self.assertLess(abs(grad - fd).max(), 2e-7)

    def test_uks_casci_ch3_all_component_finite_diff(self):
        atom = '''
        C  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.090000
        H  1.026719  0.000000 -0.363333
        H -0.513360  0.889165 -0.363333
        '''

        def run(disp=None):
            mol0 = gto.M(atom=atom, basis='3-21g', spin=1,
                         unit='Angstrom', verbose=0)
            if disp is not None:
                ia, ix, dx = disp
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
        grad = mc.nuc_grad_method().kernel()
        h = 1e-3
        fd = numpy.zeros_like(grad)
        for ia in range(mc.mol.natm):
            for ix in range(3):
                ep2 = run((ia, ix,  2*h)).e_tot
                ep1 = run((ia, ix,    h)).e_tot
                em1 = run((ia, ix,   -h)).e_tot
                em2 = run((ia, ix, -2*h)).e_tot
                fd[ia,ix] = (-ep2 + 8*ep1 - 8*em1 + em2) / (12*h)
                fd[ia,ix] *= lib.param.BOHR
        self.assertLess(abs(grad - fd).max(), 3e-5)

    def test_uks_casci_nh2_all_component_finite_diff(self):
        atom = '''
        N  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.030000
        H  0.968000  0.000000 -0.344000
        '''

        def run(disp=None):
            mol = gto.M(atom=atom if disp is None else
                        _move_atom(atom, disp[0], disp[1], disp[2]),
                        basis='3-21g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.grids.level = 1
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 6, (3,2), ncore=(2,2)).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        fd = numpy.zeros_like(grad)
        for ia in range(mc.mol.natm):
            for ix in range(3):
                fd[ia,ix] = _five_point_fd(run, ia, ix)
        self.assertLess(abs(grad - fd).max(), 3e-6)

    def test_uks_casci_ch2_triplet_gga_finite_diff(self):
        atom = '''
        C  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.080000
        H  1.020000  0.000000 -0.360000
        '''

        def run(disp=None):
            mol = gto.M(atom=atom if disp is None else
                        _move_atom(atom, disp[0], disp[1], disp[2]),
                        basis='3-21g', spin=2, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'pbe,pbe'
            mf.grids.level = 1
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 7, (4,2), ncore=(1,1)).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        for ia, ix in ((0,2), (1,0), (2,2)):
            self.assertLess(abs(grad[ia,ix] - _five_point_fd(run, ia, ix)),
                            1e-5)

    def test_uks_casci_zero_active_sector_finite_diff(self):
        atom = '''
        Li 0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.650000
        '''

        def run(disp=None):
            mol = gto.M(atom=atom if disp is None else
                        _move_atom(atom, disp[0], disp[1], disp[2]),
                        basis='sto-3g', spin=1, charge=1,
                        unit='Angstrom', verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.grids.level = 4
            mf.run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 1, (1,0), ncore=(1,1)).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        self.assertLess(abs(grad[1,2] - _five_point_fd(run, 1, 2)), 1e-7)

    def test_uks_casci_no_core_full_active_finite_diff(self):
        atom = '''
        H  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.000000
        H  0.000000  1.050000  0.100000
        '''

        def run(disp=None):
            mol = gto.M(atom=atom if disp is None else
                        _move_atom(atom, disp[0], disp[1], disp[2]),
                        basis='sto-3g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-11)
            return mcscf.UCASCI(mf, 3, (2,1), ncore=(0,0)).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        fd = numpy.zeros_like(grad)
        for ia in range(mc.mol.natm):
            for ix in range(3):
                fd[ia,ix] = _five_point_fd(run, ia, ix)
        self.assertLess(abs(grad - fd).max(), 1e-8)

    def test_uks_casci_global_hybrid_finite_diff(self):
        def run(disp=None):
            coords = numpy.asarray(((0., 0., 0.),
                                    (0., 0., 1.),
                                    (0., 1., 0.)))
            if disp is not None:
                ia, ix, dx = disp
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
        grad = mc.nuc_grad_method().kernel()
        fd = numpy.zeros_like(grad)
        for ia in range(mc.mol.natm):
            for ix in range(3):
                fd[ia,ix] = _five_point_fd(run, ia, ix)
        self.assertLess(abs(grad - fd).max(), 3e-6)

    def test_uks_casci_excited_root_finite_diff(self):
        atom = '''
        H  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.000000
        H  0.000000  1.000000  0.150000
        '''

        def run(disp=None):
            mol = gto.M(atom=atom if disp is None else
                        _move_atom(atom, disp[0], disp[1], disp[2]),
                        basis='sto-3g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-11)
            mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0))
            mc.fcisolver.nroots = 3
            return mc.run()

        mc = run()
        grad = mc.nuc_grad_method().kernel(state=1)
        h = 1e-3
        for ia, ix in ((0,2), (1,0), (2,2)):
            ep2 = run((ia, ix,  2*h)).e_tot[1]
            ep1 = run((ia, ix,    h)).e_tot[1]
            em1 = run((ia, ix,   -h)).e_tot[1]
            em2 = run((ia, ix, -2*h)).e_tot[1]
            fd = (-ep2 + 8*ep1 - 8*em1 + em2) / (12*h)
            fd *= lib.param.BOHR
            self.assertLess(abs(grad[ia,ix] - fd), 5e-7)

    def test_uks_casci_state_average_finite_diff(self):
        atom = '''
        H  0.000000  0.000000  0.000000
        H  0.000000  0.000000  1.000000
        H  0.000000  1.000000  0.150000
        '''

        def run(disp=None):
            mol = gto.M(atom=atom if disp is None else
                        _move_atom(atom, disp[0], disp[1], disp[2]),
                        basis='sto-3g', spin=1, unit='Angstrom',
                        verbose=0)
            mf = dft.UKS(mol)
            mf.xc = 'lda,vwn'
            mf.run(conv_tol=1e-11)
            mc = mcscf.UCASCI(mf, 2, (1,1), ncore=(1,0))
            mc.fcisolver.nroots = 2
            return mc.state_average_([.3, .7]).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        for ia, ix in ((0,2), (1,0), (2,2)):
            self.assertLess(abs(grad[ia,ix] - _five_point_fd(run, ia, ix)),
                            5e-7)

    def test_uks_restricted_collapse_matches_rks_casci(self):
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

    def test_restricted_casci_carbon_finite_diff_benchmark(self):
        atom = '''
        C  0.00  0.00  0.00
        H  0.10  0.02  1.09
        H  1.02  0.08 -0.35
        H -0.42  0.96 -0.28
        H -0.62 -0.78 -0.42
        '''

        def run(dx=0):
            coords = []
            for line in atom.splitlines():
                if line.strip():
                    sym, x, y, z = line.split()
                    coords.append([sym, float(x), float(y), float(z)])
            coords[4][1] += dx
            mol = gto.M(atom=coords, basis='6-31g', verbose=0)
            mf = scf.RHF(mol).run(conv_tol=1e-12)
            return mcscf.CASCI(mf, 8, 6, ncore=2).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        e1 = run(0.001).e_tot
        e2 = run(-0.001).e_tot
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertLess(abs(grad[4,0] - ref), 1e-6)

    def test_restricted_collapse_h4_inactive(self):
        mol = gto.M(atom='''
                    H 0.0 0.0 0.0
                    H 0.0 0.0 1.0
                    H 0.0 1.0 0.0
                    H 0.2 1.0 1.0''',
                    basis='sto-3g', verbose=0)
        rhf = scf.RHF(mol).run(conv_tol=1e-12)
        uhf = scf.UHF(mol).run(conv_tol=1e-12)
        rcas = mcscf.CASCI(rhf, 2, 2, ncore=1).run()
        ucas = mcscf.UCASCI(uhf, 2, (1,1), ncore=(1,1)).run()
        self.assertAlmostEqual(rcas.e_tot, ucas.e_tot, 10)
        g_r = rcas.nuc_grad_method().kernel()
        g_u = ucas.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g_r - g_u).max(), 0, 9)

    def test_restricted_collapse_ch4_larger_active_space(self):
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

    def test_state_average_finite_diff(self):
        def run(dz=0):
            mol = gto.M(atom=f'H 0 0 0; H 0 0 {1+dz}; H 0 1 0',
                        basis='sto-3g', spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            mc = mcscf.UCASCI(mf, 3, (2,1))
            mc.fcisolver.nroots = 2
            return mc.state_average_([.5, .5]).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        e1 = run(0.001).e_tot
        e2 = run(-0.001).e_tot
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 1e-6)

    def test_zero_active_occupied_spin_sector(self):
        def run(dz=0):
            mol = gto.M(atom=f'Li 0 0 0; H 0 0 {1.6+dz}',
                        basis='sto-3g', charge=1, spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 2, (1,0), ncore=(1,1)).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        e1 = run(0.001).e_tot
        e2 = run(-0.001).e_tot
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 1e-6)

    def test_zero_active_virtual_spin_sector(self):
        def run(dz=0):
            mol = gto.M(atom=f'Li 0 0 0; H 0 0 {1.6+dz}',
                        basis='sto-3g', charge=1, spin=1, verbose=0)
            mf = scf.UHF(mol).run(conv_tol=1e-12)
            return mcscf.UCASCI(mf, 1, (1,0), ncore=(1,1)).run()

        mc = run()
        grad = mc.nuc_grad_method().kernel()
        e1 = run(0.001).e_tot
        e2 = run(-0.001).e_tot
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertLess(abs(grad[1,2] - ref), 1e-6)

    def test_scanner(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 1 0',
                    basis='sto-3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        mc = mcscf.UCASCI(mf, mf.mo_coeff[0].shape[1], mol.nelec)
        gs = mc.nuc_grad_method().as_scanner()
        e, grad = gs(mol)
        e1 = gs.base('H 0 0 0; H 0 0 1.001; H 0 1 0')
        e2 = gs.base('H 0 0 0; H 0 0 0.999; H 0 1 0')
        ref = (e1-e2) / 0.002 * lib.param.BOHR
        self.assertAlmostEqual(e, -1.5056055923848675, 8)
        self.assertAlmostEqual(grad[1,2], ref, 5)

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
