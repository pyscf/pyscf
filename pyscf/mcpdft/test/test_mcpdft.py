#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Test API:
#   0. Initialize from mol, mf, and mc (done)
#   1. kernel (done)
#   2. optimize_mcscf_ (done)
#   3. compute_pdft_ (done)
#   4. energy_tot (done)
#   5. get_energy_decomposition (done)
#   6. checkpoint stuff
#   7. get_pdft_veff (maybe this elsewhere?)
# In the context of:
#   1. CASSCF, CASCI
#   2. Symmetry, with and without
#   3. State average, state average mix w/ different spin states

# Some assertAlmostTrue thresholds are loose because we are only
# trying to test the API here; we need tight convergence and grids
# to reproduce well when OMP is on.
import tempfile, h5py
import numpy as np
from pyscf import gto, scf, mcscf, lib, fci, dft
from pyscf import mcpdft
import unittest


mol_nosym = mol_sym = mf_nosym = mf_sym = mc_nosym = mc_sym = mcp = mc_chk = None


def auto_setup(xyz="Li 0 0 0\nH 1.5 0 0", fnal="tPBE"):
    mol_nosym = gto.M(atom=xyz, basis="sto3g", verbose=0, output="/dev/null")
    mol_sym = gto.M(
        atom=xyz, basis="sto3g", symmetry=True, verbose=0, output="/dev/null"
    )
    mf_nosym = scf.RHF(mol_nosym).run()
    mc_nosym = mcscf.CASSCF(mf_nosym, 5, 2).run(conv_tol=1e-8)
    mf_sym = scf.RHF(mol_sym).run()
    mc_sym = mcscf.CASSCF(mf_sym, 5, 2).run(conv_tol=1e-8)
    mcp_ss_nosym = mcpdft.CASSCF(mc_nosym, fnal, 5, 2).run(conv_tol=1e-8)
    mcp_ss_sym = (
        mcpdft.CASSCF(mc_sym, fnal, 5, 2)
        .set(chkfile=tempfile.NamedTemporaryFile().name)#, chk_ci=True)
        .run(conv_tol=1e-8)
    )
    mcp_sa_0 = mcp_ss_nosym.state_average(
        [
            1.0 / 5,
        ]
        * 5
    ).run(conv_tol=1e-8)
    solver_S = fci.solver(mol_nosym, singlet=True).set(spin=0, nroots=2)
    solver_T = fci.solver(mol_nosym, singlet=False).set(spin=2, nroots=3)
    mcp_sa_1 = (
        mcp_ss_nosym.state_average_mix(
            [solver_S, solver_T],
            [
                1.0 / 5,
            ]
            * 5,
        )
        .set(ci=None)
        .run(conv_tol=1e-8)
    )
    solver_A1 = fci.solver(mol_sym).set(wfnsym="A1", nroots=3)
    solver_E1x = fci.solver(mol_sym).set(wfnsym="E1x", nroots=1, spin=2)
    solver_E1y = fci.solver(mol_sym).set(wfnsym="E1y", nroots=1, spin=2)
    mcp_sa_2 = (
        mcp_ss_sym.state_average_mix(
            [solver_A1, solver_E1x, solver_E1y],
            [
                1.0 / 5,
            ]
            * 5,
        )
        .set(ci=None, chkfile=tempfile.NamedTemporaryFile().name)#, chk_ci=True)
        .run(conv_tol=1e-8)
    )
    mcp = [[mcp_ss_nosym, mcp_ss_sym], [mcp_sa_0, mcp_sa_1, mcp_sa_2]]
    nosym = [mol_nosym, mf_nosym, mc_nosym]
    sym = [mol_sym, mf_sym, mc_sym]
    mc_chk = [mcp_ss_sym, mcp_sa_2]
    return nosym, sym, mcp, mc_chk


def setUpModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp, mc_chk, original_grids
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
    nosym, sym, mcp, mc_chk = auto_setup()
    mol_nosym, mf_nosym, mc_nosym = nosym
    mol_sym, mf_sym, mc_sym = sym


def tearDownModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp, mc_chk, original_grids
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    mol_nosym.stdout.close()
    mol_sym.stdout.close()
    del mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp, mc_chk, original_grids


class KnownValues(unittest.TestCase):

    def test_init(self):
        ref_e = -7.924089707
        for symm in False, True:
            mol = (mol_nosym, mol_sym)[int(symm)]
            mf = (mf_nosym, mf_sym)[int(symm)]
            mc0 = (mc_nosym, mc_sym)[int(symm)]
            for i, cls in enumerate((mcpdft.CASCI, mcpdft.CASSCF)):
                scf = bool(i)
                for my_init in (mol, mf, mc0):
                    init_name = my_init.__class__.__name__
                    if init_name == "Mole" and symm:
                        continue
                    # ^ The underlying PySCF modules can't do this as of 02/06/2022
                    my_kwargs = {}
                    if isinstance(my_init, gto.Mole) or (not scf):
                        my_kwargs["mo_coeff"] = mc0.mo_coeff
                    with self.subTest(symm=symm, scf=scf, init=init_name):
                        mc = cls(my_init, "tPBE", 5, 2).run(**my_kwargs)
                        self.assertAlmostEqual(mc.e_tot, ref_e, delta=1e-6)
                        self.assertTrue(mc.converged)

    def test_df(self):
        ref_e = -7.924259
        ref_e0_sa = -7.923959
        for mf, symm in zip((mf_nosym, mf_sym), (False, True)):
            mf_df = mf.density_fit()
            mo = (mc_nosym, mc_sym)[int(symm)].mo_coeff
            for i, cls in enumerate((mcpdft.CASCI, mcpdft.CASSCF)):
                scf = bool(i)
                mc = cls(mf_df, "tPBE", 5, 2).run(mo_coeff=mo)
                with self.subTest(symm=symm, scf=scf, nroots=1):
                    self.assertAlmostEqual(mc.e_tot, ref_e, delta=1e-6)
                    self.assertTrue(mc.converged)
                nroots = 3
                if scf:
                    mc = mc.state_average(
                        [
                            1.0 / nroots,
                        ]
                        * nroots
                    ).run()
                    e_states = mc.e_states
                    ref = ref_e0_sa
                else:
                    mc.fcisolver.nroots = nroots
                    mc.kernel()
                    e_states = mc.e_tot
                    ref = ref_e
                with self.subTest(symm=symm, scf=scf, nroots=nroots):
                    self.assertAlmostEqual(e_states[0], ref, delta=1e-6)
                    self.assertTrue(mc.converged)

    def test_state_average(self):
        # grids_level = 6
        # ref = np.array ([-7.9238958646710085,-7.7887395616498125,-7.7761692676370355,
        #                 -7.754856419853813,-7.754856419853812,])
        # grids_level = 5
        # ref = np.array ([-7.923895345983219,-7.788739501036741,-7.776168040902887,
        #                 -7.75485647715595,-7.7548564771559505])
        # grids_level = 4
        # ref = np.array ([-7.923894841822498,-7.788739444709943,-7.776169108993544,
        #                 -7.754856321482755,-7.754856321482756])
        # grids_level = 3
        ref = np.array(
            [
                -7.923894179700609,
                -7.7887396628199,
                -7.776172495309403,
                -7.754856085624646,
                -7.754856085624647,
            ]
        )
        # TODO: figure out why SA always needs more precision than SS to get
        # the same repeatability. Fix if possible? In the mean time, loose
        # deltas below for the sake of speed.
        for ix, mc in enumerate(mcp[1]):
            with self.subTest(symmetry=bool(ix // 2), triplet_ms=(0, 1, "mixed")[ix]):
                self.assertTrue(mc.converged)
                self.assertAlmostEqual(
                    lib.fp(np.sort(mc.e_states)), lib.fp(ref), delta=1e-5
                )
                self.assertAlmostEqual(mc.e_tot, np.average(ref), delta=1e-5)

    def test_casci_multistate(self):
        # grids_level = 3
        ref = np.array(
            [
                -7.923894179700609,
                -7.7887396628199,
                -7.776172495309403,
                -7.754856085624646,
                -7.754856085624647,
            ]
        )
        mc = mcpdft.CASCI(mcp[1][0], "tPBE", 5, 2)
        mc.fcisolver.nroots = 5
        mc.kernel()
        with self.subTest(symmetry=False):
            self.assertTrue(mc.converged)
            self.assertAlmostEqual(lib.fp(np.sort(mc.e_tot)), lib.fp(ref), delta=1e-5)
            self.assertAlmostEqual(np.average(mc.e_tot), np.average(ref), delta=1e-5)
        e_tot = []
        mc = mcpdft.CASCI(mcp[1][2], "tPBE", 5, 2)
        ci_ref = mcp[1][2].ci
        mc.ci, mc.fcisolver.nroots, mc.fcisolver.wfnsym = ci_ref[:3], 3, "A1"
        mc.kernel()
        self.assertTrue(mc.converged)
        e_tot.extend(mc.e_tot)
        mc.ci, mc.fcisolver.nroots, mc.fcisolver.wfnsym = ci_ref[3], 1, "E1x"
        mc.fcisolver.spin = 2
        mc.kernel()
        self.assertTrue(mc.converged)
        e_tot.append(mc.e_tot)
        mc.ci, mc.fcisolver.nroots, mc.fcisolver.wfnsym = ci_ref[4], 1, "E1y"
        mc.kernel()
        self.assertTrue(mc.converged)
        e_tot.append(mc.e_tot)
        with self.subTest(symmetry=True):
            self.assertTrue(mc.converged)
            self.assertAlmostEqual(lib.fp(np.sort(e_tot)), lib.fp(ref), delta=1e-5)
            self.assertAlmostEqual(np.average(e_tot), np.average(ref), delta=1e-5)

    def test_decomposition_ss(self):  # TODO
        ref = [
            1.0583544218,
            -12.5375911135,
            5.8093938665,
            -2.1716353580,
            -0.0826115115,
            -2.2123063329,
        ]
        terms = ["nuc", "core", "Coulomb", "OT(X)", "OT(C)", "WFN(XC)"]
        for ix, mc in enumerate(mcp[0]):
            casci = mcpdft.CASCI(mc, "tPBE", 5, 2).run()
            for obj, objtype in zip((mc, casci), ("CASSCF", "CASCI")):
                test = obj.get_energy_decomposition()
                for t, r, term in zip(test, ref, terms):
                    with self.subTest(objtype=objtype, symmetry=bool(ix), term=term):
                        self.assertAlmostEqual(t, r, delta=1e-5)
                with self.subTest(objtype=objtype, symmetry=bool(ix), term="sanity"):
                    self.assertAlmostEqual(np.sum(test[:-1]), obj.e_tot, 9)

    def test_decomposition_hybrid(self):
        ref = [
            1.0583544218,
            -12.5375911135,
            5.8093938665,
            -1.6287258807,
            -0.0619586538,
            -0.5530763650,
        ]
        terms = ["nuc", "core", "Coulomb", "OT(X)", "OT(C)", "WFN(XC)"]
        for ix, mc in enumerate(mcp[0]):
            mc_scf = mcpdft.CASSCF(mc, "tPBE0", 5, 2).run()
            mc_ci = mcpdft.CASCI(mc, "tPBE0", 5, 2).run()
            for obj, objtype in zip((mc_scf, mc_ci), ("CASSCF", "CASCI")):
                test = obj.get_energy_decomposition()
                for t, r, term in zip(test, ref, terms):
                    with self.subTest(objtype=objtype, symmetry=bool(ix), term=term):
                        self.assertAlmostEqual(t, r, delta=1e-5)
                with self.subTest(objtype=objtype, symmetry=bool(ix), term="sanity"):
                    self.assertAlmostEqual(np.sum(test), obj.e_tot, 9)


    def test_decomposition_sa(self):
        ref_nuc = 1.0583544218
        ref_states = np.array(
            [
                [
                    -12.5385413915,
                    5.8109724796,
                    -2.1720331222,
                    -0.0826465641,
                    -2.2127964255,
                ],
                [
                    -12.1706553996,
                    5.5463231972,
                    -2.1601539256,
                    -0.0626079593,
                    -2.1943087132,
                ],
                [
                    -12.1768195314,
                    5.5632261670,
                    -2.1552571900,
                    -0.0656763663,
                    -2.1887042769,
                ],
                [
                    -12.1874226655,
                    5.5856701424,
                    -2.1481995107,
                    -0.0632609608,
                    -2.1690856659,
                ],
                [
                    -12.1874226655,
                    5.5856701424,
                    -2.1481995107,
                    -0.0632609608,
                    -2.1690856659,
                ],
            ]
        )
        terms = ["core", "Coulomb", "OT(X)", "OT(C)", "WFN(XC)"]
        for ix, (mc, ms) in enumerate(zip(mcp[1], [0, 1, "mixed"])):
            s = bool(ix // 2)
            objs = [
                mc,
            ]
            objtypes = [
                "CASSCF",
            ]
            if ix != 1:  # There is no CASCI equivalent to mcp[1][1]
                casci = mcpdft.CASCI(mc, "tPBE", 5, 2)
                casci.fcisolver.nroots = (
                    5 - ix
                )  # Just check the A1 roots when symmetry is enabled
                casci.ci = mc.ci[: 5 - ix]
                casci.kernel()
                objs.append(casci)
                objtypes.append("CASCI")
            for obj, objtype in zip(objs, objtypes):
                test = obj.get_energy_decomposition()
                test_nuc, test_states = test[0], np.array(test[1:]).T
                # Arrange states in ascending energy order
                e_states = getattr(obj, "e_states", obj.e_tot)
                idx = np.argsort(e_states)
                test_states = test_states[idx, :]
                e_ref = np.array(e_states)[idx]
                with self.subTest(
                    objtype=objtype, symmetry=s, triplet_ms=ms, term="nuc"
                ):
                    self.assertAlmostEqual(test_nuc, ref_nuc, 9)
                for state, (test, ref) in enumerate(zip(test_states, ref_states)):
                    for t, r, term in zip(test, ref, terms):
                        with self.subTest(
                            objtype=objtype,
                            symmetry=s,
                            triplet_ms=ms,
                            term=term,
                            state=state,
                        ):
                            self.assertAlmostEqual(t, r, delta=1e-5)
                    with self.subTest(
                        objtype=objtype,
                        symmetry=s,
                        triplet_ms=ms,
                        term="sanity",
                        state=state,
                    ):
                        self.assertAlmostEqual(
                            np.sum(test[:-1]) + test_nuc, e_ref[state], 9
                        )

    def test_decomposition_hybrid_sa(self):
        ref_nuc = 1.0583544218
        ref_states = np.array(
            [
                [
                    -12.5385413915,
                    5.8109724796,
                    -1.6290249990,
                    -0.0619850920,
                    -0.5531991067,
                ],
                [
                    -12.1706553996,
                    5.5463231972,
                    -1.6201152933,
                    -0.0469559736,
                    -0.5485771470,
                ],
                [
                    -12.1768195314,
                    5.5632261670,
                    -1.6164436229,
                    -0.0492571730,
                    -0.5471763843,
                ],
                [
                    -12.1874226655,
                    5.5856701424,
                    -1.6111471613,
                    -0.0474456546,
                    -0.5422714244,
                ],
                [
                    -12.1874226655,
                    5.5856701424,
                    -1.6111480360,
                    -0.0474456745,
                    -0.5422714244,
                ],
            ]
        )
        terms = ["core", "Coulomb", "OT(X)", "OT(C)", "WFN(XC)"]
        for ix, (mc, ms) in enumerate(zip(mcp[1], [0, 1, "mixed"])):
            s = bool(ix // 2)
            mc_scf = mcpdft.CASSCF(mc, "tPBE0", 5, 2)
            if ix == 0:
                mc_scf = mc_scf.state_average(mc.weights)
            else:
                mc_scf = mc_scf.state_average_mix(mc.fcisolver.fcisolvers, mc.weights)
            mc_scf.run(ci=mc.ci, mo_coeff=mc.mo_coeff)
            objs = [
                mc_scf,
            ]
            objtypes = [
                "CASSCF",
            ]
            if ix != 1:  # There is no CASCI equivalent to mcp[1][1]
                mc_ci = mcpdft.CASCI(mc, "tPBE0", 5, 2)
                mc_ci.fcisolver.nroots = (
                    5 - ix
                )  # Just check the A1 roots when symmetry is enabled
                mc_ci.ci = mc.ci[: 5 - ix]
                mc_ci.kernel()
                objs.append(mc_ci)
                objtypes.append("CASCI")
            for obj, objtype in zip(objs, objtypes):
                test = obj.get_energy_decomposition()
                test_nuc, test_states = test[0], np.array(test[1:]).T
                # Arrange states in ascending energy order
                e_states = getattr(obj, "e_states", obj.e_tot)
                idx = np.argsort(e_states)
                test_states = test_states[idx, :]
                e_ref = np.array(e_states)[idx]
                with self.subTest(
                    objtype=objtype, symmetry=s, triplet_ms=ms, term="nuc"
                ):
                    self.assertAlmostEqual(test_nuc, ref_nuc, 9)
                for state, (test, ref) in enumerate(zip(test_states, ref_states)):
                    for t, r, term in zip(test, ref, terms):
                        with self.subTest(
                            objtype=objtype,
                            symmetry=s,
                            triplet_ms=ms,
                            term=term,
                            state=state,
                        ):
                            self.assertAlmostEqual(t, r, delta=1e-5)
                    with self.subTest(
                        objtype=objtype,
                        symmetry=s,
                        triplet_ms=ms,
                        term="sanity",
                        state=state,
                    ):
                        self.assertAlmostEqual(
                            np.sum(test) + test_nuc, e_ref[state], 9
                        )

    def test_energy_tot(self):
        # Test both correctness and energy_tot function purity
        def get_attr(mc):
            mo_ref = lib.fp(mc.mo_coeff)
            ci_ref = lib.fp(np.concatenate(mc.ci, axis=None))
            e_states_ref = lib.fp(getattr(mc, "e_states", 0))
            return mo_ref, ci_ref, mc.e_tot, e_states_ref, mc.grids.level, mc.otxc

        def test_energy_tot_crunch(test_list, ref_list, casestr):
            for ix, (t, r) in enumerate(zip(test_list, ref_list)):
                with self.subTest(case=casestr, item=ix):
                    if isinstance(t, (float, np.floating)):
                        self.assertAlmostEqual(t, r, delta=1e-5)
                    else:
                        self.assertEqual(t, r)

        def test_energy_tot_loop_ss(e_ref_ss, diff, **kwargs):
            for ix, mc in enumerate(mcp[0]):
                ref_list = [e_ref_ss] + list(get_attr(mc))
                e_test = mc.energy_tot(**kwargs)[0]
                test_list = [e_test] + list(get_attr(mc))
                casestr = "diff={}; SS; symmetry={}".format(diff, bool(ix))
                test_energy_tot_crunch(test_list, ref_list, casestr)

        def test_energy_tot_loop_sa(e_ref_sa, diff, **kwargs):
            for ix, mc in enumerate(mcp[1]):
                ref_list = [e_ref_sa] + list(get_attr(mc))
                e_s0_test = mc.energy_tot(**kwargs)[0]
                test_list = [e_s0_test] + list(get_attr(mc))
                sym = bool(ix // 2)
                tms = (0, 1, "mixed")[ix]
                casestr = "diff={}; SA; symmetry={}; triplet_ms={}".format(
                    diff, sym, tms
                )
                test_energy_tot_crunch(test_list, ref_list, casestr)

        def test_energy_tot_loop(e_ref_ss, e_ref_sa, diff, **kwargs):
            test_energy_tot_loop_ss(e_ref_ss, diff, **kwargs)
            test_energy_tot_loop_sa(e_ref_sa, diff, **kwargs)

        # tBLYP
        e_ref_ss = mcpdft.CASSCF(mcp[0][0], "tBLYP", 5, 2).kernel()[0]
        mc_ref = (
            mcpdft.CASSCF(mcp[1][0], "tBLYP", 5, 2)
            .state_average(
                [
                    1.0 / 5,
                ]
                * 5
            )
            .run()
        )
        e_ref_sa = mc_ref.e_states[0]
        test_energy_tot_loop(e_ref_ss, e_ref_sa, "fnal", otxc="tBLYP")
        # grids_level = 2
        e_ref_ss = mcpdft.CASSCF(mcp[0][0], "tPBE", 5, 2, grids_level=2).kernel()[0]
        mc_ref = (
            mcpdft.CASSCF(mcp[1][0], "tPBE", 5, 2, grids_level=2)
            .state_average(
                [
                    1.0 / 5,
                ]
                * 5
            )
            .run()
        )
        e_ref_sa = mc_ref.e_states[0]
        test_energy_tot_loop(e_ref_ss, e_ref_sa, "grids", grids_level=2)
        # CASCI wfn
        mc_ref = mcpdft.CASCI(mf_nosym, "tPBE", 5, 2).run()
        test_energy_tot_loop_ss(
            mc_ref.e_tot, "wfn", mo_coeff=mc_ref.mo_coeff, ci=mc_ref.ci
        )
        fake_ci = [c.copy() for c in mcp[1][0].ci]
        fake_ci[0] = mc_ref.ci.copy()
        test_energy_tot_loop_sa(
            mc_ref.e_tot, "wfn", mo_coeff=mc_ref.mo_coeff, ci=fake_ci
        )

    def test_kernel_steps_casscf(self):
        ref_tot = -7.919939037859329
        ref_ot = -2.2384273324895165
        mo_ref = 0.9925428665189101
        ci_ref = 0.9886507094634355
        for ix, mc in enumerate(mcp[0]):
            e_tot = mc.e_tot
            e_ot = mc.e_ot
            e_mcscf = mc.e_mcscf
            mo = (mf_nosym, mf_sym)[ix].mo_coeff.copy()
            ci = np.zeros_like(mc.ci)
            ci[0, 0] = 1.0
            mc.compute_pdft_energy_(mo_coeff=mo, ci=ci)
            with self.subTest(case="SS", part="pdft1", symmetry=bool(ix)):
                self.assertEqual(lib.fp(mc.mo_coeff), lib.fp(mo))
                self.assertEqual(lib.fp(mc.ci), lib.fp(ci))
                self.assertEqual(mc.e_mcscf, e_mcscf)
                self.assertAlmostEqual(mc.e_tot, ref_tot, 9)
                self.assertAlmostEqual(mc.e_ot, ref_ot, 7)
            mc.e_tot = 0.0
            mc.e_ot = 0.0
            mc.optimize_mcscf_()
            with self.subTest(case="SS", part="mcscf", symmetry=bool(ix)):
                self.assertEqual(mc.e_tot, 0.0)
                self.assertEqual(mc.e_ot, 0.0)
                self.assertAlmostEqual(abs(mc.mo_coeff[0, 0]), mo_ref, delta=1e-5)
                self.assertAlmostEqual(abs(mc.ci[0, 0]), ci_ref, delta=1e-5)
                self.assertAlmostEqual(mc.e_mcscf, e_mcscf, 9)
            mc.compute_pdft_energy_()
            with self.subTest(case="SS", part="pdft2", symmetry=bool(ix)):
                self.assertAlmostEqual(mc.e_tot, e_tot, delta=1e-6)
                self.assertAlmostEqual(mc.e_ot, e_ot, delta=1e-6)
        mo_ref = 0.9908324004974881
        ci_ref = 0.988599145861302
        ref_avg = -7.7986453
        for ix, mc in enumerate(mcp[1]):
            sym = bool(ix // 2)
            tms = (0, 1, "mixed")[ix]
            e_tot = mc.e_tot
            e_ot = np.array(mc.e_ot)
            e_mcscf = np.array(mc.e_mcscf)
            e_states = np.array(mc.e_states)
            nroots = len(e_mcscf)
            mo = (mf_nosym, mf_sym)[int(sym)].mo_coeff.copy()
            ci = [c.copy() for c in mc.ci]
            ci[0][:, :] = 0.0
            ci[0][0, 0] = 1.0
            fp_fake_ci = lib.fp(np.concatenate(ci, axis=None))
            mc.compute_pdft_energy_(mo_coeff=mo, ci=ci)
            with self.subTest(case="SA", part="pdft1", symmetry=sym, triplet_ms=tms):
                self.assertEqual(lib.fp(mc.mo_coeff), lib.fp(mo))
                self.assertEqual(lib.fp(np.concatenate(mc.ci, axis=None)), fp_fake_ci)
                self.assertEqual(lib.fp(mc.e_mcscf), lib.fp(e_mcscf))
                self.assertAlmostEqual(mc.e_tot, ref_avg, delta=1e-5)
                self.assertAlmostEqual(mc.e_states[0], ref_tot, 9)
                self.assertAlmostEqual(mc.e_ot[0], ref_ot, 7)
            mc.e_tot = 0.0
            mc.e_ot = [
                0.0,
            ] * len(mc.e_ot)
            mc.fcisolver.e_states = [
                0.0,
            ] * len(mc.e_states)
            mc.optimize_mcscf_()
            with self.subTest(case="SA", part="mcscf", symmetry=sym, triplet_ms=tms):
                self.assertEqual(mc.e_tot, 0.0)
                self.assertEqual(lib.fp(mc.e_ot), 0.0)
                self.assertEqual(lib.fp(mc.e_states), 0.0)
                self.assertAlmostEqual(abs(mc.mo_coeff[0, 0]), mo_ref, delta=1e-5)
                self.assertAlmostEqual(abs(mc.ci[0][0, 0]), ci_ref, delta=1e-5)
                self.assertAlmostEqual(lib.fp(mc.e_mcscf), lib.fp(e_mcscf), 7)
            mc.compute_pdft_energy_()
            with self.subTest(case="SA", part="pdft2", symmetry=sym, triplet_ms=tms):
                self.assertAlmostEqual(mc.e_tot, e_tot, delta=1e-5)
                self.assertAlmostEqual(lib.fp(mc.e_ot), lib.fp(e_ot), delta=1e-5)
                self.assertAlmostEqual(
                    lib.fp(mc.e_states), lib.fp(e_states), delta=1e-5
                )

    def test_kernel_steps_casci(self):
        ref_tot = -7.919924747436255
        ref_ot = -2.238538447576774
        ci_ref = 0.9886507094634355
        for nroots in range(1, 3):
            for init, symmetry in zip(mcp[0], (False, True)):
                mc = mcpdft.CASCI(init, "tPBE", 5, 2)
                mc.fcisolver.nroots = nroots
                if nroots > 1:
                    mc.ci = None
                mc.kernel()
                e_tot = np.atleast_1d(mc.e_tot)[0]
                e_ot = np.atleast_1d(mc.e_ot)[0]
                e_mcscf = np.atleast_1d(mc.e_mcscf)[0]
                ci = np.zeros_like(mc.ci)
                ci.flat[0] = 1.0
                if nroots > 1:
                    ci = list(ci)
                # pyscf PR #1623 made direct_spin1_symm unable to interpret
                # a 3D fci vector array
                mc.compute_pdft_energy_(ci=ci)
                with self.subTest(part="pdft1", symmetry=symmetry, nroots=nroots):
                    self.assertEqual(lib.fp(np.array(mc.ci)), lib.fp(ci), 9)
                    self.assertEqual(np.atleast_1d(mc.e_mcscf)[0], e_mcscf)
                    self.assertAlmostEqual(np.atleast_1d(mc.e_tot)[0], ref_tot, 7)
                    self.assertAlmostEqual(np.atleast_1d(mc.e_ot)[0], ref_ot, 7)
                mc.e_tot = 0.0
                mc.e_ot = 0.0
                mc.optimize_mcscf_()
                with self.subTest(part="mcscf", symmetry=symmetry, nroots=nroots):
                    self.assertEqual(np.atleast_1d(mc.e_tot)[0], 0.0)
                    self.assertEqual(np.atleast_1d(mc.e_ot)[0], 0.0)
                    self.assertAlmostEqual(
                        abs(np.asarray(mc.ci).flat[0]), ci_ref, delta=1e-5
                    )
                    self.assertAlmostEqual(np.atleast_1d(mc.e_mcscf)[0], e_mcscf, 9)
                mc.compute_pdft_energy_()
                with self.subTest(part="pdft2", symmetry=symmetry, nroots=nroots):
                    self.assertAlmostEqual(
                        np.atleast_1d(mc.e_tot)[0], e_tot, delta=1e-5
                    )
                    self.assertAlmostEqual(np.atleast_1d(mc.e_ot)[0], e_ot, delta=1e-5)

    def test_scanner(self):
        # Putting more energy into CASSCF than CASCI scanner because this is
        # necessary for geometry optimization, which isn't available for CASCI
        mcp1 = auto_setup(xyz="Li 0 0 0\nH 1.55 0 0")[-2]
        for mol0, mc0, mc1 in zip([mol_nosym, mol_sym], mcp[0], mcp1[0]):
            mc_scan = mc1.as_scanner()
            with self.subTest(case="SS CASSCF", symm=mol0.symmetry):
                self.assertAlmostEqual(mc_scan(mol0), mc0.e_tot, delta=1e-6)
            mc2 = mcpdft.CASCI(mc1, "tPBE", 5, 2).run(mo_coeff=mc1.mo_coeff)
            mc_scan = mc2.as_scanner()
            mc_scan._scf(mol0)  # TODO: fix this in CASCI as_scanner
            # when you pass mo_coeff on call, it skips updating the _scf
            # object geometry. This breaks things like CASCI.energy_nuc (),
            # CASCI.get_hcore (), etc. which refer to the corresponding
            # _scf fns but don't default to CASCI self.mol
            e_tot = mc_scan(mol0, mo_coeff=mc0.mo_coeff, ci0=mc0.ci)
            with self.subTest(case="SS CASCI", symm=mol0.symmetry):
                self.assertAlmostEqual(e_tot, mc0.e_tot, delta=1e-6)
        for ix, (mc0, mc1) in enumerate(zip(mcp[1], mcp1[1])):
            tms = (0, 1, "mixed")[ix]
            sym = bool(ix // 2)
            mol0 = [mol_nosym, mol_sym][int(sym)]
            with self.subTest(case="SA CASSCF", symm=mol0.symmetry, triplet_ms=tms):
                mc_scan = mc1.as_scanner()
                e_tot = mc_scan(mol0)
                e_states_fp = lib.fp(np.sort(mc_scan.e_states))
                e_states_fp_ref = lib.fp(np.sort(mc0.e_states))
                self.assertAlmostEqual(e_tot, mc0.e_tot, delta=2e-6)
                self.assertAlmostEqual(e_states_fp, e_states_fp_ref, delta=2e-5)
        mc2 = mcpdft.CASCI(mcp1[1][0], "tPBE", 5, 2)
        mc2.fcisolver.nroots = 5
        mc2.run(mo_coeff=mcp[1][0].mo_coeff)
        mc_scan = mc2.as_scanner()
        mc_scan._scf(mol_nosym)  # TODO: fix this in CASCI as_scanner
        # when you pass mo_coeff on call, it skips updating the _scf
        # object geometry. This breaks things like CASCI.energy_nuc (),
        # CASCI.get_hcore (), etc. which refer to the corresponding
        # _scf fns but don't default to CASCI self.mol
        e_tot = mc_scan(mol_nosym, mo_coeff=mcp[1][0].mo_coeff, ci0=mcp[1][0].ci)
        e_states_fp = lib.fp(np.sort(e_tot))
        e_states_fp_ref = lib.fp(np.sort(mcp[1][0].e_states))
        with self.subTest(case="nroots=5 CASCI"):
            self.assertAlmostEqual(e_states_fp, e_states_fp_ref, delta=5e-5)

    def test_tpbe0(self):
        # The most common hybrid functional
        for mc in mcp[0]:
            e_ref = 0.75 * mc.e_tot + 0.25 * mc.e_mcscf
            e_test = mc.energy_tot(otxc="tPBE0")[0]
            with self.subTest(case="SS", symm=mc.mol.symmetry):
                self.assertAlmostEqual(e_test, e_ref, 10)
        for ix, mc in enumerate(mcp[1]):
            tms = (0, 1, "mixed")[ix]
            sym = bool(ix // 2)
            e_states = np.array(mc.e_states)
            e_mcscf = np.array(mc.e_mcscf)
            e_ref = 0.75 * e_states + 0.25 * e_mcscf
            e_test = [
                mc.energy_tot(otxc="tPBE0", state=i)[0] for i in range(len(mc.e_states))
            ]
            with self.subTest(case="SA CASSCF", symm=sym, triplet_ms=tms):
                e_ref_fp = lib.fp(np.sort(e_ref))
                e_test_fp = lib.fp(np.sort(e_test))
                self.assertAlmostEqual(e_test_fp, e_ref_fp, 10)

    def test_chkfile(self):
        for mc in mc_chk:
            if hasattr(mc, "e_states"):
                case = "SA CASSCF"
            else:
                case = "SS"

            with self.subTest(case=case):
                self.assertTrue(h5py.is_hdf5(mc.chkfile))
                self.assertEqual(lib.fp(mc.mo_coeff), lib.fp(lib.chkfile.load(mc.chkfile, "pdft/mo_coeff")))
                self.assertEqual(mc.e_tot, lib.chkfile.load(mc.chkfile, "pdft/e_tot"))
                self.assertEqual(lib.fp(mc.e_ot), lib.fp(lib.chkfile.load(mc.chkfile, "pdft/e_ot")))
                self.assertEqual(lib.fp(mc.e_mcscf), lib.fp(lib.chkfile.load(mc.chkfile, "pdft/e_mcscf")))

                # Requires PySCF version > 2.6.2 which is not currently available on pip
                # for state, (c_ref, c) in enumerate(zip(mc.ci, lib.chkfile.load(mc.chkfile, "pdft/ci"))):
                    # with self.subTest(state=state):
                        # self.assertEqual(lib.fp(c_ref), lib.fp(c))

                if case=="SA CASSCF":
                    self.assertEqual(lib.fp(mc.e_states), lib.fp(lib.chkfile.load(mc.chkfile, "pdft/e_states")))

    def test_h2_triplet(self):
        mol = gto.M (atom='H 0 0 0; H 1 0 0', basis='sto-3g', spin=2)
        mf = scf.RHF (mol).run ()
        mc = mcpdft.CASSCF (mf, 'tPBE', 2, 2).run ()
        # Reference from OpenMolcas v24.10
        e_ref = -0.74702903
        self.assertAlmostEqual (mc.e_tot, e_ref, 6)



if __name__ == "__main__":
    print("Full Tests for MC-PDFT energy API")
    unittest.main()
