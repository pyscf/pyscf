#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

import unittest
import os
import tempfile
import numpy
from numpy.testing import assert_allclose
from pyscf import lib, gto, scf

have_pe = False
try:
    import cppe
    import pyscf.solvent as solvent
    from pyscf.solvent import pol_embed
    have_pe = True
except ImportError:
    pass


dname = os.path.dirname(__file__)

potf = tempfile.NamedTemporaryFile()
potf.write(b'''!
@COORDINATES
3
AA
O     3.53300000    2.99600000    0.88700000      1
H     4.11100000    3.13200000    1.63800000      2
H     4.10500000    2.64200000    0.20600000      3
@MULTIPOLES
ORDER 0
3
1     -0.67444000
2      0.33722000
3      0.33722000
@POLARIZABILITIES
ORDER 1 1
3
1      5.73935000     0.00000000     0.00000000     5.73935000     0.00000000     5.73935000
2      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
3      2.30839000     0.00000000     0.00000000     2.30839000     0.00000000     2.30839000
EXCLISTS
3 3
1   2  3
2   1  3
3   1  2''')
potf.flush()
potfile = potf.name

mol = gto.M(atom='''
       6        0.000000    0.000000   -0.542500
       8        0.000000    0.000000    0.677500
       1        0.000000    0.935307   -1.082500
       1        0.000000   -0.935307   -1.082500
            ''', basis='sto3g', verbose=7,
            output='/dev/null')

def tearDownModule():
    global potf, mol
    mol.stdout.close()
    del potf, mol


def _compute_multipole_potential_integrals(pe, site, order, moments):
    if order > 2:
        raise NotImplementedError("""Multipole potential integrals not
                                  implemented for order > 2.""")
    pe.mol.set_rinv_orig(site)
    # TODO: only calculate up to requested order!
    integral0 = pe.mol.intor("int1e_rinv")
    integral1 = pe.mol.intor("int1e_iprinv") + pe.mol.intor("int1e_iprinv").transpose(0, 2, 1)
    integral2 = pe.mol.intor("int1e_ipiprinv") + pe.mol.intor("int1e_ipiprinv").transpose(0, 2, 1) + 2.0 * pe.mol.intor("int1e_iprinvip")

    # k = 2: 0,1,2,4,5,8 = XX, XY, XZ, YY, YZ, ZZ
    # add the lower triangle to the upper triangle, i.e.,
    # XY += YX : 1 + 3
    # XZ += ZX : 2 + 6
    # YZ += ZY : 5 + 7
    # and divide by 2
    integral2[1] += integral2[3]
    integral2[2] += integral2[6]
    integral2[5] += integral2[7]
    integral2[1] *= 0.5
    integral2[2] *= 0.5
    integral2[5] *= 0.5

    op = integral0 * moments[0] * cppe.prefactors(0)
    if order > 0:
        op += numpy.einsum('aij,a->ij', integral1,
                           moments[1] * cppe.prefactors(1))
    if order > 1:
        op += numpy.einsum('aij,a->ij',
                           integral2[[0, 1, 2, 4, 5, 8], :, :],
                           moments[2] * cppe.prefactors(2))

    return op

def _compute_field_integrals(pe, site, moment):
    pe.mol.set_rinv_orig(site)
    integral = pe.mol.intor("int1e_iprinv") + pe.mol.intor("int1e_iprinv").transpose(0, 2, 1)
    op = numpy.einsum('aij,a->ij', integral, -1.0*moment)
    return op

def _compute_field(pe, site, D):
    pe.mol.set_rinv_orig(site)
    integral = pe.mol.intor("int1e_iprinv") + pe.mol.intor("int1e_iprinv").transpose(0, 2, 1)
    return numpy.einsum('ij,aij->a', D, integral)

def _exec_cppe(pe, dm, elec_only=False):
    V_es = numpy.zeros((pe.mol.nao, pe.mol.nao), dtype=numpy.float64)
    for p in pe.potentials:
        moments = []
        for m in p.multipoles:
            m.remove_trace()
            moments.append(m.values)
        V_es += _compute_multipole_potential_integrals(pe, p.position, m.k, moments)

    pe.cppe_state.energies["Electrostatic"]["Electronic"] = (
        numpy.einsum('ij,ij->', V_es, dm)
    )

    n_sitecoords = 3 * pe.cppe_state.get_polarizable_site_number()
    V_ind = numpy.zeros((pe.mol.nao, pe.mol.nao), dtype=numpy.float64)
    if n_sitecoords:
        # TODO: use list comprehensions
        current_polsite = 0
        elec_fields = numpy.zeros(n_sitecoords, dtype=numpy.float64)
        for p in pe.potentials:
            if not p.is_polarizable:
                continue
            elec_fields_s = _compute_field(pe, p.position, dm)
            elec_fields[3*current_polsite:3*current_polsite + 3] = elec_fields_s
            current_polsite += 1
        pe.cppe_state.update_induced_moments(elec_fields, elec_only)
        induced_moments = numpy.array(pe.cppe_state.get_induced_moments())
        current_polsite = 0
        for p in pe.potentials:
            if not p.is_polarizable:
                continue
            site = p.position
            V_ind += _compute_field_integrals(pe, site=site, moment=induced_moments[3*current_polsite:3*current_polsite + 3])
            current_polsite += 1
    e = pe.cppe_state.total_energy
    if not elec_only:
        vmat = V_es + V_ind
    else:
        vmat = V_ind
        e = pe.cppe_state.energies["Polarization"]["Electronic"]
    return e, vmat

@unittest.skipIf(not have_pe, "CPPE library not found.")
class TestPolEmbed(unittest.TestCase):
    def test_exec_cppe(self):
        pe = solvent.PE(mol, os.path.join(dname, "pna_6w.potential"))
        numpy.random.seed(2)
        nao = mol.nao
        dm = numpy.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)

        eref, vref = _exec_cppe(pe, dm[1], elec_only=False)
        e, v = pe._exec_cppe(dm, elec_only=False)
        self.assertAlmostEqual(abs(vref - v[1]).max(), 0, 10)

        eref, vref = _exec_cppe(pe, dm[0], elec_only=True)
        e, v = pe._exec_cppe(dm, elec_only=True)
        self.assertAlmostEqual(eref, e[0], 9)
        self.assertAlmostEqual(abs(vref - v[0]).max(), 0, 9)

    def test_pol_embed_scf(self):
        mol = gto.Mole()
        mol.atom = '''
        C          8.64800        1.07500       -1.71100
        C          9.48200        0.43000       -0.80800
        C          9.39600        0.75000        0.53800
        C          8.48200        1.71200        0.99500
        C          7.65300        2.34500        0.05500
        C          7.73200        2.03100       -1.29200
        H         10.18300       -0.30900       -1.16400
        H         10.04400        0.25200        1.24700
        H          6.94200        3.08900        0.38900
        H          7.09700        2.51500       -2.01800
        N          8.40100        2.02500        2.32500
        N          8.73400        0.74100       -3.12900
        O          7.98000        1.33100       -3.90100
        O          9.55600       -0.11000       -3.46600
        H          7.74900        2.71100        2.65200
        H          8.99100        1.57500        2.99500
        '''
        mol.basis = "STO-3G"
        mol.build()
        pe_options = {"potfile": os.path.join(dname, "pna_6w.potential")}
        pe = pol_embed.PolEmbed(mol, pe_options)
        mf = solvent.PE(scf.RHF(mol), pe)
        mf.conv_tol = 1e-10
        mf.kernel()
        ref_pe_energy = -0.03424830892844
        ref_scf_energy = -482.9411084900
        assert_allclose(ref_pe_energy, mf.with_solvent.e, atol=1e-6)
        assert_allclose(ref_scf_energy, mf.e_tot, atol=1e-6)

    def test_pe_scf(self):
        pe = solvent.PE(mol, potfile)
        mf = solvent.PE(mol.RHF(), pe).run(conv_tol=1e-10)
        self.assertAlmostEqual(mf.e_tot, -112.35232445743728, 9)
        self.assertAlmostEqual(mf.with_solvent.e, 0.00020182314249546455, 9)

    def test_as_scanner(self):
        mf_scanner = solvent.PE(scf.RHF(mol), potfile).as_scanner()
        mf_scanner(mol)
        self.assertAlmostEqual(mf_scanner.with_solvent.e, 0.00020182314249546455, 9)
        # Change solute. cppe may not support this
        mf_scanner('H  0. 0. 0.; H  0. 0. .9')
        self.assertAlmostEqual(mf_scanner.with_solvent.e, 5.2407234004672825e-05, 9)

    def test_newton_rohf(self):
        mf = solvent.PE(mol.ROHF(max_memory=0), potfile)
        mf = mf.newton()
        e = mf.kernel()
        self.assertAlmostEqual(e, -112.35232445745123, 9)

        mf = solvent.PE(mol.ROHF(max_memory=0), potfile)
        e = mf.kernel()
        self.assertAlmostEqual(e, -112.35232445745123, 9)

    def test_rhf_tda(self):
        # TDA with equilibrium_solvation
        mf = solvent.PE(mol.RHF(), potfile).run()
        td = solvent.PE(mf.TDA(), potfile).run(equilibrium_solvation=True)
        ref = numpy.array([0.1506426609354755, 0.338251407831332, 0.4471267328974609])
        self.assertAlmostEqual(abs(ref - td.e).max(), 0, 8)

        # TDA without equilibrium_solvation
        mf = solvent.PE(mol.RHF(), potfile).run()
        td = solvent.PE(mf.TDA(), potfile).run()
        ref = numpy.array([0.1506431269137912, 0.338254809044639, 0.4471487090255076])
        self.assertAlmostEqual(abs(ref - td.e).max(), 0, 8)


if __name__ == "__main__":
    print("Full Tests for pol_embed")
    unittest.main()
