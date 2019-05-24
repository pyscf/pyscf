#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
from functools import reduce
import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import ao2mo
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import casscf as casscf_grad
from pyscf.grad.mp2 import _shell_prange


def grad_elec(mc, mf_grad):
    mf = mf_grad.base
    mol = mf_grad.mol
    mo_energy = mc.mo_energy
    mo_coeff = mc.mo_coeff

    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    dm1 = numpy.zeros((nmo,nmo))
    dm1[numpy.diag_indices(ncore)] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2
        dm2[i,i,ncore:nocc,ncore:nocc] = casdm1 * 2
        dm2[ncore:nocc,ncore:nocc,i,i] = casdm1 * 2
        dm2[i,ncore:nocc,ncore:nocc,i] =-casdm1
        dm2[ncore:nocc,i,i,ncore:nocc] =-casdm1
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2

    h1 = reduce(numpy.dot, (mo_coeff.T, mc._scf.get_hcore(), mo_coeff))
    h2 = ao2mo.kernel(mf._eri, mo_coeff, compact=False).reshape([nmo]*4)

# Generalized Fock, according to generalized Brillouin theorm
# Adv. Chem. Phys., 69, 63
    gfock = numpy.dot(h1, dm1)
    gfock+= numpy.einsum('iqrs,qjsr->ij', h2, dm2)
    gfock = (gfock + gfock.T) * .5
    dme0 = reduce(numpy.dot, (mo_coeff[:,:nocc], gfock[:nocc,:nocc], mo_coeff[:,:nocc].T))

    dm1 = reduce(numpy.dot, (mo_coeff, dm1, mo_coeff.T))
    dm2 = lib.einsum('ijkl,pi,qj,rk,sl->pqrs', dm2,
                     mo_coeff, mo_coeff, mo_coeff, mo_coeff)

    eri_deriv1 = mol.intor('int2e_ip1', comp=3).reshape(3,nao,nao,nao,nao)

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm1)
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        de[k] -= numpy.einsum('xijkl,ijkl->x', eri_deriv1[:,p0:p1], dm2[p0:p1]) * 2

    return de

mol = gto.Mole()
mol.atom = 'N 0 0 0; N 0 0 1.2; H 1 1 0; H 1 1 1.2'
mol.verbose = 5
mol.output = '/dev/null'
mol.build()
mf = scf.RHF(mol).run(conv_tol=1e-14)

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_casscf_grad(self):
        mc = mcscf.CASSCF(mf, 4, 4).run()
        g1 = casscf_grad.Gradients(mc).kernel()
        self.assertAlmostEqual(lib.finger(g1), -0.065094188906156134, 7)

        g1ref = grad_elec(mc, mf.nuc_grad_method())
        g1ref += rhf_grad.grad_nuc(mol)
        self.assertAlmostEqual(abs(g1-g1ref).max(), 0, 9)

        mcs = mc.as_scanner()
        pmol = mol.copy()
        e1 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e2 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        self.assertAlmostEqual(g1[1,2], (e1-e2)/0.002*lib.param.BOHR, 4)

#    def test_frozen(self):
#        mc = mcscf.CASSCF(mf, 4, 4).set(frozen=2).run()
#        gscan = mc.nuc_grad_method().as_scanner()
#        g1 = gscan(mol)[1]
#        self.assertAlmostEqual(lib.finger(g1), -0.065094188906156134, 9)
#
#        mcs = mc.as_scanner()
#        pmol = mol.copy()
#        e1 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
#        e2 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
#        self.assertAlmostEqual(g1[1,2], (e1-e2)/0.002*lib.param.BOHR, 4)

    def test_scanner(self):
        mc = mcscf.CASSCF(mf, 4, 4)
        gs = mc.nuc_grad_method().as_scanner().as_scanner()
        e, g1 = gs(mol.atom, atmlst=range(4))
        self.assertAlmostEqual(e, -108.39289688030243, 9)
        self.assertAlmostEqual(lib.finger(g1), -0.065094188906156134, 7)

    def test_state_specific_scanner(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        mc = mcscf.CASSCF(mf, 4, 4)
        gs = mc.state_specific_(2).nuc_grad_method().as_scanner()
        e, de = gs(mol)
        self.assertAlmostEqual(e, -108.68788613661442, 7)
        self.assertAlmostEqual(lib.finger(de), -0.10695162143777398, 5)

        mcs = gs.base
        pmol = mol.copy()
        e1 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201'))
        e2 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199'))
        self.assertAlmostEqual(de[1,2], (e1-e2)/0.002*lib.param.BOHR, 5)

    def test_state_average_scanner(self):
        mc = mcscf.CASSCF(mf, 4, 4)
        gs = mc.state_average_([0.5, 0.5]).nuc_grad_method().as_scanner()
        e, de = gs(mol)
        self.assertAlmostEqual(e, -108.38384621394407, 9)
        self.assertAlmostEqual(lib.finger(de), -0.1034416391211391, 7)

        mcs = gs.base
        pmol = mol.copy()
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e1 = mcs.e_average
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        e2 = mcs.e_average
        self.assertAlmostEqual(de[1,2], (e1-e2)/0.002*lib.param.BOHR, 4)

    def test_state_average_mix_scanner(self):
        mc = mcscf.CASSCF(mf, 4, 4)
        mc = mcscf.addons.state_average_mix_(mc, [mc.fcisolver, mc.fcisolver], (.5, .5))
        gs = mc.nuc_grad_method().as_scanner()
        e, de = gs(mol)
        self.assertAlmostEqual(e, -108.39289688022976, 9)
        self.assertAlmostEqual(lib.finger(de), -0.06509352771703128, 7)

        mcs = gs.base
        pmol = mol.copy()
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e1 = mcs.e_average
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        e2 = mcs.e_average
        self.assertAlmostEqual(de[1,2], (e1-e2)/0.002*lib.param.BOHR, 4)

    def test_with_x2c_scanner(self):
        with lib.light_speed(20.):
            mc = mcscf.CASSCF(mf.x2c(), 4, 4).run()
            gscan = mc.nuc_grad_method().as_scanner()
            g1 = gscan(mol)[1]
            self.assertAlmostEqual(lib.finger(g1), -0.070281684620797591, 7)

            mcs = mcscf.CASSCF(mf, 4, 4).as_scanner().x2c()
            e1 = mcs('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2')
            e2 = mcs('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2')
            self.assertAlmostEqual(g1[1,2], (e1-e2)/0.002*lib.param.BOHR, 5)

    def test_with_qmmm_scanner(self):
        from pyscf import qmmm
        mol = gto.Mole()
        mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                       H                 -0.00000000   -0.84695236    0.59109389
                       H                 -0.00000000    0.89830571    0.52404783 '''
        mol.verbose = 0
        mol.basis = '6-31g'
        mol.build()

        coords = [(0.5,0.6,0.1)]
        #coords = [(0.0,0.0,0.0)]
        charges = [-0.1]
        mf = qmmm.add_mm_charges(scf.RHF(mol), coords, charges)
        mc = mcscf.CASSCF(mf, 4, 4).as_scanner()
        e_tot, g = mc.nuc_grad_method().as_scanner()(mol)
        self.assertAlmostEqual(e_tot, -76.0461574155984, 7)
        self.assertAlmostEqual(lib.finger(g), 0.042835374915102364, 6)
        e1 = mc(''' O                  0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        e2 = mc(''' O                 -0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        ref = (e1 - e2)/0.002 * lib.param.BOHR
        self.assertAlmostEqual(g[0,0], ref, 4)

        mf = scf.RHF(mol)
        mc = qmmm.add_mm_charges(mcscf.CASSCF(mf, 4, 4).as_scanner(), coords, charges)
        e_tot, g = mc.nuc_grad_method().as_scanner()(mol)
        self.assertAlmostEqual(e_tot, -76.0461574155984, 7)
        self.assertAlmostEqual(lib.finger(g), 0.042835374915102364, 6)

    def test_symmetrize(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True)
        g = mol.RHF.run().CASSCF(4, 4).run().Gradients().kernel()
        self.assertAlmostEqual(lib.finger(g), 0.12355818572359845, 7)


if __name__ == "__main__":
    print("Tests for CASSCF gradients")
    unittest.main()

