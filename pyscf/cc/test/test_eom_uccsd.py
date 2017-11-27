#!/usr/bin/env python
import unittest
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc import gccsd
from pyscf.cc import uccsd
from pyscf.cc import eom_uccsd

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = 'cc-pvdz'
mol.spin = 0
mol.build()
mf = scf.UHF(mol).run(conv_tol=1e-12)

mol1 = gto.Mole()
mol1.verbose = 0
mol1.atom = [['O', (0.,   0., 0.)],
             ['O', (1.21, 0., 0.)]]
mol1.basis = 'cc-pvdz'
mol1.spin = 2
mol1.build()
mf1 = scf.UHF(mol1).run(conv_tol=1e-12)
orbspin = scf.addons.get_ghf_orbspin(mf1.mo_energy, mf1.mo_occ)

nocca, noccb = mol1.nelec
nvira, nvirb = 19, 20
nmo = mol1.nao_nr()
numpy.random.seed(12)
mf1.mo_coeff = numpy.random.random((2,nmo,nmo)) - .9

ucc1 = cc.UCCSD(mf1)
eris1 = ucc1.ao2mo()

numpy.random.seed(11)
no = nocca + noccb
nv = nvira + nvirb
r1 = numpy.random.random((no,nv)) - .9
r2 = numpy.random.random((no,no,nv,nv)) - .9
r2 = r2 - r2.transpose(1,0,2,3)
r2 = r2 - r2.transpose(0,1,3,2)
r1 = cc.addons.spin2spatial(r1, orbspin)
r2 = cc.addons.spin2spatial(r2, orbspin)
r1,r2 = eom_uccsd.vector_to_amplitudes_ee(
    eom_uccsd.amplitudes_to_vector_ee(r1,r2), ucc1.nmo, ucc1.nocc)
ucc1.t1 = r1
ucc1.t2 = r2

#ucc = cc.UCCSD(mf)
#ucc.conv_tol = 1e-10
#ecc, t1, t2 = ucc.kernel()

class KnownValues(unittest.TestCase):
    def test_frozen(self):
        mf1 = scf.UHF(mol1).run()
        # Freeze 1s electrons
        frozen = [[0,1], [0,1]]
        ucc = cc.UCCSD(mf1, frozen=frozen)
        ecc, t1, t2 = ucc.kernel()
        self.assertAlmostEqual(ecc, -0.34869875247588372, 8)

    def test_ipccsd(self):
        e,v = ucc.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.4335604332073799, 6)
        e,v = ucc.ipccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[2], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[4], 0.6782876002229172, 6)

        e,v = ucc.ipccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.5187659896045407, 6)

    def test_ipccsd_koopmans(self):
        e,v = ucc.ipccsd(nroots=8, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[2], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[4], 0.6782876002229172, 6)

    def test_eaccsd(self):
        e,v = ucc.eaccsd(nroots=1)
        self.assertAlmostEqual(e, 0.16737886338859731, 6)
        e,v = ucc.eaccsd(nroots=8)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[2], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[4], 0.51006797826488071, 6)

        e,v = ucc.eaccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[2], 0.24027613852009164, 6)

    def test_eaccsd_koopmans(self):
        e,v = ucc.eaccsd(nroots=8, koopmans=True)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[2], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[4], 0.73171753944933049, 6)


    def test_eomee(self):
        self.assertAlmostEqual(ecc, -0.2133432430989155, 6)
        e,v = ucc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

        e,v = ucc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)


    def test_ucc_eris(self):
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.oooo)),  128.42885326720619, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovoo)), -399.28919360279713, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovov)), -224.00460853672791, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.oovv)), -447.72717709485414, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovvo)),  49.074048887232081, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovvv)), -25.630982098908788, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.vvvv)),  863.81970681432608, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OOOO)),  41.795102437876693, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVOO)),  229.4383918957935 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVOV)),  17.093123984412543, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OOVV)),  176.634046183583  , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVVO)),  35.111442834162318, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVVV)), -85.190558629451175, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.VVVV)), -310.85064801079443, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ooOO)), -12.926905912728785, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovOO)),  192.99512386092005, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovOV)), -97.828790191416118, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ooVV)),  222.97003270555348, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovVO)), -185.61030877295417, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.ovVV)), -269.242250150269  , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.vvVV)), -122.64002905276595, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVoo)),  101.19441345421897, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OOvv)), -661.88321182324262, 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVvo)),  167.1512319411807 , 9)
        self.assertAlmostEqual(lib.finger(numpy.asarray(eris1.OVvv)),  345.67469254124296, 9)

    def test_ucc_update_amps(self):
        gmf = scf.addons.convert_to_ghf(mf1)
        gcc1 = gccsd.GCCSD(gmf)
        r1g = gcc1.spatial2spin(ucc1.t1, orbspin)
        r2g = gcc1.spatial2spin(ucc1.t2, orbspin)
        r1g, r2g = gcc1.update_amps(r1g, r2g, gcc1.ao2mo())
        u1g = gcc1.spin2spatial(r1g, orbspin)
        u2g = gcc1.spin2spatial(r2g, orbspin)
        t1, t2 = ucc1.update_amps(ucc1.t1, ucc1.t2, eris1)
        self.assertAlmostEqual(abs(r1g-gcc1.spatial2spin(t1, orbspin)).max(), 0, 8)
        self.assertAlmostEqual(abs(r2g-gcc1.spatial2spin(t2, orbspin)).max(), 0, 8)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t1, orbspin)), 94057615.146789819, 8)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t2, orbspin)),-179055267.95981187, 8)
        self.assertAlmostEqual(uccsd.energy(ucc1, r1, r2, eris1), -608386.34248185484, 8)
        e0, t1, t2 = ucc1.init_amps(eris1)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t1, orbspin)), -407067.51683501503, 8)
        self.assertAlmostEqual(lib.finger(cc.addons.spatial2spin(t2, orbspin)), 39416.531810862252, 8)
        self.assertAlmostEqual(e0, -954411.05216172978, 2)

    def test_ucc_eomee_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r1 = cc.addons.spin2spatial(r1, orbspin)
        r2 = cc.addons.spin2spatial(r2, orbspin)
        vec = eom_uccsd.amplitudes_to_vector_ee(r1,r2)
        vec1 = eom_uccsd.eomee_ccsd_matvec(eom_uccsd.EOMEE(ucc1), vec)
        self.assertAlmostEqual(lib.finger(vec1), -23390339.547188461, 6)

    def test_ucc_eomee_ccsd_diag(self):
        vec1, vec2 = eom_uccsd.EOMEE(ucc1).get_diag()
        self.assertAlmostEqual(lib.finger(vec1), -910866.47047512955, 6)
        self.assertAlmostEqual(lib.finger(vec2), -196214.91230670043, 6)

    def test_ucc_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        myeom = eom_uccsd.EOMEESpinFlip(ucc1)
        vec = numpy.random.random(myeom.vector_size()) - .9
        vec1 = eom_uccsd.eomsf_ccsd_matvec(myeom, vec)
        self.assertAlmostEqual(lib.finger(vec1), -30396412.288341973, 8)

#    def test_ucc_eomip_matvec(self):
#
#    def test_ucc_eomea_matvec(self):

if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()

