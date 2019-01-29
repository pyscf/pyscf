import make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc.eom_kccsd_ghf import EOMIP, EOMEA
from pyscf.cc import eom_gccsd
import unittest

cell = make_test_cell.test_cell_n3_diffuse()

class KnownValues(unittest.TestCase):
    def test_n3_diffuse(self):
        cell = make_test_cell.test_cell_n3_diffuse()

        nmp = [1,1,2]
        '''
        # treating 1*1*2 supercell at gamma point
        supcell = super_cell(cell,nmp)
        gmf  = scf.GHF(supcell,exxdiv=None)
        ehf  = gmf.kernel()
        gcc  = cc.GCCSD(gmf)
        gcc.conv_tol=1e-12
        gcc.conv_tol_normt=1e-10
        gcc.max_cycle=250
        ecc, t1, t2 = gcc.kernel()
        print('GHF energy (supercell) %.7f \n' % (float(ehf)/2.))
        print('GCCSD correlation energy (supercell) %.7f \n' % (float(ecc)/2.))

        eom = eom_gccsd.EOMIP(gcc)
        e1, v = eom.ipccsd(nroots=2)
        eom = eom_gccsd.EOMEA(gcc)
        e2, v = eom.eaccsd(nroots=2, koopmans=True)
        '''
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        kmf = scf.KGHF(cell, kpts=cell.make_kpts(nmp, with_gamma_point=True), exxdiv=None)
        ehf2 = kmf.kernel()
        self.assertAlmostEqual(ehf2, -6.1870676561725695, 6)

        mycc = cc.KGCCSD(kmf)
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        ecc2, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ecc2, -0.0676483716898783, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(e1_obt[0][0], -1.1489469962099519, 6)
        self.assertAlmostEqual(e1_obt[0][1], -1.1489469961858796, 6)
        self.assertAlmostEqual(e1_obt[0][2], -1.1088194518036925, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, kptlist=[0])
        self.assertAlmostEqual(e2_obt[0][0], 1.2669788613362731, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2669788614703625, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.278883205515518, 6)

        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=2, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e1_obt[0][0], -0.9074337292436309, 6)
        self.assertAlmostEqual(e1_obt[0][1], -0.9074337292161299, 6)
        self.assertAlmostEqual(e1_obt[0][2], -0.9074331788469051, 6)

        eom = EOMEA(mycc)
        e2_obt, v = eom.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(e2_obt[0][0], 1.227583017804503, 6)
        self.assertAlmostEqual(e2_obt[0][1], 1.2275830178298166, 6)
        self.assertAlmostEqual(e2_obt[0][2], 1.3830379190440196, 6)
