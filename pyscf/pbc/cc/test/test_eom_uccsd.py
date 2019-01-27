from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf.pbc.cc.eom_kccsd_uhf import EOMIP, EOMEA
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP as rEOMIP
from pyscf.pbc.cc.eom_kccsd_rhf import EOMEA as rEOMEA
from pyscf.cc import eom_uccsd
import unittest

class Test(unittest.TestCase):
    def test_he(self):
        cell = gto.Cell()
        cell.atom='''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = { 'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        nmp = [1,1,2]
        ##nmp = [1,1,1]
        ##'''
        ## treating 1*1*2 supercell at gamma point
        #supcell = super_cell(cell,nmp)
        #gmf  = scf.UHF(supcell,exxdiv=None)
        #ehf  = gmf.kernel()
        #gcc  = cc.UCCSD(gmf)
        #gcc.conv_tol=1e-12
        #gcc.conv_tol_normt=1e-10
        #gcc.max_cycle=250
        #ecc, t1, t2 = gcc.kernel()
        #print('UHF energy (supercell) %.7f \n' % (float(ehf)/2.))
        #print('UCCSD correlation energy (supercell) %.7f \n' % (float(ecc)/2.))

        #eom = eom_uccsd.EOMIP(gcc)
        #e1, v = eom.ipccsd(nroots=2)
        #eom = eom_uccsd.EOMEA(gcc)
        #e2, v = eom.eaccsd(nroots=2, koopmans=True)
        ##'''
        #exit()
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        #kmf = scf.KUHF(cell, kpts=cell.make_kpts(nmp), exxdiv=None)
        kmf = scf.KRHF(cell, kpts=cell.make_kpts(nmp), exxdiv=None)
        ehf2 = kmf.kernel()

        mycc = cc.KUCCSD(kmf)
        #mycc = cc.KRCCSD(kmf)
        mycc.conv_tol = 1e-7
        mycc.conv_tol_normt = 1e-7
        mycc.max_cycle=250
        ecc2, t1, t2 = mycc.kernel()
        print('UHF energy %.7f \n' % (float(ehf2)))
        print('UCCSD correlation energy  %.7f \n' % (float(ecc2)))

        #EOM-CCSD root 0 E = -1.23768083043397  qpwt = 0.967562
        #EOM-CCSD root 1 E = -1.215047888039312  qpwt = 0.964023
        #EOM-CCSD root 2 E = -1.183328480541004  qpwt = 0.976061
        #EOM-CCSD root 3 E = -1.183328552049644  qpwt = 0.976061
        eom = EOMIP(mycc)
        e1_obt, v = eom.ipccsd(nroots=3, kptlist=[0])
        ##eom = rEOMIP(mycc)
        ##e1_obt, v = eom.ipccsd(nroots=4, koopmans=True, kptlist=[0])
        ##eom = EOMEA(mycc)
        ##e2_obt, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[0])
        ##print(e1_obt)
        ##print(e2_obt)
        ##assert ehf/2 - ehf2 < 1e-10
        ##assert ecc/2 - ecc2 < 1e-10
        ##assert e1[0]-(e1_obt[0][0]) < 1e-7
        ##assert e2[0]-(e2_obt[0][0]) < 1e-7
