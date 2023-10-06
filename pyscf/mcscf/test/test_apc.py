import unittest
from pyscf import gto, scf, lib
import numpy as np
from pyscf.mcscf import apc

APC_VERBOSE = 5

class KnownValues(unittest.TestCase):

    def test_water(self):
        mol = gto.Mole()
        mol.atom = [('O', [0.0, 0.0, -0.13209669380597672]),
                    ('H', [0.0, 1.4315287853817316, 0.9797000689025815]),
                    ('H', [0.0, -1.4315287853817316, 0.9797000689025815])]
        mol.basis = "6-31g"
        mol.unit = "bohr"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        #With (nelec,ncas) size constraint:
        myapc = apc.APC(mf,max_size=(10,10),verbose=APC_VERBOSE)
        ncas,nelecas,casorbs = myapc.kernel()
        na,nb = nelecas
        self.assertAlmostEqual(ncas, 10)
        self.assertAlmostEqual(na, 4)
        self.assertAlmostEqual(nb, 4)
        self.assertAlmostEqual(lib.fp(np.abs(casorbs)),1.7403225684990318,4)

        #With ncas size constraint:
        myapc = apc.APC(mf,max_size=12,verbose=APC_VERBOSE)
        ncas,nelecas,casorbs = myapc.kernel()
        na,nb = nelecas
        self.assertAlmostEqual(ncas, 12)
        self.assertAlmostEqual(na, 4)
        self.assertAlmostEqual(nb, 4)
        self.assertAlmostEqual(lib.fp(np.abs(casorbs)),3.405630497055709,4)

        #With n=0
        myapc = apc.APC(mf,max_size=(2,2),n=0,verbose=APC_VERBOSE)
        ncas,nelecas,casorbs = myapc.kernel()
        na,nb = nelecas
        self.assertAlmostEqual(ncas, 2)
        self.assertAlmostEqual(na, 1)
        self.assertAlmostEqual(nb, 1)
        self.assertAlmostEqual(lib.fp(np.abs(casorbs)),6.691447764934983,4)

        #With user-input entropies:
        np.random.seed(34)
        entropies = np.random.choice(np.arange(len(mf.mo_occ)),len(mf.mo_occ),replace=False)
        chooser = apc.Chooser(mf.mo_coeff,mf.mo_occ,entropies,max_size=(8,8),verbose=APC_VERBOSE)
        ncas, nelecas, casorbs, active_idx = chooser.kernel()
        na,nb = nelecas
        self.assertAlmostEqual(ncas, 8)
        self.assertAlmostEqual(na, 3)
        self.assertAlmostEqual(nb, 3)
        self.assertAlmostEqual(lib.fp(np.abs(casorbs)),-2.449707369711791,4)

        #With user-input mos:
        mf2 = scf.RKS(mol) #example: dft MOs
        mf2.kernel()
        mf.mo_coeff = mf2.mo_coeff
        myapc = apc.APC(mf,max_size=(10,10),verbose=APC_VERBOSE)
        ncas,nelecas,casorbs = myapc.kernel()
        na,nb = nelecas
        self.assertAlmostEqual(ncas, 10)
        self.assertAlmostEqual(na, 4)
        self.assertAlmostEqual(nb, 4)
        self.assertAlmostEqual(lib.fp(np.abs(casorbs)),0.15825224792265288,4)

    def test_vinyl(self):
        mol = gto.Mole()
        mol.atom = [('C', [0.0, 1.16769663781575, -0.043031463808524656]),
                    ('C', [0.0, -1.2994536344535748, 0.15810072367732414]),
                    ('H', [0.0, 2.3842960807145257, 1.5980182111958736]),
                    ('H', [0.0, 2.087591296834979, -1.8799830935092905]),
                    ('H', [0.0, -2.9030792488761317, -1.0881451206088533])]
        mol.basis = "6-31g"
        mol.unit = "bohr"
        mol.spin = 1
        mol.build()

        #With ROHF:
        mf = scf.ROHF(mol)
        mf.kernel()
        myapc = apc.APC(mf,max_size=(10,10),verbose=APC_VERBOSE)
        ncas,nelecas,casorbs = myapc.kernel()
        na,nb = nelecas
        self.assertAlmostEqual(ncas, 9)
        self.assertAlmostEqual(na, 6)
        self.assertAlmostEqual(nb, 5)
        self.assertAlmostEqual(lib.fp(np.abs(casorbs)),-4.619108890673209,4)

        #With UHF:
        mf = scf.UHF(mol)
        mf.max_cycle = 100
        mf.kernel()
        myapc = apc.APC(mf,max_size=(10,10),verbose=APC_VERBOSE)
        ncas,nelecas,casorbs = myapc.kernel()
        na,nb = nelecas
        self.assertAlmostEqual(ncas, 9)
        self.assertAlmostEqual(na, 6)
        self.assertAlmostEqual(nb, 5)
        self.assertAlmostEqual(lib.fp(np.abs(casorbs)),4.769812303073026,4)

if __name__ == "__main__":
    print("Full tests for APC")
    unittest.main()
