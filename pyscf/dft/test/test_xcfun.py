#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto, scf
from pyscf import dft
from pyscf import lib

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = 'h 0 0 0; h 1 .5 0; h 0 4 1; h 1 0 .2'
mol.basis = 'aug-ccpvdz'
mol.build()
dm = scf.RHF(mol).run(conv_tol=1e-14).make_rdm1()
mf = dft.RKS(mol)
mf.grids.atom_grid = {"H": (50, 110)}
mf.prune = None
mf.grids.build(with_non0tab=False)
nao = mol.nao_nr()
ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_parse_xc(self):
        hyb, fn_facs = dft.libxc.parse_xc('.5*HF+.5*B3LYP,.5*VWN')
        self.assertAlmostEqual(hyb, .6, 12)
        self.assertEqual([x[0] for x in fn_facs], [1,106,131,7])
        self.assertTrue(numpy.allclose([x[1] for x in fn_facs],
                                       (0.04, 0.36, 0.405, 0.595)))

        hyb, fn_facs = dft.xcfun.parse_xc('M05')
        self.assertAlmostEqual(hyb, 0.28, 9)

    def test_lda(self):
        e,v = dft.xcfun.eval_xc('lda,', rho[0], deriv=1)[:2]
        self.assertAlmostEqual(lib.finger(e)   , -2.2883331992727571, 8)
        self.assertAlmostEqual(lib.finger(v[0]), -3.0511109323636783, 8)

    def test_lyp(self):
        e,v,f = dft.xcfun.eval_xc(',LYP', rho, deriv=2)[:3]
        self.assertAlmostEqual(numpy.dot(rho[0],e), -62.114576182676615, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[0]),-81.771670866308455, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],v[1]), 27.485383255125743, 8)
        self.assertAlmostEqual(numpy.dot(rho[0],f[0]), 186.823806251777, 7)
        self.assertAlmostEqual(numpy.dot(rho[0],f[1]), -3391.2428894571085, 6)
        self.assertAlmostEqual(numpy.dot(rho[0],f[2]), 0, 9)

    def test_beckex(self):
        rho =(numpy.array([1.    , 1., 0., 0.]).reshape(-1,1),
              numpy.array([    .8, 1., 0., 0.]).reshape(-1,1))
        e,v,f = dft.xcfun.eval_xc('b88,', rho, spin=1, deriv=2)[:3]
        self.assertAlmostEqual(lib.finger(e)   ,-0.9061911523772116   , 9)
        self.assertAlmostEqual(lib.finger(v[0]),-1.8531364353196298   , 9)
        self.assertAlmostEqual(lib.finger(v[1]),-0.0018308066137967724, 9)
        self.assertAlmostEqual(lib.finger(f[0]),-0.21602284426026866  , 9)
        self.assertAlmostEqual(lib.finger(f[1]), 0.0072053520662545617, 9)
        self.assertAlmostEqual(lib.finger(f[2]), 0.0002275350850255538, 9)

    def test_m05x(self):
        rho =(numpy.array([1., 1., 0., 0., 0., 0.165 ]).reshape(-1,1),
              numpy.array([.8, 1., 0., 0., 0., 0.1050]).reshape(-1,1))
        test_ref = numpy.array([-1.57876583, -2.12127045,-2.11264351,-0.00315462,
                                 0.00000000, -0.00444560, 3.45640232, 4.4349756])
        exc, vxc, fxc, kxc = dft.xcfun.eval_xc('m05,', rho, 1, deriv=1)
        self.assertAlmostEqual(float(exc)*1.8, test_ref[0], 5)
        self.assertAlmostEqual(abs(vxc[0]-test_ref[1:3]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[1]-test_ref[3:6]).max(), 0, 6)
        self.assertAlmostEqual(abs(vxc[3]-test_ref[6:8]).max(), 0, 5)

    def test_camb3lyp(self):
        rho = numpy.array([1., 1., 0.1, 0.1]).reshape(-1,1)
        exc, vxc, fxc, kxc = dft.libxc.eval_xc('camb3lyp', rho, 0, deriv=1)
        self.assertAlmostEqual(float(exc), -0.5752559666317147, 5)
        self.assertAlmostEqual(float(vxc[0]), -0.7709812578936763, 5)
        self.assertAlmostEqual(float(vxc[1]), -0.0029862221286189846, 7)

if __name__ == "__main__":
    print("Test xcfun")
    unittest.main()

