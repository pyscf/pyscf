import os
import unittest
import numpy as np
from pyscf import gto, scf, cc, lib
from pyscf.cc import momgfccsd


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
        cls.mol.basis = "6-31g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol_grad = 1e-8
        cls.mf.kernel()

        cls.mycc = cc.ccsd.CCSD(cls.mf)
        cls.mycc.conv_tol = 1e-10
        cls.mycc.kernel()
        cls.mycc.solve_lambda()

        gfcc = momgfccsd.MomGFCCSD(cls.mycc, niter=(5, 5))
        imds = gfcc.make_imds()
        cls.hole_moments = gfcc.build_hole_moments(imds=imds)
        cls.part_moments = gfcc.build_part_moments(imds=imds)

        cls.ips = {
                0: 0.4390402520837295,
                1: 0.43398194103807186,
                2: 0.43139244825126516,
                3: 0.42846325587576917,
                4: 0.4282277692533328,
                5: 0.42792429922566255,
                (2, True): 0.43138084146173405,
                (2, True, True): 0.43138084146173455,
        }
        cls.eas = {
                0: 0.20957238161541483,
                1: 0.19259609010353557,
                2: 0.19169190195958974,
                3: 0.19093540225391029,
                4: 0.19072953794288366,
                5: 0.19054512389397538,
                (2, True): 0.19262006823074979,
                (2, True, True): 0.19153041329652043,
        }

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.mycc, cls.hole_moments, cls.part_moments, cls.ips, cls.eas

    def test_lambda_assertion(self):
        with lib.temporary_env(self.mycc, l1=None, l2=None):
            gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(0, 0))
            self.assertRaises(ValueError, gfcc.kernel)

    def _test_moments(self, e, v, nmax, ref):
        m1 = ref[:nmax+1] / np.max(np.abs(ref[:nmax+1]), axis=(1, 2), keepdims=True)
        m2 = lib.einsum("xk,yk,nk->nxy", v[0], v[1].conj(), e[None]**np.arange(nmax+1)[:, None])
        m2 /= np.max(np.abs(m2), axis=(1, 2), keepdims=True)
        self.assertAlmostEqual(np.max(np.abs(m1-m2)), 0.0, 7)

    def _test_niter(self, niter):
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        eh, vh, ep, vp = gfcc.kernel()
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[niter])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[niter])
        self._test_moments(eh, vh, 2*niter+1, self.hole_moments)
        self._test_moments(ep, vp, 2*niter+1, self.part_moments)

    def test_0(self):
        self._test_niter(0)

    def test_1(self):
        self._test_niter(1)

    def test_2(self):
        self._test_niter(2)

    def test_3(self):
        self._test_niter(3)

    def test_4(self):
        self._test_niter(4)

    def test_5(self):
        self._test_niter(5)

    def test_amp_input(self):
        niter = 2
        imds = cc.eom_rccsd._IMDS(self.mycc)
        imds.make_ip()
        imds.make_ea()
        t1, t2, l1, l2 = self.mycc.t1, self.mycc.t2, self.mycc.l1, self.mycc.l2
        with lib.temporary_env(self.mycc, t1=None, t2=None, l1=None, l2=None):
            gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
            eh, vh, ep, vp = gfcc.kernel(t1=t1, t2=t2, l1=l1, l2=l2, imds=imds)
            self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[niter])
            self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[niter])
            self._test_moments(eh, vh, 2*niter+1, self.hole_moments)
            self._test_moments(ep, vp, 2*niter+1, self.part_moments)

    def test_mom_input(self):
        niter = 2
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        hole_moments = self.hole_moments[:2*niter+2]
        part_moments = self.part_moments[:2*niter+2]
        eh, vh, ep, vp = gfcc.kernel(hole_moments=hole_moments, part_moments=part_moments)
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[niter])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[niter])
        self._test_moments(eh, vh, 2*niter+1, self.hole_moments)
        self._test_moments(ep, vp, 2*niter+1, self.part_moments)

    def test_hermi_moments(self):
        niter = 2
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        gfcc.hermi_moments = True
        hole_moments = self.hole_moments[:2*niter+2]
        part_moments = self.part_moments[:2*niter+2]
        eh, vh, ep, vp = gfcc.kernel(hole_moments=hole_moments, part_moments=part_moments)
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[(niter, True)])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[(niter, True)])
        self._test_moments(eh, vh, 2*niter+1, 0.5*(self.hole_moments+self.hole_moments.swapaxes(1,2).conj()))
        self._test_moments(ep, vp, 2*niter+1, 0.5*(self.part_moments+self.part_moments.swapaxes(1,2).conj()))

    def test_hermi_moments(self):
        niter = 2
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        gfcc.hermi_moments = True
        gfcc.hermi_solver = True
        hole_moments = self.hole_moments[:2*niter+2]
        part_moments = self.part_moments[:2*niter+2]
        eh, vh, ep, vp = gfcc.kernel(hole_moments=hole_moments, part_moments=part_moments)
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[(niter, True, True)])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[(niter, True, True)])
        self._test_moments(eh, vh, 2*niter+1, 0.5*(self.hole_moments+self.hole_moments.swapaxes(1,2).conj()))
        self._test_moments(ep, vp, 2*niter+1, 0.5*(self.part_moments+self.part_moments.swapaxes(1,2).conj()))

    def test_misc(self):
        niter = 2
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        gfcc.reset()
        eh, vh, ep, vp = gfcc.kernel()
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[niter])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[niter])
        self._test_moments(eh, vh, 2*niter+1, self.hole_moments)
        self._test_moments(ep, vp, 2*niter+1, self.part_moments)
        dma = gfcc.make_rdm1()
        dmb = self.mycc.make_rdm1()
        self.assertAlmostEqual(np.max(np.abs(dma-dmb)), 0.0, 8)
        dma = gfcc.make_rdm1(ao_repr=True)
        dmb = self.mycc.make_rdm1(ao_repr=True)
        self.assertAlmostEqual(np.max(np.abs(dma-dmb)), 0.0, 8)

    def test_chkfile(self):
        niter = 1
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        eh, vh, ep, vp = gfcc.kernel()
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[niter])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[niter])
        self._test_moments(eh, vh, 2*niter+1, self.hole_moments)
        self._test_moments(ep, vp, 2*niter+1, self.part_moments)
        gfcc.dump_chk(chkfile="tmp.chk")
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        gfcc.update("tmp.chk")
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[niter])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[niter])
        self._test_moments(eh, vh, 2*niter+1, self.hole_moments)
        self._test_moments(ep, vp, 2*niter+1, self.part_moments)
        os.remove("tmp.chk")

    def test_density_fitting(self):
        mf = scf.RHF(self.mol)
        mf = mf.density_fit()
        mf.conv_tol_grad = 1e-8
        mf.kernel()

        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-10
        mycc.kernel()
        mycc.solve_lambda()

        niter = 3
        gfcc = momgfccsd.MomGFCCSD(self.mycc, niter=(niter, niter))
        eh, vh, ep, vp = gfcc.kernel()
        self.assertAlmostEqual(gfcc.ipgfccsd(nroots=1)[0], self.ips[niter])
        self.assertAlmostEqual(gfcc.eagfccsd(nroots=1)[0], self.eas[niter])
        self._test_moments(eh, vh, 2*niter+1, self.hole_moments)
        self._test_moments(ep, vp, 2*niter+1, self.part_moments)



if __name__ == "__main__":
    print("Tests for MomGFCCSD")
    unittest.main()
