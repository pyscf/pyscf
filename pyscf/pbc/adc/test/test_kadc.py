import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, adc
import unittest
import numpy as np
from pyscf.pbc.adc import kadc_rhf_ip, kadc_rhf_ea, kadc_ao2mo

def setUpModule():
    global cell, kmf, mycc, eris
    cell = make_test_cell.test_cell_n3_diffuse()
    kmf = scf.KRHF(cell, kpts=cell.make_kpts(
        [1,1,2], with_gamma_point=True), exxdiv=None).density_fit()
    kmf.conv_tol = 1e-10
    kmf.conv_tol_grad = 1e-6
    kmf.verbose = 0
    kmf.scf()

def tearDownModule():
    global cell, kmf
    cell.stdout.close()
    del cell, kmf

class KnownValues(unittest.TestCase):
    def test_n3_diffuse(self):

        # Running HF and MP2 and MP3 with 1x1x2 Monkhorst-Pack k-point mesh
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1729762695142645, 4)

        myadc  = adc.KRADC(kmf)
        e_mp2, t1, t2 = myadc.kernel_gs()
        self.assertAlmostEqual(e_mp2, -0.11464808675651168, 4)

        # IP-ADC(2)
        myadc.method = 'adc(2)'
        e, v, p, x = myadc.kernel(nroots=3,kptlist=[0])
        self.assertAlmostEqual(e[0][0],-1.1684386160049567,4)
        self.assertAlmostEqual(e[0][1],-1.0944210679002468,4)
        self.assertAlmostEqual(e[0][2],-1.0944210623845938,4)

        # Ensure kptlist behaves correctly
        e, v, p, x = myadc.kernel(nroots=3,kptlist=[1])
        self.assertAlmostEqual(e[0][0], -1.0362468821854942, 4)
        self.assertAlmostEqual(e[0][1], -1.0015050525294895, 4)
        self.assertAlmostEqual(e[0][2], -1.0015050525294888, 4)

        # IP-ADC(3)
        myadc.method = 'adc(3)'
        e, v, p, x = myadc.kernel(nroots=3,kptlist=[0])
        self.assertAlmostEqual(e[0][0],-1.1740951190044178, 4)
        self.assertAlmostEqual(e[0][1],-1.1536065878793587, 4)
        self.assertAlmostEqual(e[0][2],-1.153606584783398, 4)

    def make_small_reference(self):
        cell = gto.M(atom='H 0 0 0; H 0 0 .74', a=np.eye(3)*4,
                     basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                     precision=1e-7, verbose=0)
        self.mf = scf.KRHF(cell, cell.make_kpts([2, 1, 1]), exxdiv=None)
        self.mf = self.mf.density_fit(auxbasis='weigend').run(conv_tol=1e-11)

    def check_density(self, method, excitation):
        self.make_small_reference()
        obj = adc.KRADC(self.mf)
        obj.method, obj.method_type = method, excitation
        obj.kernel(nroots=1, kptlist=[0])
        dm = np.asarray(obj.make_rdm1(kptlist=[0]))[0, 0]
        ao = np.asarray(obj.make_rdm1(kptlist=[0], ao_repr=True))[0, 0]
        c = np.asarray(self.mf.mo_coeff)
        np.testing.assert_allclose(ao, c @ dm @ c.conj().transpose(0, 2, 1), atol=1e-10)
        np.testing.assert_allclose(dm, dm.conj().transpose(0, 2, 1), atol=1e-10)
        expected = self.mf.cell.nelectron*2 + (1 if excitation == 'ea' else -1)
        self.assertAlmostEqual(np.trace(dm, axis1=1, axis2=2).sum().real, expected, places=7)
        excited = obj._adc_es
        vector = excited.renormalize_eigenvectors(0, np.asarray(excited.U[0]).T.copy(), 1)[:, 0]
        module = kadc_rhf_ea if excitation == 'ea' else kadc_rhf_ip
        ref = module.make_rdm1_eigenvectors_slow(excited, vector, vector, 0)
        np.testing.assert_allclose(dm, ref, atol=1e-10)

    def test_ip_adc2(self):
        self.check_density('adc(2)', 'ip')

    def test_ea_adc2(self):
        self.check_density('adc(2)', 'ea')

    def test_ip_adc3(self):
        self.check_density('adc(3)', 'ip')

    def test_ea_adc3(self):
        self.check_density('adc(3)', 'ea')

    def test_disk_backed_integrals(self):
        self.make_small_reference()
        obj = adc.KRADC(self.mf)
        obj.method = 'adc(3)'
        ref = kadc_ao2mo.transform_integrals_incore(obj)
        out = kadc_ao2mo.transform_integrals_outcore(obj)
        try:
            for name in ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvv', 'ovvo', 'vvvv'):
                with self.subTest(block=name):
                    np.testing.assert_allclose(getattr(out, name)[:], getattr(ref, name), atol=1e-10)
        finally:
            out.feri.close()

    def check_spectrum_storage(self, excitation):
        self.make_small_reference()
        ref = adc.KRADC(self.mf)
        ref.method, ref.method_type = 'adc(3)', excitation
        er, _, pr, _ = ref.kernel(nroots=1, kptlist=[0])
        dref = ref.make_ref_rdm1()
        for transform in (kadc_ao2mo.transform_integrals_incore,
                          kadc_ao2mo.transform_integrals_outcore):
            with self.subTest(storage=transform.__name__):
                out = adc.KRADC(self.mf)
                out.method, out.method_type = 'adc(3)', excitation
                out.chnk_size = out.get_chnk_size()
                eris = transform(out)
                eo, _, po, _ = out.kernel(nroots=1, kptlist=[0], eris=eris)
                np.testing.assert_allclose(eo, er, atol=1e-9)
                np.testing.assert_allclose(po, pr, atol=1e-8)
                np.testing.assert_allclose(out.make_ref_rdm1(), dref, atol=1e-9)
                if hasattr(eris, 'feri'):
                    eris.feri.close()
        np.testing.assert_allclose(dref, np.asarray(dref).conj().transpose(0, 2, 1), atol=1e-10)
        self.assertAlmostEqual(np.trace(dref, axis1=1, axis2=2).sum().real,
                               self.mf.cell.nelectron*2, places=8)

    def test_ip_storage_spectrum(self):
        self.check_spectrum_storage('ip')

    def test_ea_storage_spectrum(self):
        self.check_spectrum_storage('ea')

if __name__ == '__main__':
    print("kadc_ip_basic tests")
    unittest.main()
