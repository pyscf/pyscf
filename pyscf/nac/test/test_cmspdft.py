import numpy as np
from pyscf import gto, scf,mcpdft
import unittest

om_ta_alpha = [0.8, 0.9, # H, He
    1.8, 1.4, # Li, Be
        1.3, 1.1, 0.9, 0.9, 0.9, 0.9, # B - Ne
    1.4, 1.3, # Na, Mg
        1.3, 1.2, 1.1, 1.0, 1.0, 1.0, # Al - Ar
    1.5, 1.4, # K, Ca
            1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, # Sc - Zn
        1.1, 1.0, 0.9, 0.9, 0.9, 0.9] # Ga - Kr
def om_treutler_ahlrichs(n, chg, *args, **kwargs):
    '''
    "Treutler-Ahlrichs" as implemented in OpenMolcas
    '''
    r = np.empty(n)
    dr = np.empty(n)
    alpha = om_ta_alpha[chg-1]
    step = 2.0 / (n+1) # = numpy.pi / (n+1)
    ln2 = alpha / np.log(2)
    for i in range(n):
        x = (i+1)*step - 1 # = numpy.cos((i+1)*step)
        r [i] = -ln2*(1+x)**.6 * np.log((1-x)/2)
        dr[i] = (step #* numpy.sin((i+1)*step)
                * ln2*(1+x)**.6 *(-.6/(1+x)*np.log((1-x)/2)+1/(1-x)))
    return r[::-1], dr[::-1]

quasi_ultrafine = {'atom_grid': (99,590),
    'radi_method': om_treutler_ahlrichs,
    'prune': False,
    'radii_adjust': None}

def diatomic(atom1, atom2, r, basis, ncas, nelecas, nstates,
             charge=None, spin=None, symmetry=False, cas_irrep=None):
    global mols
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format(atom1, atom2, r)
    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=spin,
                symmetry=symmetry, verbose=0, output='/dev/null')
    mols.append(mol)

    mf = scf.RHF(mol)

    mc = mcpdft.CASSCF(mf.run(), 'ftLDA,VWN3', ncas, nelecas, grids_attr=quasi_ultrafine).set(natorb=True)
    # Quasi-ultrafine is ALMOST the same thing as
    #   ```
    #   grid input
    #   nr=100
    #   lmax=41
    #   rquad=ta
    #   nopr
    #   noro
    #   end of grid input
    #   ```
    # in SEWARD

    if spin is not None:
        s = spin*0.5

    else:
        s = (mol.nelectron % 2)*0.5

    mc.fix_spin_(ss=s*(s+1), shift=1)
    mc = mc.multi_state([1.0/float(nstates), ]*nstates, 'cms')
    mc.conv_tol = mc.conv_tol_diabatize = 1e-10
    mc.max_cycle_macro = 100
    mc.max_cyc_diabatize = 200
    mo = None

    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep(cas_irrep)

    mc.kernel(mo)
    return mc.nac_method()

def setUpModule():
    global mols 
    mols = []

def tearDownModule():
    global mols, diatomic
    [m.stdout.close() for m in mols]
    del mols, diatomic


class KnownValues(unittest.TestCase):

    def test_nac_h2_cms2ftlsda22_sto3g(self):
        # z_orb:    no
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, 'STO-3G', 2, 2, 2)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[2.24611972496342E-01,2.24611972496344E-01],
                           [-6.59892598854941E-16, 6.54230823118723E-16]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                self.assertTrue (mc_grad.base.converged, 'energy calculation not converged')
                self.assertTrue (mc_grad.converged, 'gradient calculation not converged')
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)


    def test_nac_h2_cms3ftlsda22_sto3g(self):
        # z_orb:    no
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, 'STO-3G', 2, 2, 3)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[-2.21241754295429E-01,-2.21241754290091E-01],
                           [-2.66888744475119E-12, 2.66888744475119E-12]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                self.assertTrue (mc_grad.base.converged, 'energy calculation not converged')
                self.assertTrue (mc_grad.converged, 'gradient calculation not converged')
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_nac_h2_cms2ftlsda22_631g(self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, '6-31G', 2, 2, 2)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[2.63335709207423E-01,2.63335709207421E-01],
                           [9.47391702563375E-16,-1.02050903352196E-15]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                self.assertTrue (mc_grad.base.converged, 'energy calculation not converged')
                self.assertTrue (mc_grad.converged, 'gradient calculation not converged')
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)


    def test_nac_h2_cms3ftlsda22_631g(self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, '6-31G', 2, 2, 3)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[-2.56602732575249E-01,-2.56602732575251E-01],
                           [7.94113968580962E-16, -7.74815822050330E-16]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                self.assertTrue (mc_grad.base.converged, 'energy calculation not converged')
                self.assertTrue (mc_grad.converged, 'gradient calculation not converged')
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_nac_lih_cms2ftlsda22_sto3g(self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     yes
        mc_grad = diatomic('Li', 'H', 1.5, 'STO-3G', 2, 2, 2)

        # OpenMolcas v23.02  - PC
        de_ref = np.array([[1.59470493600856E-01,-4.49149709990789E-02 ],
                           [6.72530182376632E-02,-6.72530182376630E-02 ]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                self.assertTrue (mc_grad.base.converged, 'energy calculation not converged')
                self.assertTrue (mc_grad.converged, 'gradient calculation not converged')
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_nac_lih_cms3ftlsda22_sto3g(self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     yes
        mc_grad = diatomic('Li', 'H', 2.5, 'STO-3G', 2, 2, 3)

        # OpenMolcas v23.02 -
        de_ref = np.array([[-2.61694098289507E-01, 5.88264204831044E-02],
                           [-1.18760840775087E-01, 1.18760840775087E-01]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                self.assertTrue (mc_grad.base.converged, 'energy calculation not converged')
                self.assertTrue (mc_grad.converged, 'gradient calculation not converged')
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)


if __name__ == "__main__":
    print("Full Tests for CMS-PDFT non-adiabatic couplings of diatomic molecules")
    unittest.main()
