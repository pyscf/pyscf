from __future__ import print_function, division
import os,unittest,numpy as np

def run_tddft_iter(calculator, label, freq):
    from pyscf.nao import tddft_iter
    td = tddft_iter(gpaw=calculator, iter_broadening=1e-2)
    omegas = np.linspace(freq[0], freq[freq.shape[0]-1], freq.shape[0]) + 1j*td.eps

    pxx_nonin = np.zeros(omegas.shape[0], dtype=float)
    pxx_inter = np.zeros(omegas.shape[0], dtype=float)

    vext = np.transpose(td.moms1)
    for iomega,omega in enumerate(omegas):
        pxx_nonin[iomega] = -np.dot(td.apply_rf0(vext[0,:], omega), vext[0,:]).imag

    pxx_inter = td.comp_polariz_inter_xx(omegas).imag

    return pxx_nonin, pxx_inter



#try:
    ## run Siesta first
    #from ase.units import Ry, eV, Ha
    #from ase.calculators.siesta import Siesta
    #from ase import Atoms

    #H2O = Atoms("H2O", np.array([[0.0, -0.757, 0.587],
                                 #[0.0, +0.757, 0.587],
                                 #[0.0, 0.0, 0.0]]),
                       #cell = np.eye(3, dtype=float))
    ##H2O.center(vacuum=3.5)
    ## unit cell 1 0 0
    ##           0 1 0   => siesta default unit cell use
    ##           0 0 1

    #siesta_calc = Siesta(
            #mesh_cutoff=150 * Ry,
            #basis_set='SZ',
            #energy_shift=(10 * 10**-3) * eV,
            #xc = "PBE",
            #fdf_arguments={
               #'SCFMustConverge': False,
               #'COOP.Write': True,
               #'WriteDenchar': True,
               #'PAO.BasisType': 'split',
               #'DM.Tolerance': 1e-4,
               #'DM.MixingWeight': 0.01,
               #'MaxSCFIterations': 150,
               #'DM.NumberPulay': 4})

    #H2O.set_calculator(siesta_calc)
    #efree_siesta = H2O.get_potential_energy()

    ## run gpaw
    #from gpaw import GPAW, PoissonSolver
    #H2O_gp = Atoms("H2O", np.array([[0.0, -0.757, 0.587],
                                 #[0.0, +0.757, 0.587],
                                 #[0.0, 0.0, 0.0]]))
    #H2O_gp.center(vacuum=3.5)
    #convergence = {'density': 1e-7}
    #poissonsolver = PoissonSolver(eps=1e-14, remove_moment=1 + 3)
    #gpaw_calc = GPAW(xc='PBE', h=0.3, nbands=6,
            #convergence=convergence, poissonsolver=poissonsolver,
            #mode='lcao', setups="sg15", txt=None)

    #H2O_gp.set_calculator(gpaw_calc)
    #efree_gpaw = H2O_gp.get_potential_energy()
    #dft = True

#except:
  #dft = False



class KnowValues(unittest.TestCase):

  def test_gpaw_vs_siesta_tddft_iter(self):
    """ init ao_log with it radial orbitals from GPAW """

    print('00025   :  just do nothing...')
    return 
    
    if not dft: return
    omegas = np.linspace(0.0,2.0,500)
    pxx = {"nonin":
            {"siesta": np.zeros(omegas.shape[0], dtype=float),
             "gpaw": np.zeros(omegas.shape[0], dtype=float)},
           "inter":
            {"siesta": np.zeros(omegas.shape[0], dtype=float),
             "gpaw": np.zeros(omegas.shape[0], dtype=float)}
            }

    pxx["nonin"]["siesta"], pxx["inter"]["siesta"] = run_tddft_iter(siesta_calc, "siesta", omegas)
    pxx["nonin"]["gpaw"], pxx["inter"]["gpaw"] = run_tddft_iter(gpaw_calc, "gpaw", omegas)

    import ase.units as un
    for key, val in pxx.items():
      freq_shift = abs(omegas[np.argmax(val["siesta"])] - omegas[np.argmax(val["gpaw"])])
      self.assertLess(freq_shift*un.Ha, 5.0)


if __name__ == "__main__": unittest.main()
