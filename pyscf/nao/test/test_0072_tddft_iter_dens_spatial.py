from __future__ import print_function, division
import os,unittest
from pyscf.nao import tddft_iter
from pyscf.nao.m_comp_spatial_distributions import spatial_distribution
import h5py
import numpy as np

dname = os.path.dirname(os.path.abspath(__file__))

Ha = 27.211386024367243
td = tddft_iter(label='water', iter_broadening=0.15/Ha, xc_code='LDA,PZ',
        tol_loc=1e-4, tol_biloc=1e-6, cd=dname, verbosity=0)

class KnowValues(unittest.TestCase):
  
  def test_tddft_iter_spatial(self):
    """ Check the spatial density change distribution"""
    self.assertTrue(hasattr(td, 'xocc'))
    self.assertTrue(hasattr(td, 'xvrt'))
    self.assertEqual(td.xocc[0].shape[0], 4)
    self.assertEqual(td.xvrt[0].shape[0], 19)

    # run TDDFT
    with h5py.File(dname+"/tddft_iter_output_water_ref.hdf5", "r") as f:
        omegas = f["polarizability/frequency"][:]/Ha + 1j*td.eps
    td.comp_dens_inter_along_Eext(omegas, Eext=np.array([1.0, 1.0, 1.0]))
    np.save("density_change_pyscf.npy", td.dn)
    np.save("frequency.npy", omegas.real)
    np.save("pol_tensor.npy", td.p_mat)

    ref = h5py.File(dname+"/tddft_iter_output_water_ref.hdf5", "r")["polarizability"]
    pyscf = np.load("pol_tensor.npy")
    pyscf_freq = np.load("frequency.npy")

    for ireim, reim in enumerate(["re", "im"]):
        for i in range(3):
            for j in range(3):
                mbpt = ref["dipol_inter_iter_krylov_"+reim][j, i, :]
                if ireim == 0:
                    py = -pyscf[i, j, :].real
                elif ireim == 1:
                    py = -pyscf[i, j, :].imag
                error = np.sum(abs(mbpt-py))/py.size
                assert error < 5e-3

    # calculate spatial distribution of density change
    dn = np.load("density_change_pyscf.npy")
    freq = np.load("frequency.npy")
    box = np.array([[-15.0, 15.0],
                    [-15.0, 15.0],
                    [-15.0, 15.0]])
    dr = np.array([0.5, 0.5, 0.5])

    spd = spatial_distribution(dn, freq, box, dr = dr, label="water", 
                               tol_loc=1e-4, tol_biloc=1e-6, cd=dname)
    spd.get_spatial_density(8.35/Ha, Eext=np.array([1.0, 1.0, 1.0]))

    ref = h5py.File(dname+"/tddft_iter_output_water_ref.hdf5", "r")
    dn_mbpt = ref["field_spatial_dir_0.58_0.58_0.58_freq_8.35_inter/dens_re"][:] +\
            1.0j*ref["field_spatial_dir_0.58_0.58_0.58_freq_8.35_inter/dens_im"][:]

    Np = spd.dn_spatial.shape[1]//2
    Nm = dn_mbpt.shape[1]//2

    error = np.sum(abs(spd.dn_spatial[:, Np, :].imag - dn_mbpt[:, Nm, :].imag.T))/dn[:, Np, :].imag.size
    assert error < 1e-2



if __name__ == "__main__": unittest.main()
