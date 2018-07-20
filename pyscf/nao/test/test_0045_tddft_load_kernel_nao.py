from __future__ import print_function, division
import os,unittest,numpy as np
from pyscf.nao import tddft_iter
import h5py

dname = os.path.dirname(os.path.abspath(__file__))
td = tddft_iter(label='water', cd=dname, jcutoff=7, iter_broadening=1e-2, xc_code='RPA')
np.savetxt("kernel.txt", td.kernel)
np.save("kernel.npy", td.kernel)

hdf = h5py.File("kernel.hdf5", "w")
hdf.create_dataset("kernel_pack", data=td.kernel)
hdf.close()


class KnowValues(unittest.TestCase):

  def test_load_kernel(self):
    data_ref_nonin = np.loadtxt(dname+'/water.tddft_iter.omega.pxx.txt-ref')[:, 1]
    data_ref_inter = np.loadtxt(dname+'/water.tddft_iter.omega.inter.pxx.txt-ref')[:, 1]

    for form in ["txt", "npy", "hdf5"]:
      td = tddft_iter(label='water', cd=dname, iter_broadening=1e-2, xc_code='RPA', load_kernel=True, kernel_fname = "kernel." + form, kernel_format = form, kernel_path_hdf5="kernel_pack")

      omegas = np.linspace(0.0,2.0,150)+1j*td.eps
      pxx = -td.comp_polariz_inter_xx(omegas).imag
      data = np.array([omegas.real*27.2114, pxx])
      np.savetxt('water.tddft_iter_rpa.omega.inter.pxx.txt', data.T, fmt=['%f','%f'])

      self.assertTrue(np.allclose(data_ref_inter, pxx, rtol=1.0, atol=1e-05))


if __name__ == "__main__": unittest.main()
