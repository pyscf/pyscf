#!/Path/To/Python/binary
from __future__ import print_function, division
import os,numpy as np
from pyscf.nao import tddft_iter
import sys
import argparse
from ase.units import Ry, eV, Ha
import subprocess
from timeit import default_timer as timer

def run_tddft_iter():
  t1 = timer()
  td = tddft_iter(label="siesta", iter_broadening=0.0035/Ha, xc_code='LDA,PZ', level=0, tddft_iter_tol=1e-3, tol_loc=1e-4, tol_biloc=1e-6)
  t2 = timer()
  print("time tddft_iter = ", t2-t1)
  omegas = np.arange(2.0, 8.0, 2.5E-3)/Ha + 1j*td.eps
  t3 = timer()
  pxx = -td.comp_polariz_nonin_ave(omegas)
  t4 = timer()
  print("time chi0 = ", t4-t3)

  data = np.zeros((omegas.shape[0], 3))
  data[:, 0] = omegas.real*Ha
  data[:, 1] = pxx.real
  data[:, 2] = pxx.imag
  np.savetxt('polarizability_nonin_siesta.avg.txt', data)

  t5 = timer()
  pxx = -td.comp_polariz_inter_ave(omegas)
  t6 = timer()
  print("time chi = ", t6-t5)
  data = np.zeros((omegas.shape[0], 3))
  data[:, 0] = omegas.real*Ha
  data[:, 1] = pxx.real
  data[:, 2] = pxx.imag
  np.savetxt('polarizability_inter_siesta.avg.txt', data)
  print("nb call:")
  print("rf0_ncalls = {0}, matvec_ncalls ={1}".format(td.rf0_ncalls, td.matvec_ncalls))
  t7 = timer()
  print("total time = ", t7-t1)

parser = argparse.ArgumentParser()
parser.add_argument('--np', type=int, default=1, help="number of processor to use")
parser.add_argument('--start', type=int, default=0, help="starting calc")
parser.add_argument('--end', type=int, default=25, help="end calculation")

args_par = parser.parse_args()


xyz_range = np.arange(args_par.start, args_par.end, 25)
mpi_exe="/scratch/mbarbry/intel/intelpython2/bin/mpirun"
siesta_path="/scratch/software/SIESTA/4.0b-485-intel-2015b/siesta"

siesta_exe = mpi_exe + " -np {0} ".format(args_par.np) + siesta_path + " < siesta.fdf > siesta.out"
for i, xyz in enumerate(xyz_range):
    if xyz < 10:
        num = "00000{0}".format(xyz)
    elif xyz < 100:
        num = "0000{0}".format(xyz)
    elif xyz < 1000:
        num = "000{0}".format(xyz)
    elif xyz < 10000:
        num = "00{0}".format(xyz)
    else:
        raise ValueError("xyz too large?? {0}".format(xyz))
    path = "calc_" + num
    os.chdir(path)
    # Run siesta
    subprocess.call("export OMP_NUM_THREADS=1", shell=True)
    subprocess.call(siesta_exe, shell=True)

    # Run pyscf.nao
    subprocess.call("export OMP_NUM_THREADS={0}".format(args_par.np), shell=True)
    run_tddft_iter()
    os.chdir("../")
