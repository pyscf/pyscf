import sys
import numpy as np
import ase
import ase.dft.kpoints
import os.path
from pyscf.pbc import cc as pbccc
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools as pbctools
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf.pbc.tools.pbc import super_cell
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from pyscf.pbc.examples.helpers import build_cell
from pyscf.pbc.examples.scf import run_hf, run_khf
from pyscf.pbc.examples.cc import run_rccsd
from pyscf.pbc.mpicc import KRCCSD

from ase.lattice import bulk
a = 3.5668
A2B = 1.889725989
ase_atom = bulk('C', 'diamond', a=a*A2B)
basis = 'gth-szv'
nmp = [1,1,3]

cell = build_cell(ase_atom, ke=50., basis=basis)
# Increasing rcut so the kpoint and gamma calculations will agree to
# higher precision
cell.rcut *= 1.5
supcell = super_cell(cell, nmp)
mf = run_hf(supcell, exxdiv=None)#, conv_tol=1e-10)
supcell_energy = mf.energy_tot() / np.prod(nmp)
cc = run_rccsd(mf)
ecc = cc.ecc / np.prod(nmp)
eea, wea = cc.eaccsd(nroots=2)
elea, wlea = cc.eaccsd(nroots=2,left=True)
eacc_star = cc.eaccsd_star(eea,wea,wlea)

kmf = run_khf(cell, nmp=nmp, gamma=True, exxdiv=None)#, conv_tol=1e-10)
kpoint_energy = kmf.energy_tot()
kcc = KRCCSD(kmf)
kcc.ccsd()
ekcc = kcc.ecc
ekea, wkea = kcc.eaccsd(nroots=2,kptlist=[0])
elkea, wlkea = kcc.leaccsd(nroots=2,kptlist=[0])
eakcc_star = kcc.eaccsd_star(ekea,wkea,wlkea)
print 'cc star ', eacc_star
print 'kcc star ', eakcc_star
print "E_{k_point} - E_{gamma_sup} = %16.10e" % (kpoint_energy - supcell_energy)
print "ECC_{k_point} - ECC_{gamma_sup} = %16.10e" % (ekcc - ecc)
print "eaSTAR_{k_point} - eaSTAR_{gamma_sup} = %16.10e" % (np.linalg.norm(eacc_star - eakcc_star))
