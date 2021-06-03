#!/usr/bin/env python


'''
PiOS example according to
"Constructing molecular pi-orbital active spaces for multireference calculations of conjugated systems"
E. R. Sayfutyarova and S. Hammes-Schiffer, J. Chem. Theory Comput., 15, 1679 (2019).
'''

import numpy
from pyscf import gto
from pyscf.gto import mole
from pyscf.gto import moleintor
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from functools import reduce
from pyscf import fci
from pyscf.tools import fcidump
from pyscf import mrpt
from pyscf.mcscf.PiOS import MakePiOS


mol=gto.Mole()
mol.atom='''
C  -0.205041040016   4.293235977701   0.000000000000
C   0.403262807645   3.101311953997  -0.000000000000
C  -0.282048458140   1.831603049166   0.000000000000
C   0.338470548365   0.633304932323   0.000000000000
C  -0.338470548365  -0.633304932323   0.000000000000
C   0.282048458140  -1.831603049166   0.000000000000
C  -0.403262807645  -3.101311953997   0.000000000000
C   0.205041040016  -4.293235977701   0.000000000000
H   0.357619144805   5.215664988374   0.000000000000
H  -1.284762686441   4.375962764712   0.000000000000
H   1.488448703032   3.064020960940   0.000000000000
H  -1.367965273026   1.860302168845   0.000000000000
H   1.424632376981   0.609318533944   0.000000000000
H  -1.424632376981  -0.609318533944   0.000000000000
H   1.367965273026  -1.860302168845   0.000000000000
H  -1.488448703032  -3.064020960940   0.000000000000
H   1.284762686441  -4.375962764712   0.000000000000
H  -0.357619144805  -5.215664988374   0.000000000000
'''
mol.basis = 'aug-cc-pvtz'
mol.verbose=5
mol.spin =0
mol.build()

# Rotate the molecule so that it's not parallel to xy-plane
numpy.random.seed(1)
u = numpy.linalg.svd(numpy.random.random((3,3)))[0]
mol.set_geom_(mol.atom_coords().dot(u), unit='Bohr')


mf=scf.RHF(mol)
mf.kernel()


PiAtoms = [1,2,3,4,5,6,7,8]  #list atom numbers for your pi-system, counting from 1
N_Core,N_Actorb, N_Virt,nelec,coeff=MakePiOS(mol,mf,PiAtoms)  
#if you don't want the entire pi-space, use MakePiOS(mol,mf,PiAtomsList, nPiOcc,nPiVirt), where nPiOcc and nPiVirt determine how many HOMOs and LUMOs should be picked up
nalpha=(nelec+mol.spin)//2
nbeta=(nelec-mol.spin)//2

#=================================run CASSCF
mycas = mcscf.CASSCF(mf, N_Actorb, [nalpha,nbeta])
AS=range(N_Core,N_Core+N_Actorb)
mycas =mycas.state_average_([0.2, 0.2, 0.2,0.2,0.2])
mycas.chkfile ='cas_c8h10.chk'
mycas.fcisolver.nroots = 5
mycas.fix_spin_(ss=0)
activeMO = mcscf.sort_mo(mycas,coeff,AS,base=0)
mycas.verbose = 5
mycas.max_cycle_macro = 150
mycas.kernel(activeMO)


#==================================run CASCI followed by NEVPT2 with CASSCF orbitals
mycas = mcscf.CASCI(mf, N_Actorb, [nalpha,nbeta])
mycas.__dict__.update(scf.chkfile.load('cas_c8h10.chk', 'mcscf'))
mycas.fcisolver.nroots = 5
mycas.fix_spin_(ss=0)
mycas.verbose = 5
mycas.kernel()

ci_nevpt_e1 = mrpt.NEVPT(mycas, root=0).kernel()
ci_nevpt_e2 = mrpt.NEVPT(mycas, root=1).kernel()
ci_nevpt_e3 = mrpt.NEVPT(mycas, root=2).kernel()
ci_nevpt_e4 = mrpt.NEVPT(mycas, root=3).kernel()
ci_nevpt_e5 = mrpt.NEVPT(mycas, root=4).kernel()




