#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import numpy
import pyscf.tools
import pyscf.lib.logger as logger
from pyscf.mrpt.nevpt2 import sc_nevpt
from pyscf.dmrgscf.dmrgci import *



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    import pickle
    from pyscf.dmrgscf import settings
    settings.MPIPREFIX ='mpirun -n 3'
    settings.BLOCKEXE = '/home/shengg/blocknewest/block.spin_adapted'
    #BLOCKEXE_MPS_NEVPT is a Block executable file, which is compiled without mpi.
    #It is parallelized by mpi4py rather than in Block code. 
    settings.BLOCKEXE_MPS_NEVPT = '/tigress/shengg/block_mpsnevpt/Block/block.spin_adapted'

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-casscf',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = 'd2h'
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver.nroots=2
    mc.casci()

    file = open('MO','w')
    pickle.dump(mc.mo_coeff,file)
    file.close()
    ci_nevpt_e1 = sc_nevpt(mc,ci=mc.ci[0])
    ci_nevpt_e2 = sc_nevpt(mc,ci=mc.ci[1])

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-dmrg',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = 'd2h'
    )
    m = scf.RHF(mol)
    m.scf()

    file = open('MO','r')
    mo = pickle.load(file)
    file.close()
    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver = DMRGCI(mol,maxM=200)
    mc.fcisolver.nroots = 2
    mc.casci(mo)
    #DMRG-SC-NEVPT2 based on up to fourth order rdm.
    dmrg_nevpt_e1 = sc_nevpt(mc,ci=mc.ci[0])
    dmrg_nevpt_e2 = sc_nevpt(mc,ci=mc.ci[1])



    #Use compressed MPS as perturber functions for sc-nevpt2
    #Fourth order rdm is no longer needed..
    DMRG_MPS_NEVPT(mc,maxM=100, root=0)
    mps_nevpt_e1 = sc_nevpt(mc,ci=mc.ci[0],useMPS=True)
    DMRG_MPS_NEVPT(mc,maxM=100, root=1)
    mps_nevpt_e2 = sc_nevpt(mc,ci=mc.ci[1],useMPS=True)


    print('CI NEVPT = %.15g %.15g DMRG NEVPT  = %.15g %.15g, MPS NEVPT = %.15g %.15g' % (ci_nevpt_e1, ci_nevpt_e2, dmrg_nevpt_e1, dmrg_nevpt_e2, mps_nevpt_e1, mps_nevpt_e2,))

