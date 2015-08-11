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
    mc.fcisolver = DMRGCI(mol)
    mc.fcisolver.nroots = 2
    mc.casci(mo)
    dmrg_nevpt_e1 = sc_nevpt(mc,ci=mc.ci[0])
    dmrg_nevpt_e2 = sc_nevpt(mc,ci=mc.ci[1])



    #Use MPS perturber for nevpt

    #MPS perturber nevpt code has not been merged with master branch of Block code. 
    #And it is parallelized by mpi4py rather than in Block code. 
    settings.BLOCKEXE = '/home/shengg/block_nevpt2/block/block.spin_adapted'

    DMRG_MPS_NEVPT(mc,root=0)
    mps_nevpt_e1 = sc_nevpt(mc,ci=mc.ci[0],useMPS=True)
    DMRG_MPS_NEVPT(mc,root=1)
    mps_nevpt_e2 = sc_nevpt(mc,ci=mc.ci[1],useMPS=True)


    print('CI NEVPT = %.15g %.15g DMRG NEVPT  = %.15g %.15g, MPS NEVPT = %.15g %.15g' % (ci_nevpt_e1, ci_nevpt_e2, dmrg_nevpt_e1, dmrg_nevpt_e2, mps_nevpt_e1, mps_nevpt_e2,))

