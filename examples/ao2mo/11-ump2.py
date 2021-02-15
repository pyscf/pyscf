#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import numpy
import h5py
from pyscf import gto, scf, ao2mo

'''
UHF-MP2
'''

mol = gto.M(
    verbose = 0,
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

m = scf.UHF(mol)
print(m.scf())

def myump2(mol, mo_energy, mo_coeff, mo_occ):
    o = numpy.hstack((mo_coeff[0][:,mo_occ[0]>0] ,mo_coeff[1][:,mo_occ[1]>0]))
    v = numpy.hstack((mo_coeff[0][:,mo_occ[0]==0],mo_coeff[1][:,mo_occ[1]==0]))
    eo = numpy.hstack((mo_energy[0][mo_occ[0]>0] ,mo_energy[1][mo_occ[1]>0]))
    ev = numpy.hstack((mo_energy[0][mo_occ[0]==0],mo_energy[1][mo_occ[1]==0]))
    no = o.shape[1]
    nv = v.shape[1]
    noa = sum(mo_occ[0]>0)
    nva = sum(mo_occ[0]==0)
    eri = ao2mo.general(mol, (o,v,o,v)).reshape(no,nv,no,nv)
    eri[:noa,nva:] = eri[noa:,:nva] = eri[:,:,:noa,nva:] = eri[:,:,noa:,:nva] = 0
    g = eri - eri.transpose(0,3,2,1)
    eov = eo.reshape(-1,1) - ev.reshape(-1)
    de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(g.shape)
    emp2 = .25 * numpy.einsum('iajb,iajb,iajb->', g, g, de)
    return emp2

e = myump2(mol, m.mo_energy, m.mo_coeff, m.mo_occ)
print('E(UMP2) = %.9g, ref = -0.346926068' % e)

