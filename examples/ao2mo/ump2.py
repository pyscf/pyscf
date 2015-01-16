from pyscf import gto
from pyscf import scf
mol = gto.Mole()
mol.verbose = 0
mol.atom = 'O 0 0 0; O 0 0 1.2'
mol.basis = 'ccpvdz'
mol.spin = 2
mol.build()

m = scf.UHF(mol)
print(m.scf())

import numpy
def myump2(mol, mo_energy, mo_occ, mo_coeff):
    from pyscf import ao2mo
    o = numpy.hstack((mo_coeff[0][:,mo_occ[0]>0] ,mo_coeff[1][:,mo_occ[1]>0]))
    v = numpy.hstack((mo_coeff[0][:,mo_occ[0]==0],mo_coeff[1][:,mo_occ[1]==0]))
    eo = numpy.hstack((mo_energy[0][mo_occ[0]>0] ,mo_energy[1][mo_occ[1]>0]))
    ev = numpy.hstack((mo_energy[0][mo_occ[0]==0],mo_energy[1][mo_occ[1]==0]))
    no = o.shape[1]
    nv = v.shape[1]
    noa = sum(mo_occ[0]>0)
    nva = sum(mo_occ[0]==0)
    eri = ao2mo.outcore.general_iofree(mol, (o,v,o,v)).reshape(no,nv,no,nv)
    eri[:noa,nva:] = eri[noa:,:nva] = eri[:,:,:noa,nva:] = eri[:,:,noa:,:nva] = 0
    g = eri - eri.transpose(0,3,2,1)
    eov = eo.reshape(-1,1) - ev.reshape(-1)
    de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(g.shape)
    emp2 = .25 * numpy.einsum('iajb,iajb,iajb->', g, g, de)
    return emp2

e = myump2(mol, m.mo_energy, m.mo_occ, m.mo_coeff)
print('E(UMP2) = %.9g, ref = -0.346926068' % e)

