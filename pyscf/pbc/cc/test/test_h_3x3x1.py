import unittest
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import scf,cc
from pyscf     import cc as mol_cc
from pyscf.pbc.tools.pbc import super_cell
a0  = 4
vac = 200
bas = [[ 3.0/2.0*a0,np.sqrt(3.0)/2.0*a0,  0],
       [-3.0/2.0*a0,np.sqrt(3.0)/2.0*a0,  0],
       [          0,                  0,vac]]
pos = [['H',(-a0/2.0,0,0)],
       ['H',( a0/2.0,0,0)]]

cell = gto.M(unit='B',a=bas,atom=pos,basis='cc-pvdz',verbose=4)
nmp  = [3,3,1]
nk   = np.prod(nmp)
nao  = cell.nao_nr()


#primitive cell with k points
kpts = cell.make_kpts(nmp)
nkpts = len(kpts)
kmf  = scf.KUHF(cell,kpts,exxdiv=None).density_fit()
#kmf.chkfile = 'kpt.chk'
nao_half = nao//2
dmk = np.zeros([2, nkpts, nao, nao])
for i in range(nkpts):
    for j in range(2):
        dmk[0][i][j,j] = 0.5
        dmk[1][i][j+nao_half, j+nao_half] = 0.5

ehf = kmf.kernel(dmk)

kcc = cc.KUCCSD(kmf)
ecc,t1,t2 = kcc.kernel()

print('========================================')
print('UHF energy (kpts) %f \n' % (float(ehf)))
print('UCCSD correlation energy (kpts) %f \n' % (float(ecc)))
print('========================================')

# Gamma point supercell calculation
supcell = super_cell(cell,nmp)

dms = np.zeros([2, supcell.nao_nr(), supcell.nao_nr()])
for i in range(nkpts):
    for j in range(2):
        dms[0][j+i*nao][j+i*nao] = 0.5
        dms[1][j+i*nao+nao_half][j+i*nao+nao_half] = 0.5

gmf = scf.UHF(supcell,exxdiv=None).density_fit()
#gmf.chkfile = 'supcell.chk'
ehf = gmf.kernel(dms)

gcc = cc.UCCSD(gmf)
ecc,t1,t2 = gcc.kernel()
print('========================================')
print('UHF energy (supercell) %f' % (float(ehf)/nk))
print('UCCSD correlation energy (supercell) %f' % (float(ecc)/nk))
print('========================================')
