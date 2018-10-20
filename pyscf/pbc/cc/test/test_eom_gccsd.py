from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc

cell = gto.Cell()
cell.atom='''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = { 'He': [[0, (0.8, 1.0)],
                        [1, (1.0, 1.0)]]}
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

nmp = [1,1,2]

# treating 1*1*1 supercell at gamma point
supcell = super_cell(cell,nmp)
gmf  = scf.GHF(supcell,exxdiv=None)
ehf  = gmf.kernel()
gcc  = cc.GCCSD(gmf)
gcc.conv_tol=1e-12
gcc.conv_tol_normt=1e-10
gcc.max_cycle=250
ecc, t1, t2 = gcc.kernel()
print('GHF energy (supercell) %.7f \n' % (float(ehf)/2.))
print('GCCSD correlation energy (supercell) %.7f \n' % (float(ecc)/2.))


# Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
kmf = scf.KGHF(cell, kpts=cell.make_kpts(nmp), exxdiv=None)
ehf2 = kmf.kernel()

mycc = cc.KGCCSD(kmf)
mycc.conv_tol = 1e-12
mycc.conv_tol_normt = 1e-10
mycc.max_cycle=250
ecc2, t1, t2 = mycc.kernel()
print('GHF energy %.7f \n' % (float(ehf2)))
print('GCCSD correlation energy  %.7f \n' % (float(ecc2)))

print ehf/2 - ehf2
print ecc/2 - ecc2

quit()

eom = EOMIP(mycc)
e, v = eom.ipccsd(nroots=2, kptlist=[0])

eom = EOMEA(mycc)
eom.max_cycle = 100
e, v = eom.eaccsd(nroots=2, koopmans=True, kptlist=[0])
