from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)] ],
    basis = 'cc-pvdz',
)

mf = scf.RHF(mol)
e = mf.scf()
print('E = %.15g, ref -76.0267656731' % e)

# Given four MOs, compute the MO-integrals
import tempfile
import numpy
import h5py
from pyscf import ao2mo
gradtmp = tempfile.NamedTemporaryFile()
nocc = mol.nelectron // 2
nvir = len(mf.mo_energy) - nocc
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
ao2mo.kernel(mol, (co,cv,co,cv), gradtmp.name, intor='cint2e_ip1_sph',
             aosym='s2kl', comp=3, verbose=5)
feri = h5py.File(gradtmp.name, 'r')
grad = numpy.array(feri['eri_mo']).reshape(-1,nocc,nvir,nocc,nvir)
print('shape of gradient integrals %s' % str(grad.shape))

import tempfile
import numpy
import h5py
from pyscf import ao2mo
nocc = mol.nelectron // 2
nvir = len(mf.mo_energy) - nocc
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
hess1tmp = tempfile.NamedTemporaryFile()
ao2mo.kernel(mol, (co,cv,co,cv), hess1tmp.name, intor='cint2e_ipvip1_sph',
             dataname='hessints1', aosym='s4', comp=9, verbose=0)
with ao2mo.load(hess1tmp, 'hessints1') as eri:
    hess1 = numpy.array(eri).reshape(-1,nocc,nvir,nocc,nvir)

hess2tmp = tempfile.NamedTemporaryFile()
ao2mo.general(mol, (co,cv,co,cv), hess2tmp.name, intor='cint2e_ipip1_sph',
              dataname='hessints2', aosym='s2kl', comp=9, verbose=0)
feri = h5py.File(hess2tmp.name, 'r')
hess2 = numpy.array(feri['hessints2']).reshape(-1,nocc,nvir,nocc,nvir)

hess3tmp = tempfile.NamedTemporaryFile()
with ao2mo.load(ao2mo.general(mol, (co,cv,co,cv), hess3tmp.name, intor='cint2e_ip1ip2_sph',
                              aosym='s1', comp=9, verbose=0)) as eri:
    hess3 = numpy.array(eri).reshape(-1,nocc,nvir,nocc,nvir)

print('shape of hessian integrals: hess1 %s, hess2 %s, hess3 %s' %
      (str(hess1.shape), str(hess2.shape), str(hess3.shape)))
