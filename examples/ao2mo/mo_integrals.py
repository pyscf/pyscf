from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.build(
    verbose = 4,
    output = "out_bz",

    atom = [
    ["C", (-0.65830719,  0.61123287, -0.00800148)],
    ["C", ( 0.73685281,  0.61123287, -0.00800148)],
    ["C", ( 1.43439081,  1.81898387, -0.00800148)],
    ["C", ( 0.73673681,  3.02749287, -0.00920048)],
    ["C", (-0.65808819,  3.02741487, -0.00967948)],
    ["C", (-1.35568919,  1.81920887, -0.00868348)],
    ["H", (-1.20806619, -0.34108413, -0.00755148)],
    ["H", ( 1.28636081, -0.34128013, -0.00668648)],
    ["H", ( 2.53407081,  1.81906387, -0.00736748)],
    ["H", ( 1.28693681,  3.97963587, -0.00925948)],
    ["H", (-1.20821019,  3.97969587, -0.01063248)],
    ["H", (-2.45529319,  1.81939187, -0.00886348)],],

    basis = {"H": 'cc-pvtz',
             "C": 'cc-pvtz',},
)

mf = scf.RHF(mol)
mf.conv_threshold = 1e-8
e = mf.scf()
print('E = %.15g, ref -230.776765415' % e)

# Given four MOs, compute the MO-integrals
import tempfile
import numpy
import h5py
from pyscf import ao2mo
_eritmp = tempfile.NamedTemporaryFile()
eritmp = _eritmp.name
nocc = mol.nelectron // 2
nvir = len(mf.mo_energy) - nocc
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
ao2mo.outcore.general(mol, (co,cv,co,cv), eritmp, max_memory=500,
                      dataname='mp2_bz', verbose=5)

eia = mf.mo_energy[:nocc,None] - mf.mo_energy[None,nocc:]
f = h5py.File(eritmp, 'r')
eri = f['mp2_bz']
emp2 = 0
for i in range(nocc):
    dajb = (eia[i].reshape(-1,1) + eia.reshape(1,-1)).ravel()
    gi = numpy.array(eri[i*nvir:(i+1)*nvir]).reshape(nvir,nocc,nvir)
    t2 = (gi.ravel()/dajb).reshape(nvir,nocc,nvir)
    theta = gi.ravel()*2 - gi.transpose(2,1,0).ravel()
    emp2 += numpy.dot(t2.ravel(), theta)

print('E_MP2 = %.15g, ref = -1.0435476768' % emp2)
f.close()
