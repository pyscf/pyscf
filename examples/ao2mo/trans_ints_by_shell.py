import numpy
from pyscf import gto
from pyscf import scf
from pyscf.ao2mo import _ao2mo

# transform integrals for certain AO shells

mol = gto.Mole()
mol.build(
    verbose = 0,

    atom = '''
      C    -0.65830719   0.61123287  -0.00800148
      C     0.73685281   0.61123287  -0.00800148
      C     1.43439081   1.81898387  -0.00800148
      C     0.73673681   3.02749287  -0.00920048
      C    -0.65808819   3.02741487  -0.00967948
      C    -1.35568919   1.81920887  -0.00868348
      H    -1.20806619  -0.34108413  -0.00755148
      H     1.28636081  -0.34128013  -0.00668648
      H     2.53407081   1.81906387  -0.00736748
      H     1.28693681   3.97963587  -0.00925948
      H    -1.20821019   3.97969587  -0.01063248
      H    -2.45529319   1.81939187  -0.00886348''',

    basis = 'cc-pvdz',
)

mf = scf.RHF(mol)
mf.conv_tol = 1e-8
e = mf.scf()
print('E = %.15g, ref -230.776765415' % e)

nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron // 2
nvir = nmo - nocc

# half transformed ERIs (ij|kl) for i \in occupied MOs, j \in virtual MOs, k, l \in AOs
half1_eri = numpy.zeros((nocc,nvir,nao,nao))
mo = mf.mo_coeff

k0 = 0
for ksh in range(mol.nbas):
    # * First half integral transfromation for 'cint2e_sph'
    # * The third parameter is the shape of the MOs used for transformation.
    #   (start_id_for_i, num_of_mos_for_i, start_id_for_j, num_of_mos_for_j)
    #   In this example, it takes mo[:,0:nocc] for index (i|, mo[:,nocc:nmo]
    #   for index (j|
    # * The fourth parameter (ksh,ksh+1) is the range of AO shells of
    #   untransformed index (k|
    # * aosym assumes the permutation symmetry in (ij|, to gain efficiency in the
    #   transformation more efficient.  For clearity, no symmetry is used for |kl).
    #   The |kl) symmetry can be used by setting aosym='s4'
    buf = _ao2mo.nr_e1_('cint2e_sph', mo, (0,nocc,nocc,nvir), (ksh,ksh+1),
                        mol._atm, mol._bas, mol._env, aosym='s2ij')
    buf = buf.reshape(-1,nao,nocc,nvir).transpose(2,3,0,1)
    k1 = k0 + buf.shape[2]
    half1_eri[:,:,k0:k1,:] = buf
    k0 = k1

# second half transfromation
eri_ovov = numpy.zeros((nocc,nvir,nocc,nvir))
for i in range(nocc):
    buf = _ao2mo.nr_e2_(half1_eri[i], mo, (0,nocc,nocc,nvir))
    eri_ovov[i] = buf.reshape(nvir,nocc,nvir)

eia = mf.mo_energy[:nocc,None] - mf.mo_energy[None,nocc:]
emp2 = 0
for i in range(nocc):
    dajb = (eia[i].reshape(-1,1) + eia.reshape(1,-1)).ravel()
    gi = eri_ovov[i]
    t2 = (gi.ravel()/dajb).reshape(nvir,nocc,nvir)
    theta = gi.ravel()*2 - gi.transpose(2,1,0).ravel()
    emp2 += numpy.dot(t2.ravel(), theta)

print('E_MP2 = %.15g, ref = -0.800036532' % emp2)

