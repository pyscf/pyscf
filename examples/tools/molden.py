import sys
from pyscf import gto, scf, tools
from pyscf import lo
from pyscf.tools import molden

mol = gto.M(
    atom = '''
  C  3.2883    3.3891    0.2345       
  C  1.9047    3.5333    0.2237       
  C  3.8560    2.1213    0.1612       
  C  1.0888    2.4099    0.1396       
  C  3.0401    0.9977    0.0771       
  C  1.6565    1.1421    0.0663       
  H  3.9303    4.2734    0.3007       
  H  1.4582    4.5312    0.2815       
  H  4.9448    2.0077    0.1699       
  H  0.0000    2.5234    0.1311       
  H  3.4870    0.0000    0.0197       
  H  1.0145    0.2578    0.0000       
           ''',
    basis = 'cc-pvdz',
    symmetry = 1)

mf = scf.RHF(mol)
mf.scf()

with open('C6H6mo.molden', 'w') as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)

c_loc_orth = lo.orth.orth_ao(mol, 'meta_lowdin', lo.orth.pre_orth_ao(mol))
with open('C6H6loc.molden', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, c_loc_orth)



