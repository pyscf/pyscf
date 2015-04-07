import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf

mol = gto.M(
    verbose = 5,
    output = 'C2-1.5.out',
    atom = [
      ['C' , (0. , 0. , 0.75)],
      ['C' , (0. , 0. ,-0.75)],
    ],
    basis = 'cc-pVDZ',
    symmetry = 1,
)

myhf = scf.RHF(mol)
#myhf.irrep_nelec = {'Ag': 4, 'B1u': 4, 'B2u': 2, 'B3u': 2,}
hf_energy = myhf.scf()
print('SCF E=%.15g' % hf_energy)

cas = mcscf.CASSCF(myhf, 16, (4,4))
#def save_mo_coeff(mo_coeff, imacro, imicro):
#    label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
#    dump_mat.dump_rec(mol.stdout, mo_coeff, label, start=1, digits=9)
#cas.save_mo_coeff = save_mo_coeff
e_cas = cas.mc1step()[0]
cas.analyze()
print('CASSCF E = %.15g, ref = -75.6299729925424' % e_cas)

