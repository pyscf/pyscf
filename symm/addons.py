#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import pyscf.lib.logger

def label_orb_symm(mol, irrep_name, symm_orb, mo):
    nmo = mo.shape[1]
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    mo_s = numpy.dot(mo.T, s)
    orbsym = [None] * nmo
    for i,ir in enumerate(irrep_name):
        moso = numpy.dot(mo_s, symm_orb[i])
        for j in range(nmo):
            if not numpy.allclose(moso[j], 0, atol=1e-6):
                if orbsym[j] is None:
                    orbsym[j] = ir
                else:
                    raise ValueError('orbital %d not symmetrized' % j)
    pyscf.lib.logger.debug(mol, 'irreps of each MO %s', str(orbsym))
    return orbsym

def symmetrize_orb(mol, irrep_name, symm_orb, mo):
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    mo_s = numpy.dot(mo.T, s)
    mo1 = 0
    for csym in symm_orb:
        ovlpso = reduce(numpy.dot, (csym.T, s, csym))
        sc = numpy.linalg.solve(ovlpso, numpy.dot(mo_s, csym).T)
        mo1 = mo1 + numpy.dot(csym, sc)
    return mo1

if __name__ == "__main__":
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.build(
        atom = [['H', (0,0,0)], ['H', (0,0,1)]],
        basis = {'H': 'cc-pvdz'},
        symmetry = 1
    )
    mf = scf.RHF(mol)
    mf.scf()

    print label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
