#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import pyscf.lib.logger

def label_orb_symm(mol, irrep_name, symm_orb, mo, s=None):
    ''' Label the symmetry of given orbitals

    irrep_name can be either the symbol or the ID of the irreducible
    representation.  If the ID is provided, it returns the numeric code
    associated with XOR operator, see :py:meth:`symm.param.IRREP_ID_TABLE`

    Args:
        mol : an instance of :class:`Mole`

        irrep_name : list of str or int
            A list of irrep ID or name,  it can be either mol.irrep_id or
            mol.irrep_name.  It can affect the return "label".
        symm_orb : list of 2d array
            the symmetry adapted basis
        mo : 2d array
            the orbitals to label

    Returns:
        list of symbols or integers to represent the irreps for the given
        orbitals

    Examples:

    >>> from pyscf import gto, scf, symm
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz',verbose=0, symmetry=1)
    >>> mf = scf.RHF(mol)
    >>> mf.kernel()
    >>> symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    ['Ag', 'B1u', 'Ag', 'B1u', 'B2u', 'B3u', 'Ag', 'B2g', 'B3g', 'B1u']
    >>> symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff)
    [0, 5, 0, 5, 6, 7, 0, 2, 3, 5]
    '''
    nmo = mo.shape[1]
    if s is None:
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

def symmetrize_orb(mol, symm_orb, mo):
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    mo_s = numpy.dot(mo.T, s)
    mo1 = 0
    for csym in symm_orb:
        ovlpso = reduce(numpy.dot, (csym.T, s, csym))
        sc = numpy.linalg.solve(ovlpso, numpy.dot(mo_s, csym).T)
        mo1 = mo1 + numpy.dot(csym, sc)
    return mo1

def std_symb(gpname):
    '''std_symb('d2h') returns D2h; std_symb('D2H') returns D2h'''
    return gpname[0].upper() + gpname[1:].lower()


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

    print(label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff))
