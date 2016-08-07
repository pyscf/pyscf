#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.symm import basis
from pyscf.symm import param

THRESHOLD = 1e-9

def label_orb_symm(mol, irrep_name, symm_orb, mo, s=None, check=True):
    '''Label the symmetry of given orbitals

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
    norm = numpy.zeros((len(irrep_name), nmo))
    for i,ir in enumerate(irrep_name):
        moso = numpy.dot(mo_s, symm_orb[i])
        ovlpso = reduce(numpy.dot, (moso.T, s, moso))
        norm[i] = numpy.einsum('ik,ki->i', moso, lib.cho_solve(ovlpso, moso.T))
    iridx = numpy.argmax(norm, axis=0)
    orbsym = [irrep_name[i] for i in iridx]
    logger.debug(mol, 'irreps of each MO %s', str(orbsym))
    if check:
        largest_norm = norm[iridx,numpy.arange(nmo)]
        orbidx = numpy.where(largest_norm < 1-THRESHOLD)[0]
        if orbidx.size > 0:
            idx = numpy.where(largest_norm < 1-THRESHOLD*1e2)[0]
            if idx.size > 0:
                logger.error(mol, 'orbitals %s not symmetrized, norm = %s',
                             idx[1], norm[idx])
                raise ValueError('orbitals %s not symmetrized' %
                                 numpy.unique(idx[1]))
            else:
                logger.warn(mol, 'orbitals %s not strictly symmetrized.',
                            numpy.unique(orbidx[1]))
                logger.warn(mol, 'They can be symmetrized with '
                            'pyscf.symm.symmetrize_orb function.')
                logger.debug(mol, 'norm = %s', norm[orbidx])
    return orbsym

def symmetrize_orb(mol, mo, orbsym=None, s=None):
    '''Symmetrize the given orbitals.

    This function is different to the :func:`symmetrize_space`:  In this
    function, each orbital is symmetrized by removing non-symmetric components.
    :func:`symmetrize_space` symmetrizes the entire space by mixing different
    orbitals.

    Note this function might return non-orthorgonal orbitals.
    Call :func:`symmetrize_space` to find the symmetrized orbitals that are
    close to the given orbitals.

    Args:
        mo : 2D float array
            The orbital space to symmetrize

    Kwargs:
        orbsym : integer list
            Irrep id for each orbital.  If not given, the irreps are guessed
            by calling :func:`label_orb_symm`.
        s : 2D float array
            Overlap matrix.  If given, use this overlap than the the overlap
            of the input mol.

    Returns:
        2D orbital coefficients

    Examples:

    >>> from pyscf import gto, symm, scf
    >>> mol = gto.M(atom = 'C  0  0  0; H  1  1  1; H -1 -1  1; H  1 -1 -1; H -1  1 -1',
    ...             basis = 'sto3g')
    >>> mf = scf.RHF(mol).run()
    >>> mol.build(0, 0, symmetry='D2')
    >>> mo = symm.symmetrize_orb(mol, mf.mo_coeff)
    >>> print(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo))
    ['A', 'A', 'B1', 'B2', 'B3', 'A', 'B1', 'B2', 'B3']
    '''
    if s is None:
        s = mol.intor_symmetric('cint1e_ovlp_sph')
    if orbsym is None:
        orbsym = label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                mo, s=s, check=False)
    orbsym = numpy.asarray(orbsym)
    mo_s = numpy.dot(mo.T, s)
    mo1 = numpy.empty_like(mo)

    if orbsym[0] in mol.irrep_name:
        irrep_id = mol.irrep_name
    else:
        irrep_id = mol.irrep_id

    for i, ir in enumerate(irrep_id):
        idx = orbsym == ir
        csym = mol.symm_orb[i]
        ovlpso = reduce(numpy.dot, (csym.T, s, csym))
        sc = lib.cho_solve(ovlpso, numpy.dot(mo_s[idx], csym).T)
        mo1[:,idx] = numpy.dot(csym, sc)
    return mo1

def symmetrize_space(mol, mo, s=None):
    '''Symmetrize the given orbital space.

    This function is different to the :func:`symmetrize_orb`:  In this function,
    the given orbitals are mixed to reveal the symmtery; :func:`symmetrize_orb`
    projects out non-symmetric components for each orbital.

    Args:
        mo : 2D float array
            The orbital space to symmetrize

    Kwargs:
        s : 2D float array
            Overlap matrix.  If not given, overlap is computed with the input mol.

    Returns:
        2D orbital coefficients

    Examples:

    >>> from pyscf import gto, symm, scf
    >>> mol = gto.M(atom = 'C  0  0  0; H  1  1  1; H -1 -1  1; H  1 -1 -1; H -1  1 -1',
    ...             basis = 'sto3g')
    >>> mf = scf.RHF(mol).run()
    >>> mol.build(0, 0, symmetry='D2')
    >>> mo = symm.symmetrize_space(mol, mf.mo_coeff)
    >>> print(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo))
    ['A', 'A', 'A', 'B1', 'B1', 'B2', 'B2', 'B3', 'B3']
    '''
    from pyscf.tools import mo_mapping
    if s is None:
        s = mol.intor_symmetric('cint1e_ovlp_sph')
    nmo = mo.shape[1]
    mo_s = numpy.dot(mo.T, s)
    mo1 = []
    for i, ir in enumerate(mol.irrep_id):
        csym = mol.symm_orb[i]
        moso = numpy.dot(mo_s, csym)
        ovlpso = reduce(numpy.dot, (csym.T, s, csym))

# excluding orbitals which are already symmetrized
        diag = numpy.einsum('ik,ki->i', moso, lib.cho_solve(ovlpso, moso.T))
        idx = abs(1-diag) < 1e-8
        orb_exclude = mo[:,idx]
        mo1.append(orb_exclude)
        moso1 = moso[~idx]
        dm = numpy.dot(moso1.T, moso1)

        if dm.trace() > 1e-8:
            e, u = scipy.linalg.eigh(dm, ovlpso)
            mo1.append(numpy.dot(csym, u[:,abs(1-e) < 1e-6]))
    mo1 = numpy.hstack(mo1)
    if mo1.shape[1] != nmo:
        raise ValueError('The input orbital space is not symmetrized.\n It is '
                         'probably because the input mol and orbitals are of '
                         'different orientation.')
    snorm = numpy.linalg.norm(reduce(numpy.dot, (mo1.T, s, mo1)) - numpy.eye(nmo))
    if snorm > 1e-6:
        raise ValueError('Orbitals are not orthogonalized')
    idx = mo_mapping.mo_1to1map(reduce(numpy.dot, (mo.T, s, mo1)))
    return mo1[:,idx]

def std_symb(gpname):
    '''std_symb('d2h') returns D2h; std_symb('D2H') returns D2h'''
    return gpname[0].upper() + gpname[1:].lower()

def irrep_name2id(gpname, symb):
    '''Convert the irrep symbol to internal irrep ID

    Args:
        gpname : str
            The point group symbol
        symb : str
            Irrep symbol

    Returns:
        Irrep ID, int
    '''
    gpname = std_symb(gpname)
    symb = std_symb(symb)
    if gpname in ('Dooh', 'Coov'):
        return basis.linearmole_irrep_symb2id(gpname, symb)
    else:
        return param.IRREP_ID_TABLE[gpname][symb]

def irrep_id2name(gpname, irrep_id):
    '''Convert the internal irrep ID to irrep symbol

    Args:
        gpname : str
            The point group symbol
        irrep_id : int
            See IRREP_ID_TABLE in pyscf/symm/param.py

    Returns:
        Irrep sybmol, str
    '''
    gpname = std_symb(gpname)
    if gpname in ('Dooh', 'Coov'):
        return basis.linearmole_irrep_id2symb(gpname, irrep_id)
    else:
        return param.CHARACTER_TABLE[gpname][irrep_id][0]

def irrep_name(pgname, irrep_id):
    raise RuntimeError('This function was obsoleted. Use irrep_id2name')

def route(target, nelec, orbsym):
    '''Pick orbitals to form a determinant which has the right symmetry.
    If solution is not found, return []
    '''
    def riter(target, nelec, orbsym):
        if nelec == 1:
            if target in orbsym:
                return [orbsym.index(target)]
            else:
                return []
        else:
            for i, ir in enumerate(orbsym):
                off = i + 1
                orb_left = orbsym[off:]
                res = riter(target^ir, nelec-1, orb_left)
                if res:
                    return [i] + [off+x for x in res]
            return []
    if isinstance(orbsym, numpy.ndarray):
        orbsym = orbsym.tolist()
    return riter(target, nelec, orbsym)

def eigh(h, orbsym):
    '''Solve eigenvalue problem based on the symmetry information for basis.
    See also pyscf/lib/linalg_helper.py :func:`eigh_by_blocks`

    Examples:

    >>> from pyscf import gto, symm
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz', symmetry=True)
    >>> c = numpy.hstack(mol.symm_orb)
    >>> vnuc_so = reduce(numpy.dot, (c.T, mol.intor('cint1e_nuc_sph'), c))
    >>> orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)
    >>> symm.eigh(vnuc_so, orbsym)
    (array([-4.50766885, -1.80666351, -1.7808565 , -1.7808565 , -1.74189134,
            -0.98998583, -0.98998583, -0.40322226, -0.30242374, -0.07608981]),
     ...)
    '''
    return lib.eigh_by_blocks(h, labels=orbsym)

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

    nao, nmo = mf.mo_coeff.shape
    print(label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff))
    numpy.random.seed(1)
    u = numpy.random.random((nmo,nmo))*1e-2
    u = scipy.linalg.expm(u - u.T)
    mo = symmetrize_orb(mol, numpy.dot(mf.mo_coeff, u))
    print(label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo))

    orbsym = [0, 3, 0, 2, 5, 6]
    res = route(7, 3, orbsym)
    print(res, reduce(lambda x,y:x^y, [orbsym[i] for i in res]))
