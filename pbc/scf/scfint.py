'''
SCF (Hartree-Fock and DFT) tools for periodic systems at a *single* k-point,
    using analytical GTO integrals instead of PWs.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.

'''

import numpy as np
import pyscf.scf
import pyscf.scf.hf
import pyscf.dft
import pyscf.gto
import pyscf.lib
import pyscf.pbc.scf
from pyscf.pbc.gto import pseudo
from pyscf.pbc import tools


def get_hcore(cell, kpt=None):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.'''
    if kpt is None:
        kpt = np.zeros(3)

    # TODO: these are still on grid
    if cell.pseudo is None:
        hcore = pyscf.pbc.scf.hf.get_nuc(cell, kpt)
    else:
        hcore = (pyscf.pbc.scf.hf.get_pp(cell, kpt) +
                 get_jvloc_G0(cell, kpt))
    hcore += get_t(cell, kpt)

    # hcore = get_t(cell, kpt)
    return hcore

def get_jvloc_G0(cell, kpt=None):
    '''Get the (separately) divergent Hartree + Vloc G=0 contribution.'''

    return 1./cell.vol * np.sum(pseudo.get_alphas(cell)) * get_ovlp(cell, kpt)

def get_int1e_cross(intor, cell1, cell2, kpt=None, comp=1):
    r'''1-electron integrals from two molecules like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    nimgs = np.max((cell1.nimgs, cell2.nimgs), axis=0)
    Ls = tools.get_lattice_Ls(cell1, nimgs)
# Change the basis position only, keep all other envrionments
    cellL = cell2.copy()
    ptr_coord = cellL._atm[:,pyscf.gto.PTR_COORD]
    _envL = cellL._env
    int1e = 0
    for L in Ls:
        _envL[ptr_coord+0] = cell2._env[ptr_coord+0] + L[0]
        _envL[ptr_coord+1] = cell2._env[ptr_coord+1] + L[1]
        _envL[ptr_coord+2] = cell2._env[ptr_coord+2] + L[2]
        if kpt is None or np.all(kpt == 0):
            int1e += pyscf.gto.intor_cross(intor, cell1, cellL, comp)
        else:
            factor = np.exp(1j*np.dot(kpt, L))
            int1e += pyscf.gto.intor_cross(intor, cell1, cellL, comp) * factor
    return int1e

def get_int1e(intor, cell, kpt=None):
    '''Get the one-electron integral defined by `intor` using lattice sums.'''
    return get_int1e_cross(intor, cell, cell, kpt, 1)

def get_ovlp(cell, kpt=None):
    '''Get the overlap AO matrix.'''
    s = get_int1e('cint1e_ovlp_sph', cell, kpt)
    return s

def get_t(cell, kpt=None):
    '''Get the kinetic energy AO matrix.'''
    t = get_int1e('cint1e_kin_sph', cell, kpt)
    return t

def test_periodic_ints():
    from pyscf import gto
    import pyscf.pbc.gto as pgto
    import scf

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    Lunit = 3
    Ly = Lz = Lunit
    Lx = Lunit

    h = np.diag([Lx,Ly,Lz])

    mol.build(
        verbose = 0,
        atom = '''
            H     0    0.       0.
            H     1    0.       0.
        ''',
        basis={'H':'sto-3g'})
    #    basis={'H':[[0,(1.0,1.0)]]})

    # these are some exponents which are
    # not hard to integrate
    # mol.basis = { 'He': [[0, (1.0, 1.0)], [0, [2.0, 1.0]]] }
    #mol.basis = { 'H': [[0, (1.0, 1.0)], [0, [2.0, 1.0]]] }
    #mol.unit='A'
    #mol.build()

    cell = pgto.Cell()
    cell.__dict__ = mol.__dict__ # hacky way to make a cell
    cell.h = h
    cell.nimgs = cell.get_nimgs(1.e-6)
    # print "NIMG",
    print "NIMGS", cell.nimgs
    cell.pseudo = None
    cell.output = None
    cell.verbose = 0
    cell.build()

    gs = np.array([20,20,20])

    # Analytic
    sA=get_ovlp(cell)
    tA=get_t(cell)

    # Grid
    sG=pyscf.pbc.scf.hf.get_ovlp(cell, gs)
    tG=pyscf.pbc.scf.hf.get_t(cell, gs)

    # These differences should be 0 up to grid integration error
    print "Diff", np.linalg.norm(sA-sG) # 1.05796568891e-06
    print "Diff", np.linalg.norm(tA-tG) # 4.82330435721e-06

    print sA
    print tA
