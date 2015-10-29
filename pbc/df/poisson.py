import numpy as np
import scipy
from pyscf import gto
import pyscf.pbc.gto as pgto
import pyscf.pbc.scf as pscf
from pyscf.pbc.df import df


def get_j(cell, dm, auxcell):
    nao = dm.shape[0]
    c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    rho = np.einsum('ijk,ij->k', c3, dm)

    norm = np.zeros_like(rho)
    ip = 0
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        e = auxcell.bas_exp(ib)[0]
        c = auxcell.bas_ctr_coeff(ib)[0,0]
        if l == 0:
            norm[ip] = gto.mole._gaussian_int(2, e) * c * np.sqrt(np.pi*4)
            ip += 1
        else:
            ip += 2*l+1

    ovlp = pscf.scfint.get_ovlp(cell)
    nelec = np.einsum('ij,ij', ovlp, dm)
    rho -= nelec/cell.vol * norm

    c2 = pscf.scfint.get_t(auxcell)

    v1 = np.linalg.solve(c2, 2*np.pi*rho)
    vj = np.einsum('ijk,k->ij', c3, v1)

# remove a constant in potential
    vj += (np.dot(v1, rho) - (vj*dm).sum())/nelec * ovlp

    return vj

# define get_nuc


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 4.
    n = 30
    cell = pgto.Cell()
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])

    cell.atom = '''He     0.    0.       1.
                   He     1.    0.       1.'''
    cell.basis = {'He': [[0, (1.0, 1.0)]]}
    cell.build()
    mf = pdft.RKS(cell)
    mf.xc = 'LDA,VWN'
    auxbasis = {'He': df.genbas(2., 1.8, (100.,0.1), 0)
                     +df.genbas(2., 2. , (10.,0.1), 1)
                     +df.genbas(2., 2. , (10.,0.1), 2)}
    auxcell = df.format_aux_basis(cell, auxbasis)
    mf.get_j = lambda cell, dm, *args: get_j(cell, dm, auxcell)
    e1 = mf.scf()
    print e1 # -4.31976039944 slightly different to -4.32022187118
