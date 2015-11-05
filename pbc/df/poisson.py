import numpy as np
import scipy
from pyscf import gto
import pyscf.pbc.gto as pgto
import pyscf.pbc.scf as pscf
import pyscf.pbc.dft as pdft
from pyscf.pbc.df import df


def get_j(cell, dm, auxcell):
    idx = []
    ip = 0
# normalize aux basis
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        if l == 0:
            e = auxcell.bas_exp(ib)[0]
            ptr = auxcell._bas[ib,gto.PTR_COEFF]
            auxcell._env[ptr] = 1/np.sqrt(4*np.pi)/gto.mole._gaussian_int(2,e)
            idx.append(ip)
            ip += 1
        else:
            ip += 2*l+1

    nao = dm.shape[0]
    c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    rho = np.einsum('ijk,ij->k', c3, dm)

    ovlp = pscf.scfint.get_ovlp(cell)
    nelec = np.einsum('ij,ij', ovlp, dm)
    rho[idx] -= nelec/cell.vol

    c2 = pscf.scfint.get_t(auxcell)

    v1 = np.linalg.solve(c2, 2*np.pi*rho)
    vj = np.einsum('ijk,k->ij', c3, v1)

# remove the constant in potential
#    vj += (np.dot(v1, rho) - (vj*dm).sum())/nelec * ovlp
#    v1[idx].sum()/cell.vol == (np.dot(v1, rho) - (vj*dm).sum())/nelec
    vj -= v1[idx].sum()/cell.vol * ovlp

    return vj

def genbas(beta, bound0=(1e11,1e-8), l=0):
    basis = []
    e = bound0[0]
    while e > bound0[1]:
        basis.append([l, [e,1]])
        e /= beta
    return basis

def get_nuc(cell, auxcell):
    r'''Solving Poisson equation for Nuclear attraction

    \nabla^2|g>V_g = rho_nuc - C
    V_pq = \sum_g <pq|g> V_g
    '''

    idx = []
    ip = 0
# normalize aux basis
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        if l == 0:
            e = auxcell.bas_exp(ib)[0]
            ptr = auxcell._bas[ib,gto.PTR_COEFF]
            auxcell._env[ptr] = 1/np.sqrt(4*np.pi)/gto.mole._gaussian_int(2,e)
            idx.append(ip)
            ip += 1
        else:
            ip += 2*l+1

    # \sum_A \int ao Z_A delta(r_A)
    coords = np.asarray([cell.atom_coord(ia) for ia in range(cell.natm)])
    chargs =-np.asarray([cell.atom_charge(ia) for ia in range(cell.natm)])
    rho = np.dot(chargs, pdft.numint.eval_ao(auxcell, coords))
    rho[idx] += cell.nelectron/cell.vol

    c2 = pscf.scfint.get_t(auxcell)
    v1 = np.linalg.solve(c2, 2*np.pi*rho)

    nao = cell.nao_nr()
    c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    vnuc = np.einsum('ijk,k->ij', c3, v1)

# remove constant from potential. The constant contributes to V(G=0)
    ovlp = pscf.scfint.get_ovlp(cell)
    vnuc -= v1[idx].sum()/cell.vol * ovlp

    return vnuc



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
    auxbasis = {'He': genbas(1.8, (100.,.05), 0)
                     +genbas(2. , (10.,0.3), 1)
                     +genbas(2. , (10.,0.3), 2)}
    auxcell = df.format_aux_basis(cell, auxbasis)
#    mf.get_j = lambda cell, dm, *args: get_j(cell, dm, auxcell)
#    mf.get_hcore = lambda cell, *args: get_nuc(cell, auxcell) + pscf.hf.get_t(cell)
#    e1 = mf.scf()
#    print e1 # ~ -4.32022187118

    import pyscf.dft
    mf = pyscf.dft.RKS(cell)
    mf.kernel()
    dm = mf.make_rdm1()
    vj1 = get_j(cell, dm, auxcell)
    v1 = get_nuc(cell, auxcell)
    print vj1
    print v1
    print vj1+v1
