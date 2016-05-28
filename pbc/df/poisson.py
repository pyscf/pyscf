import numpy as np
import scipy
import pyscf.gto
import pyscf.pbc.scf as pscf
import pyscf.pbc.dft as pdft
from pyscf.pbc.df import df
from pyscf.pbc import tools


def get_j_uniform_mod(cell, dm, auxcell, kpt=None, grids=None):
    ovlp = pscf.scfint.get_ovlp(cell)
    nao = ovlp.shape[0]
    if grids is None:
        c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    else:
        c3 = df.aux_e2_grid(cell, auxcell, grids).reshape(nao,nao,-1)
    rho = np.einsum('ijk,ij->k', c3, dm)

    nelec = np.einsum('ij,ij', ovlp, dm)
    intdf = auxnorm(auxcell)
    rho -= nelec / cell.vol * intdf

    v1 = np.linalg.solve(pscf.scfint.get_t(auxcell), 2*np.pi*rho)
    vj = np.einsum('ijk,k->ij', c3, v1)

# remove the constant in potential
#    vj += (np.dot(v1, rho) - (vj*dm).sum())/nelec * ovlp
#    v1[idx].sum()/cell.vol == (np.dot(v1, rho) - (vj*dm).sum())/nelec
    vj -= np.dot(v1, intdf) / cell.vol * ovlp
    return vj

def get_nuc_uniform_mod(cell, auxcell, kpt=None, grids=None):
    r'''Solving Poisson equation for Nuclear attraction

    \nabla^2|g>V_g = rho_nuc - C
    V_pq = \sum_g <pq|g> V_g
    '''
    intdf = auxnorm(auxcell)
    coords = np.asarray([cell.atom_coord(ia) for ia in range(cell.natm)])
    chargs =-np.asarray([cell.atom_charge(ia) for ia in range(cell.natm)])
    rho = np.dot(chargs, pdft.numint.eval_ao(auxcell, coords))
    rho += cell.nelectron / cell.vol * intdf

    ovlp = pscf.scfint.get_ovlp(cell)
    nao = ovlp.shape[0]
    if grids is None:
        c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    else:
        c3 = df.aux_e2_grid(cell, auxcell, grids).reshape(nao,nao,-1)
    v1 = np.linalg.solve(pscf.scfint.get_t(auxcell), 2*np.pi*rho)
    vnuc = np.einsum('ijk,k->ij', c3, v1)

# remove constant from potential. The constant contributes to V(G=0)
    vnuc -= np.dot(v1, intdf) / cell.vol * ovlp
    return vnuc


def get_j_gaussian_mod(cell, dm, auxcell, modcell, kpt=None, grids=None):
    ovlp = pscf.scfint.get_ovlp(cell)
    nao = ovlp.shape[0]
    if grids is None:
        c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    else:
        c3 = df.aux_e2_grid(cell, auxcell, grids).reshape(nao,nao,-1)
    rho = np.einsum('ijk,ij->k', c3, dm)
    modchg = np.asarray([cell.atom_charge(ia) for ia in range(cell.natm)])
    modchg = modchg / auxnorm(modcell)
    s1 = pscf.scfint.get_int1e_cross('cint1e_ovlp_sph', auxcell, modcell)
    rho -= np.einsum('ij,j->i', s1, modchg)

    v1 = np.linalg.solve(pscf.scfint.get_t(auxcell), 2*np.pi*rho)
    vj = np.einsum('ijk,k->ij', c3, v1)

# remove the constant in potential
    intdf = auxnorm(auxcell)
    vj -= np.dot(v1, intdf) / cell.vol * ovlp
    return vj

def get_nuc_gaussian_mod(cell, auxcell, modcell, kpt=None, grids=None):
    coords = np.asarray([cell.atom_coord(ia) for ia in range(cell.natm)])
    chargs = np.asarray([cell.atom_charge(ia) for ia in range(cell.natm)])
    rho = np.dot(-chargs, pdft.numint.eval_ao(auxcell, coords))

    modchg = chargs / auxnorm(modcell)
    s1 = pscf.scfint.get_int1e_cross('cint1e_ovlp_sph', auxcell, modcell)
    rho += np.einsum('ij,j->i', s1, modchg)

    ovlp = pscf.scfint.get_ovlp(cell)
    nao = ovlp.shape[0]
    if grids is None:
        c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    else:
        c3 = df.aux_e2_grid(cell, auxcell, grids).reshape(nao,nao,-1)
    v1 = np.linalg.solve(pscf.scfint.get_t(auxcell), 2*np.pi*rho)
    vnuc = np.einsum('ijk,k->ij', c3, v1)

# remove constant from potential. The constant contributes to V(G=0)
    intdf = auxnorm(auxcell)
    vnuc -= np.dot(v1, intdf) / cell.vol * ovlp
    return vnuc

def get_jmod_pw_poisson(cell, modcell, kpt=None):
    modchg = np.asarray([cell.atom_charge(ia) for ia in range(cell.natm)])
    rhok = 0
    k2 = np.einsum('ij,ij->i', cell.Gv, cell.Gv)
    for ib in range(modcell.nbas):
        e = modcell.bas_exp(ib)[0]
        r = modcell.bas_coord(ib)
        si = np.exp(-1j*np.einsum('ij,j->i', cell.Gv, r))
        rhok += modchg[ib] * si * np.exp(-k2/(4*e))

    vk = rhok * tools.get_coulG(cell)
    # weight = vol/N,  1/vol * weight = 1/N
    # ifft has 1/N
    vw = tools.ifft(vk, cell.gs)

    coords = pdft.gen_grid.gen_uniform_grids(cell)
    aoR = pdft.numint.eval_ao(cell, coords, None)
    vj = np.dot(aoR.T.conj(), vw.reshape(-1,1) * aoR)
    return vj

#ABORTdef get_jmod_gaussian_poisson(cell, auxcell, modcell, kpt=None):
#ABORT    modchg = np.asarray([cell.atom_charge(ia) for ia in range(cell.natm)])
#ABORT    modchg = modchg / auxnorm(modcell)
#ABORT    s1 = pscf.scfint.get_int1e_cross('cint1e_ovlp_sph', auxcell, modcell)
#ABORT    rho = np.einsum('ij,j->i', s1, modchg)
#ABORT
#ABORT    v1 = np.linalg.solve(pscf.scfint.get_t(auxcell), 2*np.pi*rho)
#ABORT    ovlp = pscf.scfint.get_ovlp(cell)
#ABORT    nao = ovlp.shape[0]
#ABORT    c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
#ABORT    vj = np.einsum('ijk,k->ij', c3, v1)
#ABORT
#ABORT# remove the constant in potential
#ABORT    intdf = auxnorm(auxcell)
#ABORT    vj -= np.dot(v1, intdf) / cell.vol * ovlp
#ABORT    return vj

def genbas(beta, bound0=(1e11,1e-8), l=0):
    basis = []
    e = bound0[0]
    while e > bound0[1]:
        basis.append([l, [e,1]])
        e /= beta
    return basis

def auxnorm(cell):
    ''' \int gto dr
    '''
    norm = np.zeros(cell.nao_nr())
    ip = 0
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        e = cell.bas_exp(ib)[0]
        c = cell.bas_ctr_coeff(ib)[0,0]
        if l == 0:
            norm[ip] = pyscf.gto.mole._gaussian_int(2, e) * c*np.sqrt(np.pi*4)
            ip += 1
        else:
            ip += 2*l+1
    return norm



if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 4.
    n = 30
    cell = pgto.Cell()
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])

    cell.atom = '''He     0.    0.       0.
                   He     1.    1.       1.'''
    cell.basis = '631g'#{'He': [[0, (1.0, 1.0)]]}
    cell.verbose = 5
    cell.build()
    mf = pdft.RKS(cell)
    mf.xc = 'LDA,VWN'
    auxbasis = {'He': genbas(1.8, (100,.05), 0)
                     +genbas(2. , (10,0.1), 1)
                     +genbas(2. , (10,0.1), 2)}
    auxcell = df.format_aux_basis(cell, auxbasis)
    gds = pdft.gen_grid.BeckeGrids(cell)
    gds.level = 3
    gds.build()
    mf.get_j = lambda cell, dm, *args: get_j_uniform_mod(cell, dm, auxcell, grids=gds)
    mf.get_hcore = lambda cell, *args: get_nuc_uniform_mod(cell, auxcell, grids=gds) + pscf.hf.get_t(cell)
    print mf.scf() # ~ -4.32049313872

    modcell = df.format_aux_basis(cell, {'He': [[0, [3, 1]]]})
    vjmod = get_jmod_pw_poisson(cell, modcell, grids=gds)
    mf.get_j = lambda cell, dm, *args: get_j_gaussian_mod(cell, dm, auxcell, modcell, grids=gds) + vjmod
    mf.get_hcore = lambda cell, *args: get_nuc_gaussian_mod(cell, auxcell, modcell, grids=gds)-vjmod + pscf.hf.get_t(cell)
    print mf.scf() # ~ -4.32049449915
