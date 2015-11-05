import numpy
import numpy as np
import scipy
from pyscf import gto
import pyscf.pbc.gto as pgto
import pyscf.pbc.scf as pscf
from pyscf.pbc.df import df
import pyscf.pbc.dft
import pyscf.dft
from pyscf.pbc import tools


def get_j(cell, dm, auxcell, modcell):
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

    idx = where2(auxcell)
    #chg = numpy.ones(2)*nelec/cell.natm
    chg = numpy.ones(2)*2
    s1 = pscf.scfint.get_ovlp(auxcell)
    rho -= numpy.einsum('ij,j->i', s1[:,idx], chg)
    c2 = pscf.scfint.get_t(auxcell)
    v1 = np.linalg.solve(c2, 2*np.pi*rho)
    vj = np.einsum('ijk,k->ij', c3, v1)#.real

# remove the constant in potential
    vj -= v1[idx].sum()/cell.vol * ovlp
    return vj

# model potential
def get_jmod(cell, auxcell):
    idxb = []
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        e = auxcell.bas_exp(ib)[0]
        if l == 0 and abs(e-3.3) < .1:
            idxb.append(ib)
    chg = numpy.ones(2)*2
    modcell = auxcell.copy()
    modcell._bas = auxcell._bas[idxb].copy()
    modcell.nbas = len(modcell._bas)
    rhok = eval_rhok(chg, modcell)
    coulG = tools.get_coulG(auxcell)
    v2 = rhok * coulG
    # weight = vol/N,  1/vol * weight = 1/N
    # ifft has 1/N
    vw = tools.ifft(v2, cell.gs)

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    kpt = None
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    vj = np.dot(aoR.T.conj(), vw.reshape(-1,1)*aoR)#.real
#############
#    ip = 0
## normalize aux basis
#    for ib in range(auxcell.nbas):
#        l = auxcell.bas_angular(ib)
#        if l == 0:
#            e = auxcell.bas_exp(ib)[0]
#            ptr = auxcell._bas[ib,gto.PTR_COEFF]
#            auxcell._env[ptr] = 1/np.sqrt(4*np.pi)/gto.mole._gaussian_int(2,e)
#            ip += 1
#        else:
#            ip += 2*l+1
#    idx = where2(auxcell)
#    chg = numpy.ones(2)*2
#    s1 = pscf.scfint.get_ovlp(auxcell)
#    rho = numpy.einsum('ij,j->i', s1[:,idx], chg)
#    c2 = pscf.scfint.get_t(auxcell)
#    v1 = np.linalg.solve(c2, 2*np.pi*rho)
#
#    nao = cell.nao_nr()
#    c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
#    vj = np.einsum('ijk,k->ij', c3, v1)
    return vj

def eval_rhok(rho, cell):
    rhok = numpy.zeros(cell.Gv.shape[0], dtype=numpy.complex)
    k2 = numpy.einsum('ij,ij->i', cell.Gv, cell.Gv)
    for ib in range(cell.nbas):
        e = cell.bas_exp(ib)[0]
        r = cell.bas_coord(ib)
        si = numpy.exp(-1j*numpy.einsum('ij,j->i', cell.Gv, r))
        rhok += rho[ib] * si * numpy.exp(-k2/(4*e))
    return rhok

def where2(auxcell):
    ip = 0
    idx = []
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        e = auxcell.bas_exp(ib)[0]
        if l == 0:
            if abs(e-3.3)<.1:
                idx.append(ip)
            ip += 1
        else:
            ip += 2*l+1
    return np.array(idx)


def get_nuc(cell, auxcell, modcell):
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

    ovlp = pscf.scfint.get_ovlp(cell)
    idx = where2(auxcell)
    chg = numpy.ones(2)*2
    s1 = pscf.scfint.get_ovlp(auxcell)
    rho += numpy.einsum('ij,j->i', s1[:,idx], chg)

    c2 = pscf.scfint.get_t(auxcell)
    v1 = np.linalg.solve(c2, 2*np.pi*rho)

    nao = cell.nao_nr()
    c3 = df.aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    vnuc = np.einsum('ijk,k->ij', c3, v1)

# remove constant from potential. The constant contributes to V(G=0)
    vnuc -= v1[idx].sum()/cell.vol * ovlp

    return vnuc


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 14.
    n = 30
    cell = pgto.Cell()
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])
    cell.nimgs = [1,1,1]

    cell.atom = [['He' , ( L/2+0., L/2+0. ,   L/2+1.)],
                 ['He' , ( L/2+1., L/2+0. ,   L/2+1.)]]
    cell.basis = '631g'#{'He': [[0, (1.0, 1.0)]]}
    cell.build()
    mf = pdft.RKS(cell)
    mf.xc = 'LDA,VWN'
    #auxbasis = {'He': df.genbas(3., 2.8, (10.,0.1), 0)+df.genbas(3., 2. , (10.,0.4), 1)}
    auxbasis = {'He': df.genbas(3.3, 1.8, (10000.,0.1), 0)
                     +df.genbas(3.3, 2. , (10.,0.3), 1)
                     +df.genbas(3.3, 2. , (10.,0.3), 2)}
    auxcell = df.format_aux_basis(cell, auxbasis)
    print auxcell.nimgs
    auxcell.nimgs = [2,2,2]
    #auxcell.nimgs = [4,4,4]
    vjmod = get_jmod(cell, auxcell)
    mf.get_j = lambda cell, dm, *args: get_j(cell, dm, auxcell, None) + vjmod
    mf.get_hcore = lambda cell, *args: get_nuc(cell, auxcell, None)-vjmod + pscf.hf.get_t(cell)
    e1 = mf.scf()
    print e1 # ~ -4.32022187118
#    print mf.mo_energy

