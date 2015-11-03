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

    idx = where2(auxcell)
    chg = numpy.ones(2)*nelec/cell.natm
    s1 = pscf.scfint.get_ovlp(auxcell)
    rho -= numpy.einsum('ij,j->i', s1[:,idx], chg)
    c2 = pscf.scfint.get_t(auxcell)
    v1 = np.linalg.solve(c2, 2*np.pi*rho)
    vj = np.einsum('ijk,k->ij', c3, v1)#.real

#CHECK: shoule I remove a constant potential?
#    vj += (np.dot(v1, rho) - (vj*dm).sum())/nelec * ovlp

# model potential
    idxb = []
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        e = auxcell.bas_exp(ib)[0]
        if l == 0 and abs(e-3.3) < .1:
            idxb.append(ib)
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
    v3 = np.dot(aoR.T.conj(), vw.reshape(-1,1)*aoR)#.real
    vj += v3
    print v3
    return vj

def eval_rhok(rho, cell):
    rhok = numpy.zeros(cell.Gv.T.shape[0], dtype=numpy.complex)
    k2 = numpy.einsum('ij,ij->i', cell.Gv.T, cell.Gv.T)
    for ib in range(cell.nbas):
        e = cell.bas_exp(ib)[0]
        r = cell.bas_coord(ib)
        si = numpy.exp(-1j*numpy.einsum('ij,j->i', cell.Gv.T, r))
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


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 4.
    n = 30
    cell = pgto.Cell()
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([n,n,n])

    cell.atom = [['He' , ( L/2+0., L/2+0. ,   L/2+1.)],
                 ['He' , ( L/2+1., L/2+0. ,   L/2+1.)]]
    cell.basis = {'He': [[0, (1.0, 1.0)]]}
    cell.build()
    mf = pdft.RKS(cell)
    mf.xc = 'LDA,VWN'
    #auxbasis = {'He': df.genbas(3., 2.8, (10.,0.1), 0)+df.genbas(3., 2. , (10.,0.4), 1)}
    auxbasis = {'He': df.genbas(3.3, 1.8, (100.,0.3), 0)
                     +df.genbas(3.3, 2. , (10.,0.3), 1)
                     +df.genbas(3.3, 2. , (10.,0.3), 2)}
    auxcell = df.format_aux_basis(cell, auxbasis)
    mf.get_j = lambda cell, dm, *args: get_j(cell, dm, auxcell)
    e1 = mf.scf()
    print e1 # ~ -4.32022187118
