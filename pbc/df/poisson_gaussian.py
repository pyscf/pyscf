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
    vj = np.einsum('ijk,k->ij', c3, v1)
    vj = vj.real

# remove a constant in potential
#    vj += (np.dot(v1, rho) - (vj*dm).sum())/nelec * ovlp
#    print (np.dot(v1, rho) - (vj*dm).sum())

# model potential
    idxb = []
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        e = auxcell.bas_exp(ib)[0]
        if l == 0 and abs(e-2.) < .1:
            idxb.append(ib)
    modcell = auxcell.copy()
    modcell._bas = auxcell._bas[idxb].copy()
    modcell.nbas = len(modcell._bas)
    rhok = eval_rhok(chg, modcell)
    coulG = tools.get_coulG(auxcell)
    v2 = rhok * coulG
    v2 = tools.ifft(v2, cell.gs)# / cell.vol

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    kpt = None
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    v3 = np.dot(aoR.T.conj(), v2.reshape(-1,1)*aoR).real
    #print vj
    vj += v3
    #print v3
    return vj

#def eval_rhok(rho, cell):
#    kcell = cell.copy()
#    _env = []
#    ptr = len(cell._env)
#    for ib in range(kcell.nbas):
#        e = cell.bas_exp(ib)[0]
#        _env.append(1/(4*e))
#        _env.append(np.sqrt(4*np.pi)*(np.pi/e)**1.5)
#        kcell._bas[ib,gto.PTR_EXP] = ptr
#        kcell._bas[ib,gto.PTR_COEFF] = ptr + 1
#        ptr += 2
#    kcell._env = numpy.hstack((cell._env, _env))
#    SI = kcell.get_SI()
#    aoR = pyscf.dft.numint.eval_ao(kcell, kcell.Gv.T.copy())
#    rhok = np.einsum('ij,ji,j->i', aoR, SI, rho)
#    return rhok.real

def eval_rhok(rho, cell):
    rhok = numpy.zeros(cell.Gv.T.shape[0], dtype=numpy.complex)
    for ib in range(cell.nbas):
        e = cell.bas_exp(ib)[0]
        r = cell.bas_coord(ib)
        si = numpy.exp(-1j*numpy.einsum('ij,j->i', cell.Gv.T, r))
        k2 = numpy.einsum('ij,ij->i', cell.Gv.T, cell.Gv.T)
        rhok += rho[ib] * si * numpy.exp(-k2/(4*e))
    return rhok

def where2(auxcell):
    ip = 0
    idx = []
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        e = auxcell.bas_exp(ib)[0]
        if l == 0:
            if abs(e-2)<.1:
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

    cell.atom = '''He     0.    0.       1.
                   He     1.    0.       1.'''
    cell.basis = {'He': [[0, (1.0, 1.0)]]}
    cell.build()
    mf = pdft.RKS(cell)
    mf.xc = 'LDA,VWN'
    auxbasis = {'He': df.genbas(2., 2.8, (10.,0.1), 0)+df.genbas(2., 2. , (10.,0.4), 1)}
    #auxbasis = {'He': df.genbas(2., 1.8, (100.,0.1), 0)
    #                 +df.genbas(2., 2. , (10.,0.1), 1)
    #                 +df.genbas(2., 2. , (10.,0.1), 2)}
    auxcell = df.format_aux_basis(cell, auxbasis)
    mf.get_j = lambda cell, dm, *args: get_j(cell, dm, auxcell)
    e1 = mf.scf()
    print e1
