
import numpy
import pyscf.df.incore
import pyscf.lib
import pyscf.lib.parameters as param
import pyscf.gto.mole
from pyscf.lib import logger
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc import gto
from pyscf.pbc import tools

def format_aux_basis(cell, auxbasis='weigend'):
    '''
    See df.incore.format_aux_basis
    '''
    auxcell = cell.copy()
    auxcell.basis = auxbasis
    auxcell.precision = 1.e-9
    auxcell.build(False,False)
    auxcell.nimgs = auxcell.get_nimgs(auxcell.precision)

    return auxcell

def aux_e2(cell, auxcell, intor):
    '''3-center AO integrals (ij|L), where L is the auxiliary basis.

    Implements double summation over lattice vectors: \sum_{lm} (i[l]j[m]|L[0]).
    '''
    # sum over largest number of images in either cell or auxcell
    nimgs = numpy.max((cell.nimgs, auxcell.nimgs), axis=0)
    Ls = tools.pbc.get_lattice_Ls(cell, nimgs)
    logger.debug1(cell, "Images summed over in DFT %s", nimgs)
    logger.debug2(cell, "Ls = %s", Ls)

    nao = cell.nao_nr()
    nao_pair = nao*(nao+1) // 2
    nao_pair = nao*nao
    naoaux = auxcell.nao_nr()
    cellL = cell.copy()
    cellR = cell.copy()
    _envL = cellL._env
    _envR = cellR._env
    ptr_coord = cellL._atm[:,pyscf.gto.PTR_COORD]
    buf = numpy.zeros((nao_pair,naoaux))
    for l, L1 in enumerate(Ls):
        _envL[ptr_coord+0] = cell._env[ptr_coord+0] + L1[0]
        _envL[ptr_coord+1] = cell._env[ptr_coord+1] + L1[1]
        _envL[ptr_coord+2] = cell._env[ptr_coord+2] + L1[2]
        for m in range(l):
            _envR[ptr_coord+0] = cell._env[ptr_coord+0] + Ls[m][0]
            _envR[ptr_coord+1] = cell._env[ptr_coord+1] + Ls[m][1]
            _envR[ptr_coord+2] = cell._env[ptr_coord+2] + Ls[m][2]

            buf += pyscf.df.incore.aux_e2(cellL, auxcell, intor, mol1=cellR)
        buf += .5 * pyscf.df.incore.aux_e2(cellL, auxcell, intor, mol1=cellL)
    eri = buf.reshape(nao,nao,-1)
    return eri + eri.transpose(1,0,2)

def aux_e2_grid(cell, auxcell, grids=None):
    '''3-center AO integrals (ij|L), where L is the auxiliary basis.

    Implements double summation over lattice vectors: \sum_{lm} (i[l]j[m]|L[0]).
    '''
    if grids is None:
        grids = gen_grid.BeckeGrids(cell)
        grids.build_()
    elif grids.weights is None:
        raise RuntimeError('grids is not initialized')

    ao = numpy.asarray(numint.eval_ao(cell, grids.coords).real, order='C')
    auxao = numpy.asarray(numint.eval_ao(auxcell, grids.coords).real, order='C')
    if isinstance(grids.weights, numpy.ndarray):
        auxao *= grids.weights.reshape(-1,1)
    else:
        auxao *= grids.weights
    ng, nao = ao.shape
    naoaux = auxao.shape[1]
    #aux_e2 = numpy.einsum('ri,rj,rk',ao,ao,auxao)
    aux_e2 = numpy.zeros((nao*nao,naoaux))
    for p0, p1 in prange(0, ng, 240):
        tmp = numpy.einsum('ri,rj->rij', ao[p0:p1], ao[p0:p1])
        pyscf.lib.dot(tmp.reshape(p1-p0,-1).T, auxao[p0:p1], 1, aux_e2, 1)
    return aux_e2.reshape(nao,nao,-1)

def aux_3c1e_grid(cell, auxcell, grids=None):
    return aux_e2_grid(cell, auxcell, gs, grids)


def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)
