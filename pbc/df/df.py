
import numpy
import pyscf.df.incore
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
    nimgs = numpy.max((cell.nimgs, auxcell.nimgs),axis=0)
    Ls = tools.pbc.get_lattice_Ls(cell, nimgs)
    logger.debug1(cell, "Images summed over in DFT %s", nimgs)
    logger.debug2(cell, "Ls = %s", Ls)

    # cell with *all* images
    rep_cell = cell.copy()
    rep_cell.atom = []
    for L in Ls:
        for atom, coord in cell._atom:
            rep_cell.atom.append([atom, coord + L])
    rep_cell.unit = 'B'
    rep_cell.build(False,False)
    rep_cell.nimgs = rep_cell.get_nimgs(rep_cell.precision)

    rep_aux_e2 = pyscf.df.incore.aux_e2(rep_cell, auxcell, intor, aosym='s1')

    nao = cell.nao_nr()
    naoaux = auxcell.nao_nr()
    nL = len(Ls)

    rep_aux_e2=rep_aux_e2.reshape(nao*nL,nao*nL,-1)

    aux_e2=numpy.zeros([nao,nao,naoaux])

    # double lattice sum
    for l in range(len(Ls)):
        for m in range(len(Ls)):
            aux_e2+=rep_aux_e2[l*nao:(l+1)*nao,m*nao:(m+1)*nao,:]

    aux_e2.reshape([nao*nao,naoaux])

    return aux_e2

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
    aux_e2 = numpy.einsum('ri,rj,rk',ao,ao,auxao)
    return aux_e2

def aux_3c1e_grid(cell, auxcell, grids=None):
    return aux_e2_grid(cell, auxcell, gs, grids)



