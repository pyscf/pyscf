import numpy
from pyscf import gto
from pyscf import lib
from pyscf.dft import numint
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf
from pyscf.pbc import tools
import pyscf.pbc.gto.pseudo.pp

cell = pbcgto.Cell()

L = 4.
n = 2
cell.unit = 'A'
cell.atom = '''
        Fe    0.           0.           0.        ;
        Fe    2.           0.1          0.        ;
        Fe    0.89170002   2.67509998   2.67509998;
    '''
cell.h = numpy.diag([L,L,L])
cell.gs = numpy.array([n,n,n])

#cell.basis = 'gth-szv'
cell.basis = 'sto3g'
cell.pseudo = 'gth-pade'

cell.build()

#ref = pbcgto.pseudo.get_gth_projG(cell, cell.Gv)
#proj_ia = ref[1][0]
#proj_ia_l = proj_ia[0]
#proj_ia_lm = proj_ia_l[0]
#print proj_ia_lm[0].shape


def get_pp(cell, kpt=numpy.zeros(3)):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    nao = cell.nao_nr()

    SI = cell.get_SI()
    vlocG = pbcgto.pseudo.get_vlocG(cell)
    vpplocG = -numpy.sum(SI * vlocG, axis=0)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs)
    vpploc = numpy.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)

    # vppnonloc evaluated in reciprocal space
    aokG = numpy.empty(aoR.shape, numpy.complex128)
    for i in range(nao):
        aokG[:,i] = tools.fftk(aoR[:,i], cell.gs, coords, kpt)
    ngs = len(aokG)

    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    fakemol._env = numpy.zeros(5)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = 3
    fakemol._bas[0,gto.PTR_COEFF] = 4
    Gv = numpy.asarray(cell.Gv)
    G_rad = lib.norm(Gv, axis=1)

    vppnl = numpy.zeros((nao,nao), dtype=numpy.complex128)
    for ia in range(cell.natm):
        pp = cell._pseudo[cell.atom_symbol(ia)]
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            hl = numpy.asarray(hl)
            fakemol._bas[0,gto.ANG_OF] = l
            fakemol._env[3] = .5*rl**2
            fakemol._env[4] = rl**(l+1.5)*numpy.pi**1.25
            pYlm_part = numint.eval_ao(fakemol, Gv, deriv=0)

            pYlm = numpy.empty((nl,l*2+1,ngs))
            for k in range(nl):
                qkl = pyscf.pbc.gto.pseudo.pp._qli(G_rad*rl, l, k)
                pYlm[k] = projG_part.T * qkl

            # pYlm is real
            SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
            SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
            tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
            vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
    vppnl *= (1./ngs**2)

    return vpploc + vppnl


ref = scf.hf.get_pp(cell)
dat = get_pp(cell)
print numpy.allclose(ref, dat)
