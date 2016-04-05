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

def _real2cmplx(ao, l):
    if l == 0:
        return ao + 0j
    elif l == 1:
        ao = ao[:,(1,2,0)]

    n = 2 * l + 1
    u = numpy.zeros((n,n),dtype=complex)
    u[l,l] = 1
    s2 = numpy.sqrt(.5)
    for m in range(1, l+1, 2):
        u[l-m,l-m] =-s2 * 1j
        u[l+m,l-m] = s2
        u[l-m,l+m] =-s2 * 1j
        u[l+m,l+m] =-s2
    for m in range(2, l+1, 2):
        u[l-m,l-m] =-s2 * 1j
        u[l+m,l-m] = s2
        u[l-m,l+m] = s2 * 1j
        u[l+m,l+m] = s2
    return numpy.dot(ao, u).conj()


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
    bas = []
    env = []
    ptr = 0
    fakemol._atm[0,gto.PTR_COORD] = ptr
    env.append([0.]*3)
    ptr += 3
    for ia in range(cell.natm):
        pp = cell._pseudo[cell.atom_symbol(ia)]
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            bas.append([0, l, 1, 1, 0, ptr, ptr+1, 0])
            env.append([.5*rl**2, rl**(l+1.5)*numpy.pi**1.25])
            ptr += 2
    fakemol._bas = numpy.asarray(bas, dtype=numpy.int32)
    fakemol._env = numpy.hstack(env)
    Gv = numpy.asarray(cell.Gv)
    G_rad = lib.norm(Gv, axis=1)
    fakeao = numint.eval_ao(fakemol, Gv, deriv=0)

    vppnl = numpy.zeros((nao,nao), dtype=numpy.complex128)
    p0 = 0
    for ia in range(cell.natm):
        pp = cell._pseudo[cell.atom_symbol(ia)]
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            hl = numpy.asarray(hl)

            pkl_part = fakeao[:,p0:p0+l*2+1]
            p0 += l * 2 + 1
            pkl_part = _real2cmplx(pkl_part, l)
            pkl = []
            for k in range(nl):
                qkl = pyscf.pbc.gto.pseudo.pp._qli(G_rad*rl, l, k)
                if qkl.ndim == 0:
                    pkl.append(pkl_part * qkl)
                else:
                    pkl.append(numpy.einsum('gm,g->gm', pkl_part, qkl))
            pkl = numpy.asarray(pkl) * 1j**l

            for m in range(-l,l+1):
                SPG_lm_aoG = numpy.zeros((nl,nao), dtype=numpy.complex128)
                for i in range(nl):
                    SPG_lmi = SI[ia,:] * pkl[i,:,l+m]
                    SPG_lm_aoG[i,:] = numpy.einsum('g,gp->p', SPG_lmi.conj(), aokG)
                for i in range(nl):
                    for j in range(nl):
                        vppnl += hl[i,j]*numpy.einsum('p,q->pq',
                                                   SPG_lm_aoG[i,:].conj(),
                                                   SPG_lm_aoG[j,:])
    vppnl *= (1./ngs**2)

    return vpploc + vppnl


ref = scf.hf.get_pp(cell)
dat = get_pp(cell)
print numpy.allclose(ref, dat)
