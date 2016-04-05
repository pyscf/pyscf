import numpy
from pyscf import gto
from pyscf import lib
from pyscf.dft import numint
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf
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
fakeao = numint.eval_ao(fakemol, Gv, deriv=0)

G_rad = lib.norm(Gv, axis=1)
Gs,thetas,phis = pyscf.pbc.gto.pseudo.pp.cart2polar(Gv)

p0 = 0
for ia in range(cell.natm):
    pp = cell._pseudo[cell.atom_symbol(ia)]
    for l, proj in enumerate(pp[5:]):
        rl, nl, hl = proj

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
        pkl = numpy.asarray(pkl)

        for m in range(-l,l+1):
            projG_ang = pyscf.pbc.gto.pseudo.pp.Ylm(l,m,thetas,phis).conj()
            for i in range(nl):
                projG_radial = pyscf.pbc.gto.pseudo.pp.projG_li(Gs,l,i,rl)
                ref = projG_radial*projG_ang
                print l+m, numpy.allclose(ref, pkl[i,:,l+m])

