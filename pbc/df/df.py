
import numpy
import pyscf.df.incore
import pyscf.lib.parameters as param
import pyscf.gto.mole
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc import gto
from pyscf.pbc.scf import scfint

def format_aux_basis(cell, auxbasis='weigend'):
    '''
    See df.incore.format_aux_basis
    '''
    #auxmol=pyscf.df.incore.format_aux_basis(cell, auxbasis)
    auxcell=cell.copy()
    auxcell.basis=auxbasis
    precision=1.e-9 # FIXME
    auxcell.build(False,False)
    #auxcell.nimgs=gto.get_nimgs(auxcell, precision)

    # auxcell.__dict__.update(auxmol.__dict__)
    # print "MOL"
    # print auxmol._basis
    # print auxcell._basis

    return auxcell

def aux_e2(cell, auxcell, intor):
    '''3-center AO integrals (ij|L), where L is the auxiliary basis.

    Implements double summation over lattice vectors: \sum_{lm} (i[l]j[m]|L[0]).
    '''
    nimgs=[0,0,0]
    # sum over largest number of images in either cell or auxcell
    for i in range(3):
        ci = cell.nimgs[i]
        ai = auxcell.nimgs[i]
        nimgs[i] = ci if ci > ai else ai

    print "Images summed over in DFT", nimgs
    Ls = scfint.get_lattice_Ls(cell, nimgs)
    Ms = Ls

    # cell with *all* images
    rep_cell = cell.copy()
    rep_cell.atom = []
    for L in Ls:
        for atom, coord in cell._atom:
            #print "Lattice L", L
            rep_cell.atom.append([atom, coord + L])
    rep_cell.unit = 'B'
    rep_cell.build(False,False)
    #print "ATOMS", rep_cell.atom

    rep_aux_e2 = pyscf.df.incore.aux_e2(rep_cell, auxcell, intor, aosym='s1')

    nao = cell.nao_nr()
    naoaux = auxcell.nao_nr()
    nL = len(Ls)
    
    rep_aux_e2=rep_aux_e2.reshape(nao*nL,nao*nL,-1)
    
    aux_e2=numpy.zeros([nao,nao,naoaux])

    # print rep_aux_e2.shape
    # for (i, j, k), val in numpy.ndenumerate(rep_aux_e2):
    #     print (i,j,k), val

    # double lattice sum
    for l in range(len(Ls)):
        for m in range(len(Ls)):
            aux_e2+=rep_aux_e2[l*nao:(l+1)*nao,m*nao:(m+1)*nao,:]

    aux_e2.reshape([nao*nao,naoaux])
    
    return aux_e2

def aux_e2_grid(cell, auxcell, gs):

    coords=gen_grid.gen_uniform_grids(cell)
    nao=cell.nao_nr()
    ao=numint.eval_ao(cell, coords)
    auxao=numint.eval_ao(auxcell, coords)
    naoaux=auxcell.nao_nr()

    aux_e2=numpy.einsum('ri,rj,rk',ao,ao,auxao)*cell.vol/coords.shape[0]
    aux_e2.reshape([nao*nao,naoaux])

    return aux_e2


def test_df():
    from pyscf import gto
    from pyscf.lib.parameters import BOHR
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import numpy as np
    import scipy

    B = BOHR

    Lunit = 4
    Ly = Lz = Lunit
    Lx = Lunit

    h = np.diag([Lx,Ly,Lz])
    
    # these are some exponents which are 
    # not hard to integrate
    cell = pgto.Cell()
    #cell.atom = [['He', (2*B, 0.5*Ly*B, 0.5*Lz*B)],
    #             ['He', (3*B, 0.5*Ly*B, 0.5*Lz*B)]]
    cell.atom = [['He', (0,0,0)], ['He', (1,1,1)]]
    cell.basis = { 'He': [[0, (1.0, 1.0)]] }

    #cell.unit='A'
    cell.unit='B'
    
    cell.h = h
    cell.nimgs = [3,3,3]
    cell.pseudo = None
    cell.output = None
    cell.verbose = 0
    cell.gs = np.array([40,40,40])

    cell.build(0, 0)

    # DF overlap
    auxcell=format_aux_basis(cell, auxbasis=cell.basis)
    c3=aux_e2(cell, auxcell, 'cint3c1e_sph')
    c3grid=aux_e2_grid(cell, auxcell, cell.gs)

    for i in range(cell.nao_nr()):
        for j in range(cell.nao_nr()):
            for k in range(cell.nao_nr()):
                print 'i,j,k', i,j,k, c3[i,j,k], c3grid[i,j,k]
    
    print np.linalg.norm(c3-c3grid) # should be zero within integration error

def auxnorm(cell):
    norm=numpy.zeros([cell.nao_nr()])
    ip=0
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        e = cell.bas_exp(ib)[0]
        c = cell.bas_ctr_coeff(ib)[0,0]
        if l == 0:
            norm[ip] = pyscf.gto.mole._gaussian_int(2, e) * c*numpy.sqrt(numpy.pi*4)
            ip += 1
        else:
            ip += 2*l+1
    return norm

def genbas(rho_exp, beta, bound0=(1e11,1e-8), l=0):
    #rho_exp = 2.  # unit charge, distribution ~ e^{-4*r^2}
    #beta = 3
    basis1 = []
    basis2 = []
    e = rho_exp
    while e < bound0[0]:
        basis1.append([l, [e,1.]])
        e *= beta
    rho_id = len(basis1) - 1
    e = rho_exp
    while e > bound0[1]:
        e /= beta
        basis2.append([l, [e,1.]])
    basis = list(reversed(basis1)) + basis2
    return basis


def test_poisson():
    from pyscf import gto
    from pyscf.lib.parameters import BOHR
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import numpy as np
    import scipy

    B = BOHR

    Lunit = 4.*B
    Ly = Lz = Lunit*B
    Lx = Lunit*B

    h = np.diag([Lx,Ly,Lz])
    
    cell = pgto.Cell()
    cell.atom = '''He     1.    1.       1.'''
    cell.basis = {'He': [[0, (1.0, 1.0)]]}
    cell.h = h
    n = 30
    cell.gs = np.array([n,n,n])
    cell.pseudo = None
    cell.output = None
    cell.verbose = 0
    cell.build()

    bas=genbas(2., 1.8,(1000.,0.3), 0)
    auxcell=format_aux_basis(cell, {'He': bas })

    gs = np.array([40,40,40])

    ovlp=scfint.get_ovlp(cell)
    dm=2*scipy.linalg.inv(ovlp) 

    print "DM", dm
    nelec=np.einsum('ij,ij',dm,ovlp)
    print "nelec", nelec

    j=pscf.hf.get_j(cell, dm)
    ref_j=np.einsum('ij,ij',j,dm)
    print "Reference chargeless Coulomb", ref_j

    nao=dm.shape[0]
    c3=aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    rho=numpy.einsum('ijk,ij->k', c3, dm)

    norm=auxnorm(auxcell)
    rho1=rho-nelec*norm/cell.vol

    c2 = scfint.get_t(auxcell) * 2

    v1 = numpy.linalg.solve(c2, rho1*4*numpy.pi) # computed this way
                                                 # v1 still has some constant
                                                 # This doesn't matter
                                                 # when integrating with the 
                                                 # chargeless density
                                                 # but this constant can also
                                                 # be set to zero by fitting the
                                                 # potential
                                                 # chargeless Gaussians
        
    print "Charged Coulomb energy", numpy.dot(v1,rho)  # Integral with charged density. This gives
                             # a contribution from the constant in v, 
                             # = nelec * \int v

    print "DF Chargeless Coulomb energy", numpy.dot(v1,rho1) # Integral with the chargeless density. This
                             # should agree with the k-space Coulomb
                             # sum which omits K=0

    print "Error of DF from reference", numpy.dot(v1,rho1)-ref_j # should be close to zero

#test_poisson()
