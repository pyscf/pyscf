
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

    print "AUX NIMGS", nimgs
    Ls = scfint.get_lattice_Ls(cell, nimgs)
    Ms = Ls

    # cell with *all* images
    rep_cell = cell.copy()
    rep_cell.atom = []
    for L in Ls:
        for atom, coord in cell._atom:
            rep_cell.atom.append([atom, coord + L])

    rep_cell.build(False,False)

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

def aux_e2_grid(cell, auxcell, gs):

    coords=gen_grid.gen_uniform_grids(cell, gs)
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

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    Lunit = 2
    Ly = Lz = Lunit
    Lx = Lunit

    h = np.diag([Lx,Ly,Lz])
    
    mol.atom.extend([['He', (2*B, 0.5*Ly*B, 0.5*Lz*B)],
                     ['He', (3*B, 0.5*Ly*B, 0.5*Lz*B)]])

    # these are some exponents which are 
    # not hard to integrate
    mol.basis = { 'He': [[0, (1.0, 1.0)]] }
    mol.unit='A'
    mol.build(0, 0)

    cell = pgto.Cell()
    cell.__dict__ = mol.__dict__ # hacky way to make a cell
    cell.h = h
    #cell.vol = scipy.linalg.det(cell.h)
    cell.nimgs = [3,3,3]
    cell.pseudo = None
    cell.output = None
    cell.verbose = 0
    cell.build(0, 0)

    gs = np.array([40,40,40])

    # DF overlap
    auxcell=format_aux_basis(cell, auxbasis=cell.basis)
    c3=aux_e2(cell, auxcell, 'cint3c1e_sph')
    c3grid=aux_e2_grid(cell, auxcell, gs)

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

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    Lunit = 4.
    Ly = Lz = Lunit
    Lx = Lunit

    h = np.diag([Lx,Ly,Lz])
    
    #mol.atom.extend([['He', (0.5*Lx*B, 0.5*Ly*B, 0.5*Lz*B)]])

    mol.build(
        verbose = 0,
        atom = '''He     0    0.       0.
        #He     1    0.       0.
        #He     0    1.       0.
        #He     0    0.       1.
        #He     1    1.       0.
        #He     1    0.       1.
        ''',
        basis={'He': [[0, (1.0, 1.0)]]})
    #basis={'He':'sto-3g'})


    # these are some exponents which are 
    # not hard to integrate
    #mol.basis = { 'He': [[0, (1.0, 1.0)]] }
    #mol.unit='A'
    #mol.build()

    cell = pgto.Cell()
    cell.__dict__.update(mol.__dict__) # hacky way to make a cell
    cell.h = h
    #cell.vol = scipy.linalg.det(cell.h)
    #cell.nimgs = cell.get_nimgs(cell, 1.e-7)
    n = 30
    cell.gs = np.array([n,n,n])
    cell.pseudo = None
    cell.output = None
    cell.verbose = 0
    cell.build()

    #bas=genbas(2., 1.8,(1000.,0.3), 0)+
    bas=genbas(2., 1.8,(10.,1.0), 0)+genbas(1., 1.8,(10.,1.0), 1)+genbas(2., 1.8,(10.,1.0), 2)
    #bas="weigend"
    #bas=[[0, (1.0, 1.0)], [0, (2.0, 1.0)]]
    #print "basis", bas
    auxcell=format_aux_basis(cell, {'He': bas })

    #print "NIMGS", auxcell.nimgs, cell.nimgs

    gs = np.array([40,40,40])
    #ew_eta, ew_cut = pbc.ewald_params(cell, gs, 1.e-7)
    #mf=scf.RKS(cell, gs, ew_eta, ew_cut)
    #mf.scf()

    #dm=mf.make_rdm1()
    # DF overlap
    ovlp1=pscf.hf.get_ovlp(cell)#, gs)
    print ovlp1, 'o'
    ovlp=scfint.get_ovlp(cell)
    print ovlp, 'o'
    dm=2*scipy.linalg.inv(ovlp) 
    exit()

    print "DM", dm
    nelec=np.einsum('ij,ij',dm,ovlp)
    print "nelec", nelec

    j=pscf.hf.get_j(cell, dm, gs)
    ref_j=np.einsum('ij,ij',j,dm)
    print "Coulomb", ref_j

    nao=dm.shape[0]
    c3=aux_e2(cell, auxcell, 'cint3c1e_sph').reshape(nao,nao,-1)
    rho=numpy.einsum('ijk,ij->k', c3, dm)

    print "rho is", rho
    norm=auxnorm(auxcell)
    rho1=rho-nelec*norm/cell.vol

    print "norm is", norm

    c2 = scfint.get_t(auxcell) * 2
    print "C2", c2
    v1 = numpy.linalg.solve(c2, rho1*4*numpy.pi)
    #v2 = numpy.einsum('ijk,k->ij', c3, v1)
    
    print numpy.dot(v1,rho)  
    print numpy.dot(v1,rho1) # Integral with the chargeless density. This
                             # should agree with the k-space Coulomb
                             # sum which omits K=0
    print numpy.dot(v1,rho)-ref_j # should be close to zero

test_poisson()
