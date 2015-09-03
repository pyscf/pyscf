import itertools
import math
import numpy as np
import numpy.fft
import scipy.linalg
import pyscf.gto.mole
import pyscf.dft.numint


pi=math.pi
sqrt=math.sqrt
exp=math.exp

'''PBC module. Notation follows Marx and Hutter (MH), "Ab Initio Molecular Dynamics"
'''

class Cell:
    def __init__(self, **kwargs):
        self.mol = None
        self.h = None
        self.vol = 0.

def get_Gv(cell, gs):
    '''
    cube of G vectors (Eq. (3.8), MH),

    Indices(along each direction go as
    [0...gs, -gs...-1]
    to follow FFT convention

    Returns 
        np.array([3, ngs], np.float64)
    '''
    invhT=scipy.linalg.inv(cell.h.T)

    # maybe there's a better numpy way?
    gxrange=range(gs[0]+1)+range(-gs[0],0)
    gyrange=range(gs[1]+1)+range(-gs[1],0)
    gzrange=range(gs[2]+1)+range(-gs[2],0)
    gxyz=np.array([gxyz for gxyz in 
                   itertools.product(gxrange, gyrange, gzrange)])

    Gv=2*pi*np.dot(invhT,gxyz.T)

    return Gv

def get_SI(cell, Gv):
    '''
    structure factor (Eq. (3.34), MH)

    Returns 
        np.array([natm, ngs], np.complex128)
    '''
    ngs=Gv.shape[1]
    mol=cell.mol
    SI=np.empty([mol.natm, ngs], np.complex128)
    mol=cell.mol

    for ia in range(mol.natm):
        SI[ia,:]=np.exp(-1j*np.dot(Gv.T, mol.atom_coord(ia)))

    return SI

def ewald_rs(cell, gs, ew_eta, ew_cut):
    '''
    Real-space Ewald sum 

    Formulation of Martin, App. F2.

    Returns
        float
    '''
    mol=cell.mol
    chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
    coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

    ewovrl = 0.
    # real-space sum
    for (ix, iy, iz) in np.ndindex(ew_cut):
        L=ix*cell.h[:,0]+iy*cell.h[:,1]+iz*cell.h[:,2]

        # prime in summation to avoid self-interaction in unit cell
        for ia in range(mol.natm):
            qi = chargs[ia]
            ri = coords[ia]

            if ix == 0 and iy == 0 and iz == 0:
                for ja in range(ia):
                    qj = chargs[ja]
                    rj = coords[ja]
                    r = np.linalg.norm(ri-rj)
                    ewovrl += qi * qj / r * math.erfc(ew_eta * r)
            else:
                #print "hello", ix,iy,iz, ewovrl
                for ja in range(mol.natm):
                    qj=chargs[ja]
                    rj=coords[ja]

                    r=np.linalg.norm(ri-rj+L)
                    ewovrl += qi * qj / r * math.erfc(ew_eta * r)
        
    # last line of Eq. (F.5) in Martin 
    ewself  = -1./2. * np.dot(chargs,chargs) * 2 * ew_eta / sqrt(pi)
    ewself += -1./2. * np.sum(chargs)**2 * pi/(ew_eta**2 * cell.vol)
    
    # g-space sum (using g grid) (Eq. (F.6) in Martin 
    # - note must include 4pi/cell.vol)
    ewg=0.
    Gv=get_Gv(cell, gs)
    SI=get_SI(cell, Gv)

    coulG=get_coulG(cell, gs)
    print Gv.shape
    print SI.shape
    for ia in range(mol.natm):
        tmp=np.empty(Gv.shape[1])
        for ig in range(Gv.shape[1]):
            tmp[ig]=SI[ia,ig]*exp(-np.linalg.norm(Gv[:,ig])**2/(8*ew_eta**2))

        ewg+=np.sum(np.array(tmp)*coulG)
    ewg/=cell.vol
    
    return ewovrl + ewself + ewg




# def ewald_rs(cell, ewrc, ewcut):
#     '''
#     Real-space Ewald sum 

#     This implements Eq. (3.46), Eq. (3.47) in MH, but 
#     has only a single Ewald length (Rc is same for all ions).
#     This is equivalent to the formulation in Martin, App. F.3,
#     and should be equivalent to the uniform background formulae of Martin, App. F.2

#     ewrc : Ewald length (Rc param in MH)
#     ewcut : [ewcutx, ewcuty, ewcutz]

#     Returns
#         float
#     '''
#     mol=cell.mol
#     chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
#     coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

#     ewovrl = 0.
#     for (ix, iy, iz) in np.ndindex(ewcut):
#         L=ix*cell.h[:,0]+iy*cell.h[:,1]+iz*cell.h[:,2]

#         # prime in summation to avoid self-interaction in unit cell
#         for ia in range(mol.natm):
#             qi = chargs[ia]
#             ri = coords[ia]

#             if ix == 0 and iy == 0 and iz == 0:
#                 for ja in range(ia):
#                     qj = chargs[ja]
#                     rj = coords[ja]
#                     r = np.linalg.norm(ri-rj)
#                     ewovrl += qi * qj / r * math.erfc(r / math.sqrt(2*ewrc**2))
#             else:
#                 for ja in range(mol.natm):
#                     qj=chargs[ja]
#                     rj=coords[ja]

#                     r=np.linalg.norm(ri-rj+L)
#                     ewovrl += qi * qj / r * math.erfc(r / math.sqrt(2*ewrc**2))
        
#     ewself = 1./(math.sqrt(2*pi) * ewrc) * np.dot(chargs,chargs)
    
#     return ewovrl - ewself

def gen_qv(ngs):
    '''
    integer cube of indices, 0...ngs-1 along each direction
    ngs: [ngsx, ngsy, ngsz]

    Returns 
         3 * (ngsx*ngsy*ngsz) matrix
         [0, 0, ... ngsx-1]
         [0, 0, ... ngsy-1]
         [0, 1, ... ngsz-1]
    '''
    return np.array(list(np.ndindex(tuple(ngs)))).T

def setup_ao_grids(cell, gs):
    '''
    Real-space AO uniform grid, following Eq. (3.19) (MH)
    '''
    mol=cell.mol
    ngs=2*gs+1
    qv=gen_qv(ngs)
    invN=np.diag(1./np.array(ngs))
    R=np.dot(np.dot(cell.h, invN), qv)
    coords=R.T.copy() #pyscf notation for grids (also make C-contiguous with copy)
    weights=np.ones(coords.shape)
    return coords, weights

def get_aoR(cell, coords):
    aoR=pyscf.dft.numint.eval_ao(cell.mol, coords)
    return aoR

def get_rhoR(cell, aoR, dm):
    rhoR=pyscf.dft.numint.eval_rho(cell.mol, aoR, dm)
    return rhoR

def fft(f, gs):
    '''
    3D FFT R to G space.
    
    f is evaluated on a real-space grid, assumed flattened
    to a 1d array corresponding to the index order of gen_q, where q=(u, v, w)
    u = range(0, ngs[0]), v = range(0, ngs[1]), w = range(0, ngs[2]).
    
    (re: Eq. (3.25), we assume Ns := ngs = 2*gs+1)

    After FT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv

    Returns
        FFT 1D array in same index order as Gv (natural order of numpy.fft)

    Note: FFT norm factor is 1., as in MH and in numpy.fft
    '''
    ngs=2*gs+1
    f3d=np.reshape(f, ngs)
    g3d=np.fft.fftn(f3d)
    return np.ravel(g3d)

def ifft(g, gs):
    '''
    Note: invFFT norm factor is 1./N - same as numpy but **different** 
    from MH (they use 1.)
    '''
    ngs=2*gs+1
    g3d=np.reshape(g, ngs)
    f3d=np.fft.ifftn(g3d)
    return np.ravel(f3d)

def get_nuc(cell, gs, ewrc=1.e-6):
    '''
    Bare nuc-el AO matrix (G=0 component removed)
    '''
    mol=cell.mol

    chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
    nuc_coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

    coords, weights=setup_ao_grids(cell,gs)
    aoR=get_aoR(cell, coords)

    vneR=np.zeros(aoR.shape[0])

    for a in range(mol.natm):
        qa = chargs[a]
        ra = nuc_coords[a]
        riav = [np.linalg.norm(ra-ri) for ri in coords]
        #vneR-=np.array([qa/ria*math.erf(ria/ewrc)
        #                for ria in riav]) # Eq. (3.40) MH
        vneR-=np.array([qa/ria for ria in riav])

    # Set G=0 component to 0.
    vneG=fft(vneR, gs)
    vneG[0]=0.
    vneR=ifft(vneG,gs)

    nao=aoR.shape[1]
    vne=np.zeros([nao, nao])
    for i in range(nao):
        for j in range(nao):
            vne[i,j]=cell.vol/aoR.shape[0]*np.vdot(aoR[:,i],vneR*aoR[:,j])

    return vne

def get_t(cell, gs):
    '''
    Kinetic energy AO matrix
    '''
    Gv=get_Gv(cell, gs)
    G2=np.array([np.inner(Gv[:,i], Gv[:,i]) for i in xrange(Gv.shape[1])])

    coords, weights=setup_ao_grids(cell, gs)
    aoR=get_aoR(cell, coords)
    aoG=np.empty(aoR.shape, np.complex128)
    TaoG=np.empty(aoR.shape, np.complex128)
    #TaoR=np.empty(aoR.shape, np.complex128)

    nao=aoR.shape[1]
    for i in range(nao):
        aoG[:,i]=fft(aoR[:,i], gs)
        TaoG[:,i]=0.5*G2*aoG[:,i]
        #TaoR[:,i]=ifft(TaoG[:,i], gs)
                
    t=np.empty([nao,nao])
    # for i in range(nao):
    #     for j in range(nao):
    #         t[i,j]=np.vdot(aoR[:,i],TaoR[:,j])*cell.vol/aoR.shape[0]
    for i in range(nao):
        for j in range(nao):
            t[i,j]=np.vdot(aoG[:,i],TaoG[:,j])*cell.vol/(aoR.shape[0])**2

    return t

def get_ovlp(cell, gs):
    '''
    Overlap AO matrix
    '''
    coords, weights=setup_ao_grids(cell, gs)
    aoR=get_aoR(cell, coords)
    aoG=np.empty(aoR.shape, np.complex128)
    nao=aoR.shape[1]
    for i in range(nao):
        aoG[:,i]=fft(aoR[:,i], gs)

    s=np.empty([nao,nao])
    for i in range(nao):
        for j in range(nao):
            s[i,j]=cell.vol/len(aoR)*np.vdot(aoR[:,i],aoR[:,j])
    return s
    
def get_coulG(cell, gs):
    '''
    Coulomb kernel in G space (4*pi/G^2 for G!=0, 0 for G=0)
    '''
    Gv=get_Gv(cell, gs)

    coulG=np.zeros(Gv.shape[1]) 

    # must be better way to code this loop !!
    # keep coulG[0]=0.0
    coulG[1:]=4*pi/np.array([np.inner(Gv[:,i], Gv[:,i]) 
                             for i in xrange(1, Gv.shape[1])])
    
    return coulG

def get_j(cell, dm, gs):
    '''
    Coulomb AO matrix 
    '''
    coulG=get_coulG(cell, gs)

    coords, weights=setup_ao_grids(cell, gs)
    aoR=get_aoR(cell, coords)

    rhoR=get_rhoR(cell, aoR, dm)
    rhoG=fft(rhoR, gs)

    vG=coulG*rhoG
    vR=ifft(vG, gs)

    nao=aoR.shape[1]
    vj=np.zeros([nao,nao])
    for i in range(nao):
        for j in range(nao):
            vj[i,j]=cell.vol/aoR.shape[0]*np.dot(aoR[:,i],vR*aoR[:,j])
           
    return vj

    

# def ecoul(cell, dm, gs, ewrc, ewcut):
#     '''
#     Combined e-e, e-n Coulomb energy using Ewald sums (Eq. (3.45) MH)
    
#     ******** this function is broken ********
#     Returns
#         float
#     '''
#     Gv=get_Gv(cell, gs)
#     SI=get_SI(cell, Gv)

#     coulG=get_coulG(cell, gs)

#     coords, weights=setup_ao_grids(cell, gs)
#     ngs=len(weights)
#     aoR=get_aoR(cell, coords)
#     rhoR=get_rhoR(cell, aoR, dm)
#     rhoG=fft(rhoR, gs)

#     rhocG=np.zeros(rhoG.shape, np.complex128)
#     mol=cell.mol
#     chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
#     print "charges", chargs
#     for ia in range(mol.natm):
#         qi = chargs[ia]
#         rhocG-=np.array([qi/sqrt(4*pi)*
#                          exp(-0.25*np.linalg.norm(Gv[:,g])**2 * ewrc**2)
#                          *SI[ia,g] for g in range(SI.shape[1])])
    
#     #rho=rhoG+1./cell.vol*rhocG ## what is the right factor?
#     rho=rhoG+rhocG
#     #rho=rhoG
    
#     e_g=.5*cell.vol*np.sum(coulG*np.abs(rho)**2)/ngs**2

#     e_rs=ewald_rs(cell, ewrc, ewcut)
#     print e_g, e_rs
#     return e_g+e_rs

def test():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    L=40
    h=np.eye(3.)*L


    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    #mol.basis = { 'He': 'STO-3G'}
    mol.basis = { 'He': [[0,
                          (1.0, 1.0)]] }
#        (0.2089, 0.307737),]] }
    
    mol.build()
    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
    
    cell=Cell()
    cell.mol=mol
    cell.h=h
    
    cell.vol=scipy.linalg.det(cell.h)
    gs=np.array([10,10,10]) # number of G-points in grid. Real-space dim=2*gs+1
    Gv=get_Gv(cell, gs)

    # SI=get_SI(cell, Gv)
    # ewrc=0.5
    # ewcut=(1,1,1)
    # ew_rs=ewald_rs(cell, ewrc, ewcut)

    coords, weights=setup_ao_grids(cell, gs)

    aoR=get_aoR(cell, coords)
    print aoR.shape

    dm=m.make_rdm1()

    # print "Kinetic"
    # tao=get_t(cell, gs)
    # print tao

    # tao2 = mol.intor_symmetric('cint1e_kin_sph') 
    # print tao2

    # print "kinetic energies"
    # print np.dot(np.ravel(tao), np.ravel(dm))
    # print np.dot(np.ravel(tao2), np.ravel(dm))
    # sao=get_ovlp(cell,gs)

    # print "Overlap"
    # print sao
    # print m.get_ovlp()
    # print np.dot(np.ravel(sao), np.ravel(dm))
    # print np.dot(np.ravel(m.get_ovlp()), np.ravel(dm))

    # print "Coulomb"
    jao=get_j(cell,dm,gs)
    print jao
    print m.get_j(dm)
    # # print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))
    # print "Coulomb energy", .5*np.dot(np.ravel(dm),np.ravel(jao))

    print "Nuc-el"
    neao=get_nuc(cell,gs)
    # print neao
    vne=mol.intor_symmetric('cint1e_nuc_sph') 
    # print vne
    print np.dot(np.ravel(dm), np.ravel(vne))
    print np.dot(np.ravel(dm), np.ravel(neao))
    print "Normalization"
    rhoR=get_rhoR(cell, aoR, dm)

    print "(Hartree + vne) * DM"
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne))
    print np.einsum("ij,ij",dm,neao+jao)

    # # print cell.vol
    # # print rhoR.shape
    # # # should be 2.0, gets 1.99004869382. With 27 million pts!!
    print cell.vol/len(rhoR)*np.sum(rhoR) 

    # print "total coulomb"

    ewcut=(10,10,10)

    # print ecoul(cell, dm, gs, ewrc, ewcut)
    for ew_eta in [0.5, 1., 2.]:

        ew=ewald_rs(cell, gs, ew_eta, ewcut)
        print "ewald divergent terms summation", ew

    print "actual coulomb", .5*np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne))

    print "fft coul + ewald", np.einsum("ij,ij",dm,neao+.5*jao)+ew
