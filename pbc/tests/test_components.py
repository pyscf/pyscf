import numpy as np

from pyscf import gto
from pyscf.dft import rks
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft.gen_grid import gen_uniform_grids
from pyscf.pbc.dft.numint import eval_ao, eval_rho

# These are the components to be tested:
from pyscf.pbc.scf import hf as pbchf

def test_components(pseudo=None):
    # The molecular calculation
    mol = gto.Mole()
    mol.unit = 'B'
    L = 60
    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    mol.basis = 'sto-3g'
    mol.build()

    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    print(m.scf()) # -2.90705411168
    dm = m.make_rdm1()
    
    # The periodic calculation
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.h = np.diag([L,L,L])
    cell.gs = np.array([80,80,80])
    cell.nimgs = [0,0,0]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()

    # These should match reasonably well (roughly with accuracy of normalization)
    print "Kinetic energy" 
    tao = pbchf.get_t(cell) 
    tao2 = mol.intor_symmetric('cint1e_kin_sph') 
    print np.dot(np.ravel(tao), np.ravel(dm))  # 2.82793077196
    print np.dot(np.ravel(tao2), np.ravel(dm)) # 2.82352636524
    
    print "Overlap"
    sao = pbchf.get_ovlp(cell)
    print np.dot(np.ravel(sao), np.ravel(dm)) # 1.99981725342
    print np.dot(np.ravel(m.get_ovlp()), np.ravel(dm)) # 2.0

    # The next two entries should *not* match, since G=0 component is removed
    print "Coulomb (G!=0)"
    jao = pbchf.get_j(cell, dm)
    print np.dot(np.ravel(dm),np.ravel(jao))  # 4.03425518427
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm))) # 4.22285177049

    # The next two entries should *not* match, since G=0 component is removed
    print "Nuc-el (G!=0)"
    if cell.pseudo:
        vppao = pbchf.get_pp(cell)
        print np.dot(np.ravel(dm), np.ravel(vppao)) 
    else:
        neao = pbchf.get_nuc(cell)
        print np.dot(np.ravel(dm), np.ravel(neao))  # -6.50203360062
    vne = mol.intor_symmetric('cint1e_nuc_sph') 
    print np.dot(np.ravel(dm), np.ravel(vne))   # -6.68702326551

    print "Normalization" 
    coords = gen_uniform_grids(cell)
    aoR = eval_ao(cell, coords)
    rhoR = eval_rho(cell, aoR, dm)
    print cell.vol/len(rhoR)*np.sum(rhoR) # 1.99981725342 (should be 2.0)
    
    print "(Hartree + vne) * DM"
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne))
    if cell.pseudo:
        print np.einsum("ij,ij", dm, vppao + jao + pbchf.get_jvloc_G0(cell))
    else:
        print np.einsum("ij,ij", dm, neao + jao)

    ew_cut = (40,40,40)
    ew_eta = 0.05
    for ew_eta in [0.1, 0.5, 1.]:
        ew = pbchf.ewald(cell, ew_eta, ew_cut)
        print "Ewald (eta, energy)", ew_eta, ew # should be same for all eta

    print "Ewald divergent terms summation", ew

    # These two should now match if the box is reasonably big to
    # remove images, and ngs is big.
    print "Total coulomb (analytic) ", 
    print (.5*np.dot(np.ravel(dm), np.ravel(m.get_j(dm))) 
            + np.dot(np.ravel(dm), np.ravel(vne)) ) # -4.57559738004
    if not cell.pseudo:
        print "Total coulomb (fft coul + ewald)", 
        print np.einsum("ij,ij", dm, neao + .5*jao) + ew # -4.57948259115

    # Exc
    cell.ew_eta, cell.ew_cut = ew_eta, ew_cut
    mf = pbcdft.RKS(cell)
    mf.xc = 'LDA,VWN_RPA'

    rks.get_veff(mf, cell, dm)
    print "Exc", mf._exc # -1.05967570089

if __name__ == '__main__':
    test_components()
    #test_components('gth-lda')
