import numpy as np
from pyscf.fci import cistring
from pyscf.fci import direct_uhf
import time

def ft_solver(h1e,g2e,fcisolver,norb,nelec,T,mu,m=80):

    na, nb = nelec
    ne = na + nb
    ncia = cistring.num_strings(norb,na)
    ncib = cistring.num_strings(norb,nb)

    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
    def hop(c):
        hc = fcisolver.contract_2e(h2e,c,norb,nelec)
        return hc
        #return hc.reshape(-1)
    def qud(c):
        dm1, dm2 = fcisolver.make_rdm12s(c,norb,nelec)
        return np.asarray(dm1), np.asarray(dm2)

    nsamp = 10
    RDM1 = np.zeros((2,norb,norb))
    RDM2 = np.zeros((3,norb,norb,norb,norb))
    E = 0.
    Z = 0.
    for nsamp in range(nsamp):
        ci0 = np.random.randn(ncia, ncib)
        rdm1, rdm2, e, z = lanczos(ci0,norb,ne,T,mu,hop,qud,m)
        RDM1 += np.asarray(rdm1)
        RDM2 += np.asarray(rdm2)
        E    += e
        Z    += z

    return RDM1/nsamp, RDM2/nsamp, E/nsamp, Z/nsamp

#######################################################################
def lanczos(v0,norb,ne,T,mu,hop, qud, m=50,Min_b=1e-7):
    #lanczos algorithm for one initial vector v0
    tri_diag, tri_off, krylov_space = [], [], []
    # initial steps
    
    v0 = v0/np.linalg.norm(v0)
    Hv = hop(v0)
    krylov_space.append(v0)
    #tri_diag.append(np.dot(v0, Hv))
    tri_diag.append(np.sum(v0*Hv))
    v1 = Hv - tri_diag[0] * v0
    tri_off.append(np.linalg.norm(v1))
    if tri_off[0] < Min_b:
        print("V0 is an eigen function of the Hamiltonian!")
        rdm1, rdm2 = qud(v0)
        e = tri_diag[0]
        z = np.exp(-(e-mu*ne)/T)
        return rdm1, rdm2, e, z

    v1 = v1/tri_off[0]
    Hv = hop(v1)
    #tri_diag.append(np.dot(v1, Hv))
    tri_diag.append(np.sum(v1*Hv))
    krylov_space.append(v1)
    for i in range(1, m-1):
        v2 = Hv - tri_off[i-1]*v0 - tri_diag[i]*v1
        tri_off.append(np.linalg.norm(v2))
        if tri_off[i] < Min_b: # the Hamiltonian is exact
            tri_off.pop()
            break    
        v2 = v2 / tri_off[i]
        krylov_space.append(v2)
        Hv = hop(v2)
        #tri_diag.append(np.dot(v2, Hv))
        tri_diag.append(np.sum(v2*Hv))
        v0 = v1.copy()
        v1 = v2.copy()

    tri_diag, tri_off = np.asarray(tri_diag), np.asarray(tri_off)
    krylov_space = np.asarray(krylov_space) 
    lan_e, lan_v = eigh_trimat(tri_diag, tri_off)
    ave_state = np.dot(np.transpose(krylov_space,(1,2,0)).conj(), lan_v)
    coef = np.exp(-(lan_e - mu*ne)/(2.*T)) * lan_v[0,:]
    exp_e = np.exp(-(lan_e - mu*ne)/T)
    E = np.sum(exp_e * lan_e * (lan_v[0,:]**2))
    Z = np.sum(exp_e * (lan_v[0,:]**2))
    psi = np.dot(ave_state, coef.T.conj())
    RDM1, RDM2 = qud(psi)
        

    return RDM1.real, RDM2.real, E, Z
            
#######################################################################
def eigh_trimat(a1, b1):
    mat = np.diag(b1, -1) + np.diag(a1, 0) + np.diag(b1, 1)
    e, w = np.linalg.eigh(mat)
    # w[:, i] is the ith eigenvector
    return e, w


if __name__ == "__main__":
    from pyscf.fci import direct_uhf as fcisolver
    import time
    T = 0.02
    mu = 0
    norb = 4
    nelec = (2, 2)
    for na in range(0,norb+1):
        for nb in range(0,norb+1):
            print '--------------------------------'
            print nelec
            nelec = (na,nb)
            h1e = np.zeros((norb,norb))
            for i in range(norb):
                h1e[i,(i+1)%norb] = -1.
                h1e[i,(i-1)%norb] = -1.
            h1e[0,-1] = 0
            h1e[-1,0] = 0
            #h1e[0,0] = 0.1
            #h1e[1,1] = 0.4
            #noise = np.random.randn(norb, norb)*0.0
            #noise += noise.T
            #h1e += noise
            h1e = (h1e, h1e)
            eri = np.zeros((norb,norb,norb,norb))
            for i in range(norb):
                eri[i,i,i,i] = 4
            g2e = (np.zeros((norb,)*4), eri,np.zeros((norb,)*4))
            rdm1, rdm2, e, z = ft_solver(h1e,g2e,fcisolver,norb,nelec,T,mu,m=80)
            e /= z
            rdm1 /=z
            rdm2 /=z
            e0, v = fcisolver.kernel(h1e,g2e,norb,nelec,nroots=1)
            #print e0
            rdm10,rdm20 = fcisolver.make_rdm12s(v,norb,nelec)
 
        #print rdm1[0]
        #print rdm10[0]
        
            print e/norb - e0/norb
            print np.linalg.norm(rdm1-rdm10)
            print np.linalg.norm(rdm2-rdm20)
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
