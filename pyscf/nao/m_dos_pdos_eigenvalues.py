import numpy as np
from numpy import zeros_like, zeros 

def eigen_dos(ksn2e, zomegas, nkpoints=1): 
  """ Compute the Density of States using the eigenvalues """
  dos = np.zeros(len(zomegas))
  for iw,zw in enumerate(zomegas): dos[iw] = (1.0/(zw - ksn2e)).sum().imag
  return -dos/np.pi/nkpoints
  

def eigen_pdos(ksn2e, zomegas, nkpoints=1): 
  """ Compute the Partial Density of States using the eigenvalues """
  jmx = gw.ao_log.jmx   #varies over L
  jksn2w = np.zeros([jmx+1]+list(ksn2e.shape))
  over = gw.overlap_lil().toarray()
  orb2j = gw.get_orb2j()
  for j in range(jmx+1):
    mask = (orb2j==j)    
    for k in range (ksn2e.shape[0]):
      for s in range(gw.nspin):
        for n in range(gw.norbs):
          jksn2w[j,k,s,n] = np.dot( np.dot(mask*gw.mo_coeff_gw[k,s,n,:,0], over), gw.mo_coeff_gw[k,s,n,:,0])
  
  pdos = np.zeros((gw.nspin,jmx+1,len(zomegas)))
  for s in range(gw.nspin):
      for j in range(jmx+1):
        for iw,zw in enumerate(zomegas):
          pdos[s,j,iw] = (jksn2w[j,:,s,:]/(zw - ksn2e[:,s,:])).sum().imag 
  return -pdos/np.pi/nkpoints


def read_qp_molgw (filename):
     '''reading QP energies from MOLGW output for DFT calculations'''
     with open(filename) as f:
         if (gw.nspin != int(f.readline())): raise NameError('incompatibility in no.spin')
         if (gw.norbs != int(f.readline())): raise NameError('incompatibility in no.orbitals! check basis!')
         with open (filename+'.txt','w') as f1:
             for line in f.readlines()[0:]: #first two lines are already read
                 f1.write(line)
     data = np.loadtxt('ENERGY_QP.txt')
     qp = np.zeros((1,gw.nspin, gw.norbs),dtype=float)
     for s in range (gw.nspin):
         qp[0,s,:] = data[:,s+1]
     import os
     os.remove(filename+'.txt')
     return qp


def plot (d_qp=None):
    """This plot DOS and PDOS for spin-polarized calculations in both mean-field and GW levels"""
    import matplotlib.pyplot as plt
    from matplotlib import rc
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    d_qp= gw.mo_energy_gw if d_qp is None else d_qp 

    #reading mean-field from Pyscf
    d_mf = gw.mo_energy

    #reading QP energies from PySCF for GW@UHF
    #d_qp = gw.mo_energy_gw

    #reading QP energies from output of MOLGW for GW@DFT
    #d_qp = read_qp_molgw ('ENERGY_QP')

    #mesh along X
    omegas = np.arange(-1.0, 1.0, 0.005)+1j*0.01

    #Total DOS
    dos_qp = np.zeros((gw.nspin, len(omegas)))
    dos_mf = np.zeros((gw.nspin, len(omegas)))
    for s in range (gw.nspin):
        dos_qp [s,:] = eigen_dos(d_qp[0,s],omegas)
        dos_mf [s,:] = eigen_dos(d_mf[0,s],omegas)

    data=np.zeros((len(omegas), 5))
    data[:,0] = omegas.real*27.2114
    data[:,1] = dos_mf[0].clip(min=0)
    data[:,2] = dos_qp[0].clip(min=0)
    data[:,3] = dos_mf[1].clip(min=0)
    data[:,4] = dos_qp[1].clip(min=0)
    #np.savetxt('mas-dos.dat', data.T, fmt='%14.6f', header='  Energy(eV)\t     mf_UP\t     G0W0_UP\t     mf_DN\t     G0W0_DN')

    #plotting DOS
    plt.plot(data.T[0], data.T[1], label='MF Spin-UP', linestyle=':',color='r')
    plt.fill_between(data.T[0], 0, data.T[1], facecolor='r',alpha=0.1, interpolate=True)
    plt.plot(data.T[0], data.T[2], label='QP Spin-UP',color='r')
    plt.fill_between(data.T[0], 0, data.T[2], facecolor='r',alpha=0.5, interpolate=True)
    plt.plot(data.T[0],-data.T[3], label='MF Spin-DN', linestyle=':',color='b')
    plt.fill_between(data.T[0], 0, -data.T[3], facecolor='b',alpha=0.1, interpolate=True)
    plt.plot(data.T[0],-data.T[4], label='QP Spin-DN',color='b')
    plt.fill_between(data.T[0], 0, -data.T[4], facecolor='b',alpha=0.5, interpolate=True)
    plt.axvline(x=gw.fermi_energy*27.2114,color='k', linestyle='--') #label='Fermi Energy'
    plt.axhline(y=0,color='k')
    plt.title('Total DOS', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=15) 
    plt.ylabel('Density of States (electron/eV)', fontsize=15)
    plt.legend()
    #plt.savefig("dos.svg", dpi=900)
    plt.show()
    plt.clf()

    #PDOS
    pdos_qp= eigen_pdos(d_qp,omegas)
    pdos_mf= eigen_pdos(d_mf,omegas)
    size = pdos_qp.shape[1] #s,p,d,f,g,...
    print(size,'Angular momentums will be drawn.')
    data=np.zeros(((size*4)+1, pdos_qp.shape[-1]))
    data[0,:] = omegas.real*27.2114
    data[1:size+1, :] = pdos_qp[0].clip(min=0)          #QP PDOS spin-up
    data[size+1:2*size+1, :] = pdos_mf[0].clip(min=0)   #mean-field PDOS spin-up
    data[2*size+1:3*size+1, :] = pdos_qp[1].clip(min=0) #QP PDOS spin-dn
    data[3*size+1:, :] = pdos_mf[1].clip(min=0)         #mean-field PDOS spin-dn
    #np.savetxt('pdos.dat', data.T, fmt='%14.3f', header='  Energy(eV)\t QP_s_UP\t QP_p_UP\t QP_d_UP\t .... MF_s_UP\t MF_p_UP\t MF_d_UP\t .... | QP_s_DN\t QP_p_DN\t QP_d_DN\t ... MF_s_DN\t MF_p_DN\t MF_d_DN ...')
  
    #plotting PDOS
    orb_name = ['$s$','$p$','$d$','$f$','$g$','$h$','$i$','$k$']
    orb_colo = ['r','g','b','y','k','m','c']
    for i, (n,c) in enumerate(zip(orb_name[0:size],orb_colo[0:size])):
        #GW_spin_UP
        plt.plot(data[0], data[i+1], label='QP- '+n,color=c)
        plt.fill_between(data[0], 0, data[i+1], facecolor=c, alpha=0.5, interpolate=True)
        #MF_spin_UP
        plt.plot(data[0], data[i+size+1], label='MF- '+n, linestyle=':',color=c)
        plt.fill_between(data[0], 0, data[i+size+1], facecolor=c, alpha=0.1, interpolate=True)
        #GW_spin_DN
        plt.plot(data[0], -data[i+2*size+1], label='_nolegend_',color=c)
        plt.fill_between(data[0], 0, -data[i+2*size+1], facecolor=c, alpha=0.5, interpolate=True)
        #MF_spin_DN
        plt.plot(data[0], -data[i+3*size+1], label='_nolegend_', linestyle=':',color=c)
        plt.fill_between(data[0], 0, -data[i+3*size+1], facecolor=c, alpha=0.1, interpolate=True)
    plt.axvline(x=gw.fermi_energy*27.2114,color='k', linestyle='--') #label='Fermi Energy'
    plt.axhline(y=0,color='k')
    plt.title('PDOS', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=15) 
    plt.ylabel('Partial-Density of States (electron/eV)', fontsize=15)
    plt.legend()
    #plt.savefig("Pdos.svg", dpi=900)
    plt.show()


if __name__=='__main__':
    '''
    This computes DOS and PDOS of CN molecule based on the eigenvelues obtained by 
    PySCF for GW@UHF or MOLGW for GW@DFT and plots DOS and PDOS diagrams.
    '''
    from pyscf import gto, scf, dft
    from pyscf.dft import UKS
    from pyscf.nao import gw as gw_c
    mol = gto.M( verbose = 1, atom ='C 0.0, 0.0, -0.611046 ; N 0.0, 0.0, 0.523753' , basis = 'cc-pvdz', spin=1, charge=0)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()
    #mf=UKS(mol).run(conv_tol=1e-14)
    #mf.xc='b3lyp'
    #mf.scf()

    gw = gw_c(mf=mf, gto=mol, verbosity=1, niter_max_ev=20)  
    gw.kernel_gw()
    plot() 
