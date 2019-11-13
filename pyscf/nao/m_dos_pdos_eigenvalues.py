import numpy as np

def eigen_dos(ksn2e, zomegas, nkpoints=1): 
  """ Compute the Density of States using the eigenvalues """
  dos = np.zeros(len(zomegas))
  for iw,zw in enumerate(zomegas): dos[iw] = (1.0/(zw - ksn2e)).sum().imag
  return -dos/np.pi/nkpoints
  

def eigen_pdos(mf, ksn2e, zomegas, nkpoints=1): 
  """ Compute the Partial Density of States using the eigenvalues """
  jmx = mf.ao_log.jmx   #varies over L
  jksn2w = np.zeros([jmx+1]+list(ksn2e.shape))
  over = mf.overlap_lil().toarray()
  orb2j = mf.get_orb2j()
  for j in range(jmx+1):
    mask = (orb2j==j)    
    for k in range (ksn2e.shape[0]):
      for s in range(mf.nspin):
        for n in range(mf.norbs):
          jksn2w[j,k,s,n] = np.dot( np.dot(mask*mf.mo_coeff_gw[k,s,n,:,0], over), mf.mo_coeff_gw[k,s,n,:,0])
  
  pdos = np.zeros((mf.nspin,jmx+1,len(zomegas)))
  for s in range(mf.nspin):
      for j in range(jmx+1):
        for iw,zw in enumerate(zomegas):
          pdos[s,j,iw] = (jksn2w[j,:,s,:]/(zw - ksn2e[:,s,:])).sum().imag 
  return -pdos/np.pi/nkpoints


def read_qp_molgw (mf, filename):
     '''reading QP energies from MOLGW output for DFT calculations'''
     with open(filename) as f:
         if (mf.nspin != int(f.readline())): raise NameError('incompatibility in no.spin')
         if (mf.norbs != int(f.readline())): raise NameError('incompatibility in no.orbitals! check basis!')
         with open (filename+'.txt','w') as f1:
             for line in f.readlines()[0:]: #first two lines are already read
                 f1.write(line)
     data = np.loadtxt('ENERGY_QP.txt')
     qp = np.zeros((1,mf.nspin, mf.norbs),dtype=float)
     for s in range (mf.nspin):
         qp[0,s,:] = data[:,s+1]
     import os
     os.remove(filename+'.txt')
     return qp


def plot (mf, d_qp = None, show = None):
    """This plots spin-polarized DOS and PDOS for calculations in both mean-field and GW levels"""

    if not hasattr(mf, 'mo_energy_gw'): print('First kernel_gw() must be run!')
    if d_qp is None:
        d_qp= mf.mo_energy_gw 
    elif (d_qp =='MOLGW'):
        d_qp = read_qp_molgw (mf,filename='./ENERGY_QP')       

    #reading mean-field from Pyscf
    d_mf = mf.mo_energy

    #reading QP energies from PySCF for GW@UHF
    #d_qp = mf.mo_energy_gw

    #reading QP energies from output of MOLGW for GW@DFT
    #d_qp = read_qp_molgw ('ENERGY_QP')

    #mesh along X
    omegas = np.arange(-1.0, 1.0, 0.005)+1j*0.01
    fermi = mf.fermi_energy*27.2114

    #Total DOS
    dos_qp = np.zeros((mf.nspin, len(omegas)))
    dos_mf = np.zeros((mf.nspin, len(omegas)))
    for s in range (mf.nspin):
        dos_qp [s,:] = eigen_dos(d_qp[0,s],omegas)
        dos_mf [s,:] = eigen_dos(d_mf[0,s],omegas)

    data=np.zeros((len(omegas), 5))
    data[:,0] = omegas.real*27.2114
    data[:,1] = dos_mf[0].clip(min=0)
    data[:,2] = dos_qp[0].clip(min=0)
    data[:,3] = dos_mf[1].clip(min=0)
    data[:,4] = dos_qp[1].clip(min=0)
    np.savetxt('dos_eigen.txt', data, fmt='%9.5f', header='  Energy(eV)\t mf_UP\t G0W0_UP\t | mf_DN\t G0W0_DN', footer='Fermi energy = {:5f}.'.format(fermi))
    #plotting DOS
    if (show==True): dosplot(filename = None, data = data, fermi = mf.fermi_energy*27.2114)
 

    #PDOS
    pdos_qp= eigen_pdos(mf, d_qp, omegas)
    pdos_mf= eigen_pdos(mf, d_mf, omegas)
    size = pdos_qp.shape[1] #s,p,d,f,g,...
    data=np.zeros(((size*4)+1, pdos_qp.shape[-1]))
    data[0,:] = omegas.real*27.2114
    data[1:size+1, :] = pdos_qp[0].clip(min=0)          #QP PDOS spin-up
    data[size+1:2*size+1, :] = pdos_mf[0].clip(min=0)   #mean-field PDOS spin-up
    data[2*size+1:3*size+1, :] = pdos_qp[1].clip(min=0) #QP PDOS spin-dn
    data[3*size+1:, :] = pdos_mf[1].clip(min=0)         #mean-field PDOS spin-dn
    np.savetxt('pdos_eigen.txt', data.T, fmt='%9.5f', header='  Energy(eV)\t QP_s_UP\tQP_p_UP\tQP_d_UP\t... MF_s_UP\tMF_p_UP\tMF_d_UP\t... | QP_s_DN\tQP_p_DN\tQP_d_DN\t... MF_s_DN\t MF_p_DN\t MF_d_DN...', footer= 'Fermi energy is {:5f} and {}-angular momentums are resolved.'.format(fermi, size))
    #plotting PDOS
    if (show==True): pdosplot(filename = None, data = data, size = size, fermi = mf.fermi_energy*27.2114)



def dosplot (filename = None, data = None, fermi = None):
    if (filename is not None): data = np.loadtxt(filename)
    elif (data is not None): data = data

    import matplotlib.pyplot as plt
    from matplotlib import rc
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(data.T[0], data.T[1], label='MF Spin-UP', linestyle=':',color='r')
    plt.fill_between(data.T[0], 0, data.T[1], facecolor='r',alpha=0.1, interpolate=True)
    plt.plot(data.T[0], data.T[2], label='QP Spin-UP',color='r')
    plt.fill_between(data.T[0], 0, data.T[2], facecolor='r',alpha=0.5, interpolate=True)
    plt.plot(data.T[0],-data.T[3], label='MF Spin-DN', linestyle=':',color='b')
    plt.fill_between(data.T[0], 0, -data.T[3], facecolor='b',alpha=0.1, interpolate=True)
    plt.plot(data.T[0],-data.T[4], label='QP Spin-DN',color='b')
    plt.fill_between(data.T[0], 0, -data.T[4], facecolor='b',alpha=0.5, interpolate=True)
    if (fermi!=None): plt.axvline(x=fermi ,color='k', linestyle='--') #label='Fermi Energy'
    plt.axhline(y=0,color='k')
    plt.title('Total DOS', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=15) 
    plt.ylabel('Density of States (electron/eV)', fontsize=15)
    plt.legend()
    plt.savefig("dos_eigen.svg", dpi=900)
    plt.show()


def pdosplot (filename = None, data = None, size = None,  fermi = None):
    if (filename is not None): data = np.loadtxt(filename).T
    elif (data is not None): data = data
    if (size is None): print('Please give number of resolved angular momentum!')
    if (fermi is None): print ('Please give fermi energy')


    import matplotlib.pyplot as plt
    from matplotlib import rc
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    orb_name = ['$s$','$p$','$d$','$f$','$g$','$h$','$i$','$k$']
    orb_colo = ['r','g','b','y','k','m','c','w']
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
    plt.axvline(x=fermi, color='k', linestyle='--') #label='Fermi Energy'
    plt.axhline(y=0,color='k')
    plt.title('PDOS', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=15) 
    plt.ylabel('Projected Density of States (electron/eV)', fontsize=15)
    plt.legend()
    plt.savefig("pdos.svg", dpi=900)
    plt.show()


if __name__=='__main__':
    '''
    This computes spin-polarized DOS and PDOS of CN molecule based on the eigenvelues obtained by 
    PySCF for GW@UHF or MOLGW for GW@DFT and plots DOS and PDOS diagrams.
    '''
    from pyscf import gto, scf, dft
    from pyscf.dft import UKS
    from pyscf.nao import gw as gw_c
    mol = gto.M( verbose = 1, atom ='C 0.0, 0.0, -0.611046 ; N 0.0, 0.0, 0.523753' , basis = 'cc-pvdz', spin=1, charge=0)
    mol.build()
    mf = scf.UHF(mol)
    mf.kernel()
    #mf=UKS(mol).run(conv_tol=1e-14)    #mo_coeff_gw = mo_coeff in G0W0
    #mf.xc='b3lyp'
    #mf.scf()

    gw = gw_c(mf=mf, gto=mol, verbosity=1, niter_max_ev=1)  
    gw.kernel_gw()
    plot(mf=gw, show=True) 
