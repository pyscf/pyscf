from __future__ import print_function, division
import numpy as np
from numpy import zeros, dot

def omask2wgts_loops(mf, omask, over):
  """ Finds weights """
  ksn2w = zeros(mf.mo_energy.shape[:])
  for k in range(mf.nkpoints):
    for s in range(mf.nspin):
      for n in range(mf.norbs):
        ksn2w[k,s,n] = dot( dot(omask*mf.mo_coeff[k,s,n,:,0], over), mf.mo_coeff[k,s,n,:,0])
  return ksn2w


def gdos(mf, zomegas, omask=None, mat=None, nkpoints=1):
  """ Compute some masked (over atomic orbitals) or total Density of States or any population analysis """
  mat = mf.hsx.s4_csr.toarray() if mat is None else mat
  omask = np.ones(mf.norbs) if omask is None else omask
  ksn2w = omask2wgts_loops(mf, omask, mat)
  gdos = zeros(len(zomegas))
  for iw,zw in enumerate(zomegas):
    gdos[iw] = (ksn2w[:,:,:]/(zw - mf.mo_energy[:,:,:])).sum().imag

  return -gdos/np.pi/nkpoints


def lsoa_dos(mf, zomegas, lsoa=None, nkpoints=1): 
  """ Compute the Partial Density of States according to a list of atoms """
  lsoa = range(mf.natoms) if lsoa is None else lsoa

  mask = zeros(mf.norbs)
  for a in lsoa: mask[mf.atom2s[a]:mf.atom2s[a+1]] = 1.0

  #over = mf.hsx.s4_csr.toarray()
  if hasattr(mf, 'hsx') :
    over = mf.hsx.s4_csr.toarray()
  else:
    over = mf.overlap_lil().toarray()
  dos = gdos(mf, zomegas, mask, over, nkpoints)
  return dos


def pdos(mf, zomegas, nkpoints=1): 
  """ Compute the Partial Density of States (resolved in angular momentum of the orbitals) using the eigenvalues and eigenvectors in wfsx """
  
  jmx = mf.ao_log.jmx
  if hasattr(mf, 'hsx') :
    over = mf.hsx.s4_csr.toarray()
  else:
    over = mf.overlap_lil().toarray()

  orb2j = mf.get_orb2j()
  
  pdos = zeros((jmx+1,len(zomegas)))
  for j in range(jmx+1):
    mask = (orb2j==j)
    pdos[j] = gdos(mf, zomegas, mask, over, nkpoints)

  return pdos

#
# Example of plotting DOS calculated by GW calculation.
#
if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from pyscf import gto, scf
    from pyscf.nao import gw as gw_c

    mol = gto.M( verbose = 0, atom = '''C 0.0, 0.0, -0.611046 ; N 0.0, 0.0, 0.523753''', basis = 'cc-pvdz', spin=1, charge=0)
    gto_mf_UHF = scf.UHF(mol)
    gto_mf_UHF.kernel()
    gw = gw_c(mf=gto_mf_UHF, gto=mol, verbosity=1, niter_max_ev=20)  
    omegas = np.arange(-1.0, 1.0, 0.005)+1j*0.01
    dos= lsoa_dos(mf=gw, zomegas=omegas)
    pdos= pdos(mf=gw, zomegas=omegas)
    data=np.zeros((pdos.shape[0]+2, pdos.shape[1]))
    data[0,:] = omegas.real*27.2114
    data[1, :] = dos.clip(min=0)
    data[2:, :] = pdos.clip(min=0)
    np.savetxt('dos.dat', data.T, fmt='%14.6f', header='  Energy(eV)\t     Total DOS\t    s_state\t   p_state\t  d_state')

    #plotting DOS and PDOS
    x = data.T [:,0]    #Energies
    y1 = data.T [:,1]   #Total DOS
    y2 = data.T [:,2]   #s_state
    y3 = data.T [:,3]   #p_state
    y4 = data.T [:,4]   #d_state
    plt.plot(x, y1, label='Total DOS')
    plt.plot(x, y2, label='s_state')
    plt.plot(x, y3, label='p_state')
    plt.plot(x, y4, label='d_state')
    plt.axvline(x=gw.fermi_energy*27.2114,color='k', linestyle='--', label='Fermi Energy')
    plt.title('DOS', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=15) 
    plt.ylabel('Density of States (electron/eV)', fontsize=15)
    plt.legend()
    plt.show()

    #plots DOS for each atoms 
    for i in range (gw.natoms):
        local_dos= lsoa_dos(mf=gw, zomegas=omegas,lsoa=[i])
        data[1,:] = local_dos.clip(min=0)
        plt.plot(x, data.T [:,1], label='Local DOS of atom '+gw.sp2symbol[i])
    plt.xlabel('Energy (eV)', fontsize=15) 
    plt.axvline(x=gw.fermi_energy*27.2114,color='k', linestyle='--', label='Fermi Energy')
    plt.ylabel('Local Density of States (electron/eV)', fontsize=12)
    plt.legend()
    plt.show()

