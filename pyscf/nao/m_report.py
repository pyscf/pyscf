from __future__ import print_function, division
import numpy as np
from pyscf.data.nist import HARTREE2EV
import time

start_time = time.time()

def report_gw (self):
    """ Prints the energy levels of mean-field and G0W0"""
    import re
    if not hasattr(self, 'mo_energy_gw'): self.kernel_gw()
    emfev = self.mo_energy[0].T * HARTREE2EV
    egwev = self.mo_energy_gw[0].T * HARTREE2EV
    file_name= ''.join(self.get_symbols())
    # The output should be possible to write more concise...
    with open('report_'+file_name+'.out','w') as out_file:
        print('-'*30,'|G0W0 eigenvalues (eV)|','-'*30)
        out_file.write('-'*30+'|G0W0 eigenvalues (eV)|'+'-'*30+'\n')
        if self.nspin==1:
            out_file.write('Energy-sorted MO indices \t {}'.format(self.argsort[0]))
            if (np.allclose(self.argsort[0][:self.nfermi[0]],np.sort(self.argsort[0][:self.nfermi[0]]))==False):
                    print ("Warning: Swapping in orbital energies below Fermi has happened!")
                    out_file.write("\nWarning: Swapping in orbital energies below Fermi has happened!")
            print("\n   n  %14s %14s %7s " % ("E_mf", "E_gw", "occ") )
            out_file.write("\n   n  %14s %14s %7s \n" % ("E_mf", "E_gw", "occ") )
            for ie,(emf,egw,f) in enumerate(zip(emfev,egwev,self.mo_occ[0].T)):
                print("%5d  %14.7f %14.7f %7.2f " % (ie, emf[0], egw[0], f[0]) )
                out_file.write("%5d  %14.7f %14.7f %7.2f\n" % (ie, emf[0], egw[0], f[0]) )
            print('\nFermi energy        (eV):%16.7f'%(self.fermi_energy* HARTREE2EV))
            out_file.write('\nFermi energy        (eV):%16.7f\n'%(self.fermi_energy* HARTREE2EV))
            print('G0W0 HOMO energy    (eV):%16.7f' % (egwev[self.nfermi[0]-1,0]))
            out_file.write('G0W0 HOMO energy    (eV):%16.7f\n'%(egwev[self.nfermi[0]-1,0]))
            print('G0W0 LUMO energy    (eV):%16.7f' % (egwev[self.nfermi[0],0]))
            out_file.write('G0W0 LUMO energy    (eV):%16.7f\n'%(egwev[self.nfermi[0],0]))
            print('G0W0 HOMO-LUMO gap  (eV):%16.7f' %(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0]))
            out_file.write('G0W0 HOMO-LUMO gap  (eV):%16.7f\n'%(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0]))
        elif self.nspin==2:
            for s in range(2):
                out_file.write('\nEnergy-sorted MO indices for spin {}\t {}'.format(str(s+1),self.argsort[s][max(self.nocc_0t[s]-10,0):min(self.nocc_0t[s]+10, self.norbs)]))
                if (np.allclose(self.argsort[s][:self.nfermi[s]],np.sort(self.argsort[s][:self.nfermi[s]]))==False):
                    print ("Warning: Swapping in orbital energies below Fermi has happened at spin {} channel!".format(s+1))
                    out_file.write("\nWarning: Swapping in orbital energies below Fermi has happened at spin {} channel!\n".format(s+1))
            print("\n    n %14s %14s  %7s | %14s %14s  %7s" % ("E_mf_up", "E_gw_up", "occ_up", "E_mf_down", "E_gw_down", "occ_down"))
            out_file.write("\n    n %14s %14s  %7s | %14s %14s  %7s\n" % ("E_mf_up", "E_gw_up", "occ_up", "E_mf_down", "E_gw_down", "occ_down"))
            for ie,(emf,egw,f) in enumerate(zip(emfev,egwev,self.mo_occ[0].T)):
                print("%5d  %14.7f %14.7f %7.2f | %14.7f %14.7f %7.2f" % (ie, emf[0], egw[0], f[0],  emf[1], egw[1], f[1]) )
                out_file.write ("%5d  %14.7f %14.7f %7.2f | %14.7f %14.7f %7.2f\n" % (ie, emf[0], egw[0], f[0],  emf[1], egw[1], f[1]) )
            print('\nFermi energy        (eV):%16.7f'%(self.fermi_energy* HARTREE2EV))
            out_file.write('\nFermi energy        (eV):%16.7f\n'%(self.fermi_energy* HARTREE2EV))
            print('G0W0 HOMO energy    (eV):%16.7f %16.7f'%(egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1]-1,1]))
            out_file.write('G0W0 HOMO energy    (eV):%16.7f %16.7f\n'%(egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1]-1,1]))
            print('G0W0 LUMO energy    (eV):%16.7f %16.7f'%(egwev[self.nfermi[0],0],egwev[self.nfermi[1],1]))
            out_file.write('G0W0 LUMO energy    (eV):%16.7f %16.7f\n'%(egwev[self.nfermi[0],0],egwev[self.nfermi[1],1]))
            print('G0W0 HOMO-LUMO gap  (eV):%16.7f %16.7f'%(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1],1]-egwev[self.nfermi[1]-1,1]))
            out_file.write('G0W0 HOMO-LUMO gap  (eV):%16.7f %16.7f\n'%(egwev[self.nfermi[0],0]-egwev[self.nfermi[0]-1,0],egwev[self.nfermi[1],1]-egwev[self.nfermi[1]-1,1]))
        else:
            raise RuntimeError('not implemented...')
        print('G0W0 Total energy   (eV):%16.7f' %(self.etot_gw*HARTREE2EV))
        out_file.write('G0W0 Total energy   (eV):%16.7f\n'%(self.etot_gw*HARTREE2EV))
        elapsed_time = time.time() - start_time
        print('\nTotal running time is: {}\nJOB DONE! \t {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),time.strftime("%c")))
        out_file.write('\nTotal running time is: {}\nJOB DONE! \t {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),time.strftime("%c"))) 
        out_file.close


def report_mfx(self, dm1=None):
    """
    This collects the h_core, Hartree (K) and Exchange(J) and Fock expectation value with given density matrix.
    """
    #import re
    #import sys
    #file_name= ''.join(self.get_symbols())
    #sys.stdout=open('report_mf_'+file_name+'.out','w')

    if dm1 is None: dm1 = self.make_rdm1()
    if dm1.ndim==5 : dm1=dm1[0,...,0]
    assert dm1.shape == (self.nspin, self.norbs, self.norbs), "Given density matrix has wrong dimension"
    H = self.get_hcore()
    ecore = (H*dm1).sum()
    co = self.get_j()
    fock = self.get_fock()
    exp_h = np.zeros((self.nspin, self.norbs))
    exp_co = np.zeros((self.nspin, self.norbs))
    exp_x = np.zeros((self.nspin, self.norbs))
    exp_f = np.zeros((self.nspin, self.norbs))
    if self.nspin==1:
        x = -0.5* self.get_k()
        mat_h = np.dot(self.mo_coeff[0,0,:,:,0], H)   
        exp_h = np.einsum('nb,nb->n', mat_h, self.mo_coeff[0,0,:,:,0])
        mat_co = np.dot(self.mo_coeff[0,0,:,:,0], co)  
        exp_co = np.einsum('nb,nb->n', mat_co, self.mo_coeff[0,0,:,:,0])
        mat_x = np.dot(self.mo_coeff[0,0,:,:,0], x)
        exp_x = np.einsum('nb,nb->n', mat_x, self.mo_coeff[0,0,:,:,0])
        mat_f = np.dot(self.mo_coeff[0,0,:,:,0], fock)
        exp_f = np.einsum('nb,nb->n', mat_f, self.mo_coeff[0,0,:,:,0])
        print('='*20,'| Restricted HF expectation values (eV) |','='*20)
        print('%2s  %13s  %13s  %13s  %13s  %13s  %3s'%('no.','<H_core>','<K>  ','<Sigma_x>','Fock   ','MF energy ','occ'))
        for i, (a,b,c,d,e,f) in enumerate(zip(exp_h.T*HARTREE2EV,exp_co.T*HARTREE2EV, exp_x.T*HARTREE2EV,exp_f.T*HARTREE2EV,self.mo_energy.T*HARTREE2EV, self.mo_occ[0].T)):   
          if (i==self.nfermi[0]): print('-'*84)
          print(' %3d  %13.6f  %13.6f %13.6f %13.6f  %13.6f  %3d'%(i, a,b,c,d,e,f))
        Vha = 0.5*(co*dm1).sum()
        EX = 0.5*(x*dm1).sum()

    elif self.nspin==2:
        x = -self.get_k()
        cou = co[0]+co[1]
        Vha = 0.5*(cou*dm1).sum()
        for s in range(self.nspin):
          mat_h = np.dot(self.mo_coeff[0,s,:,:,0], H)   
          exp_h[s] = np.einsum('nb,nb->n', mat_h, self.mo_coeff[0,s,:,:,0])
          mat_co = np.dot(self.mo_coeff[0,s,:,:,0], cou)  
          exp_co[s] = np.einsum('nb,nb->n', mat_co, self.mo_coeff[0,s,:,:,0])
          mat_x = np.dot(self.mo_coeff[0,s,:,:,0], x[s])
          exp_x[s] = np.einsum('nb,nb->n', mat_x, self.mo_coeff[0,s,:,:,0])
          mat_f = np.dot(self.mo_coeff[0,s,:,:,0], fock[s])
          exp_f[s] = np.einsum('nb,nb->n', mat_f, self.mo_coeff[0,s,:,:,0])
        print('='*59,'| Unrestricted HF expectation values (eV) |','='*60)
        print('%2s  %13s  %13s  %13s  %13s  %13s  %3s |%13s  %13s  %13s  %13s  %13s  %3s '%('no.','<H_core>','<K>  ','<Sigma_x>','Fock   ','MF energy ','occ','<H_core>','<K>   ','<Sigma_x>','Fock  ','MF energy','occ'))
        for i , (a,b,c,d,e,f) in enumerate(zip(exp_h.T*HARTREE2EV,exp_co.T*HARTREE2EV, exp_x.T*HARTREE2EV,exp_f.T*HARTREE2EV,self.mo_energy.T*HARTREE2EV, self.mo_occ[0].T)):
          if (i==self.nfermi[0] or i==self.nfermi[1]): print('-'*163)
          print(' %3d  %13.6f  %13.6f %13.6f %13.6f  %13.6f  %3d  | %13.6f  %13.6f  %13.6f  %13.6f %13.6f  %3d'%(i, a[0],b[0],c[0],d[0],e[0],f[0],a[1],b[1],c[1],d[1],e[1],f[1]))
        EX = 0.5*(x*dm1).sum()

    if hasattr(self, 'mf'): 
        print('\nmean-field Nucleus-Nucleus   (Ha):%16.6f'%(self.energy_nuc()))
        print('mean-field Core energy       (Ha):%16.6f'%(ecore))
        print('mean-field Exchange energy   (Ha):%16.6f'%(EX))
        print('mean-field Hartree energy    (Ha):%16.6f'%(Vha))
        print('mean-field Total energy      (Ha):%16.6f'%(self.mf.e_tot))
        if (self.nspin==2):
            sp = self.spin/2
            s_ref = sp*(sp+1)
            ss = self.mf.spin_square()
            print('<S^2> and  2S+1                  :%16.7f %16.7f'%(ss[0],ss[1]))
            print('Instead of                       :%16.7f %16.7f'%(s_ref, 2*sp+1))
    #sys.stdout.close()




def sigma_xc(self):
    """
    This calculates the Exchange expectation value and correlation part of self energy, when:
    self.get_k() = Exchange operator/energy
    mat1 is product of this operator and molecular coefficients and it will be diagonalized in expval by einsum
    Sigma_c = E_GW - E_HF
    """
    if self.nspin==1:
      mat = -0.5*self.get_k()
      mat1 = np.dot(self.mo_coeff[0,0,:,:,0], mat)
      expval = np.einsum('nb,nb->n', mat1, self.mo_coeff[0,0,:,:,0]).reshape((1,self.norbs))
      print('---| Expectationvalues of Exchange energy(eV) |---\n %3s  %16s  %3s'%('no.','<Sigma_x> ','occ'))
      for i, (a,b) in enumerate(zip(expval.T*HARTREE2EV,self.mo_occ[0].T)):   #self.h0_vh_x_expval[0,:self.nfermi[0]+5] to limit the virual states
        if (i==self.nfermi[0]): print('-'*50)
        print (' %3d  %16.6f  %3d'%(i,a[0], b[0]))
    elif self.nspin==2:
      mat = -self.get_k()
      expval = np.zeros((self.nspin, self.norbs))
      for s in range(self.nspin):
        mat1 = np.dot(self.mo_coeff[0,s,:,:,0], mat[s])
        expval[s] = np.einsum('nb,nb->n', mat1, self.mo_coeff[0,s,:,:,0])
      print('--------| Expectationvalues of Exchange energy(eV) |--------\n %3s  %16s  %3s  | %13s  %4s'%('no.','<Sigma_x>','occ','<Sigma_x>','occ'))
      for i , (a,b) in enumerate(zip(expval.T* HARTREE2EV,self.mo_occ[0].T)):
        if (i==self.nfermi[0] or i==self.nfermi[1]): print('-'*60)
        print(' %3d  %16.6f  %3d  | %13.6f  %3d'%(i, a[0],b[0],a[1], b[1]))
    if hasattr(self,'mo_energy_gw'):
        sigma_gw_c = self.mo_energy_gw - self.mo_energy
        #sigma_gw_c= np.asanyarray([gw.mo_energy_gw[0,s,nn] -  gw.mo_energy[0,s,nn] for s,nn in enumerate(gw.nn) ]) #Only corrected by GW not scisorres
        if self.nspin==1:
            print('\n---| Correlation contribution of GW@HF (eV) |---\n %3s  %16s  %3s'%('no.','<Sigma_c> ','occ'))
            for i, (a,b) in enumerate(zip(sigma_gw_c.T*HARTREE2EV,self.mo_occ[0].T)):   #self.h0_vh_x_expval[0,:self.nfermi[0]+5] to limit the virual states
                if (i==self.nfermi[0]): print('-'*48)
                print (' %3d  %16.6f  %3d'%(i,a[0], b[0]))
        elif self.nspin==2:
            print('\n--------| Correlation contribution of GW@HF (eV) |---------\n %3s  %16s  %3s  | %13s  %4s'%('no.','<Sigma_c>','occ','<Sigma_c>','occ'))
            for i , (a,b) in enumerate(zip(sigma_gw_c.T* HARTREE2EV,self.mo_occ[0].T)):
                if (i==self.nfermi[0] or i==self.nfermi[1]): print('-'*60)
                print(' %3d  %16.6f  %3d  | %13.6f  %3d'%(i, a[0],b[0],a[1], b[1]))
            
        

#
# Example of reporting expectation values of mean-field calculations.
#
if __name__=='__main__':
    import numpy as np 
    from pyscf import gto, scf
    from pyscf.nao import gw as gw_c
    HARTREE2EV=27.2114
    mol = gto.M( verbose = 0, atom = 'O 0.0, 0.0, 0.622978 ; O 0.0, 0.0, -0.622978',basis = 'cc-pvdz', spin=2, charge=0)
    gto_mf = scf.UHF(mol)
    gto_mf.kernel()
    gw = gw_c(mf=gto_mf, gto=mol, verbosity=3, niter_max_ev=1, kmat_algo='sm0_sum')
    gw.report_mf()  #prints the energy levels of mean-field components
    #gw.kernel_gw()
    #gw.report()     #gives G0W0 spectra
