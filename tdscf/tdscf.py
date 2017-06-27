import numpy as np
import scipy, os, time
import scipy.linalg
from tdscf import *
import func
from pyscf import gto, dft, scf, ao2mo, df
from pyscf.lib import logger
from tdfields import *
from cmath import *
import ctypes
import sys

FsPerAu = 0.0241888

class tdscf:
        """
        A general TDSCF object.
        Other types of propagations may inherit from this.
        """
        def __init__(self,the_scf_,prm=None,output = 'log.dat', prop_=True):
                """
                Args:
                        the_scf_: rks object 
                        prm: parameter (look at examples/tdscf how it is written)
                        ouput: str of output filename
                Returns:
                        Nothing.
                """
                self.stdout = sys.stdout
                self.verbose = 1 # for now (to use PYSCF's logger)
                self.Enuc = the_scf_.energy_nuc()#the_scf_.e_tot - dft.rks.energy_elec(the_scf_,the_scf_.make_rdm1())[0]
                self.eri3c = None
                self.eri2c = None
                self.n_aux = None
                self.hyb = the_scf_._numint.hybrid_coeff(the_scf_.xc, spin=(the_scf_.mol.spin>0)+1)
                self.adiis = None
                self.Exc = None
                #Global numbers
                self.t = 0.0
                self.n_ao = None
                self.n_mo = None
                self.n_occ = None
                self.n_virt = None
                self.n_e = None
                #Global Matrices
                self.rho = None # Current MO basis density matrix. (the idempotent (0,1) kind)
                self.rhoM12 = None # For MMUT step
                self.F = None # (LAO x LAO)
                self.K = None
                self.J = None
                self.eigs = None # current fock eigenvalues.
                self.S = None # (ao X ao)
                self.C = None # (ao X mo)
                self.X = None
                self.V = None # LAO, current MO
                self.H = None # (ao X ao)  core hamiltonian.
                self.B = None # for ee2
                self.log = []
                # Objects
                self.the_scf  = the_scf_
                self.mol = the_scf_.mol
                self.MyBO = None
                self.auxmol_set()
                self.params = dict()
                self.initialcondition(prm)
                self.field = fields(the_scf_, self.params)
                self.field.InitializeExpectation(self.rho, self.C)
                start = time.time()
                self.prop(output)
                end = time.time()
                logger.log(self,'Propagation time: %f', end-start)
                return



        def auxmol_set(self,auxbas = "weigend"):
                logger.log(self,'GENERATING INTEGRAL')
                auxmol = gto.Mole()
                auxmol.atom = self.mol.atom
                auxmol.basis = auxbas
                auxmol.build()
                mol = self.mol
                nao = self.n_ao = mol.nao_nr()
                naux = self.n_aux = auxmol.nao_nr()
                atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
                eri3c = df.incore.aux_e1(mol, auxmol, intor='cint3c2e_sph', aosym='s1', comp=1 )
                eri3c = eri3c.reshape(naux,nao,nao).T
                logger.log(self,'ERI3C INTEGRALS GENERATED')
                eri2c = np.empty((naux,naux))
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                        pl = 0
                        for l in range(mol.nbas, mol.nbas+auxmol.nbas):
                                shls = (k, l)
                                buf = gto.getints_by_shell('cint2c2e_sph', shls, atm, bas, env)
                                dk, dl = buf.shape
                                eri2c[pk:pk+dk,pl:pl+dl] = buf
                                pl += dl
                        pk += dk
                logger.log(self,'ERI2C INTEGRALS GENERATED')
                self.eri3c = eri3c
                self.eri2c = eri2c
                RSinv = MatrixPower(eri2c,-0.5)
                self.B = np.einsum('ijp,pq->ijq', self.eri3c, RSinv) # (AO,AO,n_aux)
                # change to (LAO,LAO,n_aux) BpqR

                self.S = self.the_scf.get_ovlp() # (ao X ao)
                self.X = MatrixPower(self.S,-1./2.) # AO, LAO
                logger.log(self,'BpqR GENERATED')

                return

        def FockBuild(self,P,it = -1):
                """
                Updates self.F given current self.rho (both complex.)
                Fock matrix with HF
                Args:
                    P = LAO density matrix.
                Returns:
                    Fock matrix(lao) . Updates self.F
                """
                if self.params["Model"] == "TDHF":
                        Pt = 2.0*TransMat(P,self.X,-1)
                        J,K = self.get_jk(Pt)
                        Veff = 0.5*(J+J.T.conj()) - 0.5*(0.5*(K + K.T.conj()))
                        if self.adiis and it > 0:
                                return TransMat(self.adiis.update(self.S,Pt,self.H + Veff),self.X)
                        else:
                                return  TransMat(self.H + 0.5*(J+J.T.conj()) - 0.5*(0.5*(K + K.T.conj())),self.X)
                elif self.params["Model"] == "TDDFT":
                        Pt = 2 * TransMat(P,self.X,-1)
                        self.J = self.get_j(Pt)
                        Veff = self.J.astype(complex)
                        Veff += self.get_vxc(Pt)
                        if self.adiis and it > 0:
                                return TransMat(self.adiis.update(self.S,Pt,self.H + Veff),self.X)
                        else:
                                return TransMat(self.H + Veff,self.X)


        def get_vxc(self,P):
                '''
                Args:
                        P: AO density matrix

                Returns:
                        Vxc: Exchange and Correlation matrix (AO)
                '''
                
                nelec, excsum, Vxc = self.the_scf._numint.nr_vxc(self.mol, self.the_scf.grids, self.the_scf.xc, P)
                self.Exc = excsum
                Vxc  = Vxc.astype(complex)
                if(self.hyb > 0.01):
                        K = self.get_k(P)
                        self.K = K
                        Vxc += -0.5 * self.hyb * K
                return Vxc

        def get_jk(self, P):
                '''
                Args:
                        P: AO density matrix
                Returns:
                        J: Coulomb matrix
                        K: Exchange matrix
                '''
                return self.get_j(P), self.get_k(P)

        def get_j(self,P):
                '''
                J is made from real part of rho (I think it's correct)
                Args:
                        P: AO density matrix

                Returns:
                        J: Coulomb matrix (AO)
                '''
                naux = self.n_aux
                nao = self.n_ao
                rho = np.einsum('ijp,ij->p', self.eri3c, P)
                rho = np.linalg.solve(self.eri2c, rho)
                jmat = np.einsum('p,ijp->ij', rho, self.eri3c)
                return jmat

        def get_k(self,P):
                '''
                Args:
                        P: AO density matrix
                Returns:
                        K: Exchange matrix (AO)
                '''
                naux = self.n_aux
                nao = self.n_ao
                kpj = np.einsum('ijp,jk->ikp', self.eri3c, P)
                pik = np.linalg.solve(self.eri2c, kpj.reshape(-1,naux).T.conj())
                rkmat = np.einsum('pik,kjp->ij', pik.reshape(naux,nao,nao), self.eri3c)
                return rkmat

        def initialcondition(self,prm):
                print '''
                ===================================
                |  Realtime TDSCF module          |
                ===================================
                | J. Parkhill, T. Nguyen          |
                | J. Koh, J. Herr,  K. Yao        |
                ===================================
                | Refs: 10.1021/acs.jctc.5b00262  |
                |       10.1063/1.4916822         |
                ===================================
                '''
                n_ao = self.n_ao = self.the_scf.make_rdm1().shape[0]
                n_mo = self.n_mo = n_ao # should be fixed.
                n_occ = self.n_occ = int(sum(self.the_scf.mo_occ)/2)
                self.n_virt = self.n_mo - self.n_occ
                logger.log(self,'n_ao: %d       n_mo: %d        n_occ: %d', n_ao,n_mo,n_occ)
                self.ReadParams(prm)
                self.InitializeLiouvillian()
                return

        def ReadParams(self,prm):
                '''
                Set Defaults, Read the file and fill the params dictionary
                '''
                self.params["Model"] = "TDDFT" #"TDHF"; the difference of Fock matrix and energy
                self.params["Method"] = "MMUT"#"MMUT"
                self.params["BBGKY"]=0
                self.params["TDCIS"]=1

                self.params["dt"] =  0.02
                self.params["MaxIter"] = 15000

                self.params["ExDir"] = 1.0
                self.params["EyDir"] = 1.0
                self.params["EzDir"] = 1.0
                self.params["FieldAmplitude"] = 0.01
                self.params["FieldFreq"] = 0.9202
                self.params["Tau"] = 0.07
                self.params["tOn"] = 7.0*self.params["Tau"]
                self.params["ApplyImpulse"] = 1
                self.params["ApplyCw"] = 0

                self.params["StatusEvery"] = 5000
                self.params["Print"]=0
                # Here they should be read from disk.
                if(prm != None):
                        for line in prm.splitlines():
                                s = line.split()
                                if len(s) > 1:
                                        if s[0] == "MaxIter" or s[0] == str("ApplyImpulse") or s[0] == str("ApplyCw") or s[0] == str("StatusEvery"):
                                                self.params[s[0]] = int(s[1])
                                        elif s[0] == "Model" or s[0] == "Method":
                                                self.params[s[0]] = s[1]
                                        else:
                                                self.params[s[0]] = float(s[1])

                print "============================="
                print "         Parameters"
                print "============================="
                print "Model:", self.params["Model"]
                print "Method:", self.params["Method"]
                print "dt:", self.params["dt"]
                print "MaxIter:", self.params["MaxIter"]
                print "ExDir:", self.params["ExDir"]
                print "EyDir:", self.params["EyDir"]
                print "EzDir:", self.params["EzDir"]
                print "FieldAmplitude:", self.params["FieldAmplitude"]
                print "FieldFreq:", self.params["FieldFreq"]
                print "Tau:", self.params["Tau"]
                print "tOn:", self.params["tOn"]
                print "ApplyImpulse:", self.params["ApplyImpulse"]
                print "ApplyCw:", self.params["ApplyCw"]
                print "StatusEvery:", self.params["StatusEvery"]
                print "=============================\n\n"

                return

        def InitializeLiouvillian(self):
                '''
                Get an initial Fock matrix.
                '''
                self.V = np.eye(self.n_ao)
                self.C = self.X.copy() # Initial set of orthogonal coordinates.
                self.InitFockBuild() # updates self.C
                self.rho = 0.5*np.diag(self.the_scf.mo_occ).astype(complex)
                self.rhoM12 = self.rho.copy()
                return

        def InitFockBuild(self):
                '''
                Using Roothan's equation to build a Fock matrix and initial density matrix
                Returns:
                        self consistent density in Lowdin basis.
                '''
                start = time.time()
                n_occ = self.n_occ
                Ne = self.n_e = 2.0 * n_occ
                err = 100
                it = 0
                self.H = self.the_scf.get_hcore()
                S = self.S.copy()
                X = MatrixPower(S,-1./2.)
                SX = np.dot(S,X)
                Plao = 0.5*TransMat(self.the_scf.get_init_guess(self.mol, self.the_scf.init_guess), SX).astype(complex)
                self.adiis = self.the_scf.diis

                self.F = self.FockBuild(Plao)
                Plao_old = Plao
                E = self.energy(Plao)+ self.Enuc

                while (err > 10**-10):
                        # Diagonalize F in the lowdin basis
                        self.eigs, self.V = np.linalg.eig(self.F)
                        idx = self.eigs.argsort()
                        self.eigs.sort()
                        self.V = self.V[:,idx].copy()
                        # Fill up the density in the MO basis and then Transform back
                        Pmo = 0.5*np.diag(self.the_scf.mo_occ).astype(complex)
                        Plao = TransMat(Pmo,self.V,-1)
                        Eold = E
                        E = self.energy(Plao)
                        self.F = self.FockBuild(Plao,it)
                        err = abs(E-Eold)
                        if (self.verbose > 1):
                                logger.log(self, 'Ne: %f', np.trace(Pmo))
                                logger.log(self, 'Iteration: %d         Energy: %.11f      Error = %.11f', it, E, err)
                        it += 1
                Pmo = 0.5*np.diag(self.the_scf.mo_occ).astype(complex)
                Plao = TransMat(Pmo,self.V,-1)
                self.C = np.dot(self.X,self.V)
                self.rho = TransMat(Plao,self.V)
                self.rhoM12 = TransMat(Plao,self.V)
                logger.log(self, 'Ne: %f', np.trace(Pmo))
                logger.log(self, 'Converged Energy: %f', E)
                #logger.log(self, 'Eigenvalues: %f',  self.eigs.real)
                print "Eigenvalues: ", self.eigs.real
                end = time.time()
                logger.log(self, 'Initial Fock Built time: %f', end-start)
                return Plao

        def Split_RK4_Step_MMUT(self, w, v , oldrho , time, dt ,IsOn):
                Ud = np.exp(w*(-0.5j)*dt);
                U = TransMat(np.diag(Ud),v,-1)
                RhoHalfStepped = TransMat(oldrho,U,-1)
                # If any TCL propagation occurs...
                DontDo="""
                SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
                v2 = (dt/2.0) * k1;
                v2 += RhoHalfStepped;
                SplitLiouvillian(  v2, k2,tnow+(dt/2.0),IsOn);
                v3 = (dt/2.0) * k2;
                v3 += RhoHalfStepped;
                SplitLiouvillian(  v3, k3,tnow+(dt/2.0),IsOn);
                v4 = (dt) * k3;
                v4 += RhoHalfStepped;
                SplitLiouvillian(  v4, k4,tnow+dt,IsOn);
                newrho = RhoHalfStepped;
                newrho += dt*(1.0/6.0)*k1;
                newrho += dt*(2.0/6.0)*k2;
                newrho += dt*(2.0/6.0)*k3;
                newrho += dt*(1.0/6.0)*k4;
                newrho = U*newrho*U.t();
                """
                newrho = TransMat(RhoHalfStepped,U,-1)

                return newrho

        def TDDFTstep(self,time):
                if (self.params["Method"] == "MMUT"):
                        self.F = self.FockBuild(TransMat(self.rho,self.V,-1)) # is LAO basis
                        self.F = np.conj(self.F)
                        Fmo_prev = TransMat(self.F, self.V)
                        self.eigs, rot = np.linalg.eig(Fmo_prev)
                        # print Fmo_prev, rot
                        # Rotate all the densities into the current fock eigenbasis.
                        self.rho = TransMat(self.rho, rot)
                        self.rhoM12 = TransMat(self.rhoM12, rot)
                        self.V = np.dot(self.V , rot)
                        self.C = np.dot(self.X , self.V)
                        # propagation is done in the current eigenbasis.
                        Fmo = np.diag(self.eigs).astype(complex)
                        # Check that the energy is still correct.
                        Hmo = TransMat(self.H,self.C)
                        FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.C,time)
                        w,v = scipy.linalg.eig(FmoPlusField)
                        NewRhoM12 = self.Split_RK4_Step_MMUT(w, v, self.rhoM12, time, self.params["dt"], IsOn)
                        NewRho = self.Split_RK4_Step_MMUT(w, v, NewRhoM12, time,self.params["dt"]/2.0, IsOn)
                        self.rho = 0.5*(NewRho+(NewRho.T.conj()));
                        self.rhoM12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))
                else:
                        raise Exception("Unknown Method...")
                return

        def step(self,time):
                """
                Performs a step
                Updates t, rho, and possibly other things.
                """
                if (self.params["Model"] == "TDDFT" or self.params["Model"] == "TDHF"):
                        return self.TDDFTstep(time)
                elif(self.params["Model"] == "EE2"):
                        return self.EE2step(time)
                return

        def dipole(self):
                return self.field.Expectation(self.rho, self.C)

        def energy(self,Plao,IfPrint=False):
                """
                P: Density in LAO basis.
                """
                if (self.params["Model"] == "TDHF" or self.params["Model"] == "BBGKY"):
                        Hlao = TransMat(self.H,self.X)
                        return (self.Enuc+np.trace(np.dot(Plao,Hlao+self.F))).real
                elif self.params["Model"] == "TDDFT":
                        Hlao = TransMat(self.H,self.X)
                        P = TransMat(Plao,self.X,-1)
                        J = self.J.copy()
                        Exc = self.Exc
                        if(self.hyb > 0.01):
                                Exc -= 0.5 * self.hyb * TrDot(P,self.K)
                        # if not using auxmol
                        EH = TrDot(Plao,2*Hlao)
                        EJ = TrDot(P,J)
                        E = EH + EJ + Exc + self.Enuc

                        return E.real

        def loginstant(self,iter):
                """
                time is logged in atomic units.
                """
                np.set_printoptions(precision = 7)
                tore = str(self.t)+" "+str(self.dipole().real).rstrip(']').lstrip('[')+ " " +str(self.energy(TransMat(self.rho,self.V,-1),False))+" "+str(np.trace(self.rho))

                if iter%self.params["StatusEvery"] ==0 or iter == self.params["MaxIter"]-1:
                        logger.log(self, 't: %f fs    Energy: %f a.u.   Total Density: %f', self.t*FsPerAu,self.energy(TransMat(self.rho,self.V,-1)), 2*np.trace(self.rho))
                        logger.log(self, 'Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f', self.dipole().real[0],self.dipole().real[1],self.dipole().real[2] )
                return tore

        def prop(self,output):
                """
                The main tdscf propagation loop.
                """
                iter = 0
                self.t = 0
                f = open(output,'a')
                logger.log(self,'Energy Gap (eV): %f',abs(self.eigs[self.n_occ]-self.eigs[self.n_occ-1])*27.2114)
                logger.log(self,'\n\nPropagation Begins')
                start = time.time()
                while (iter<self.params["MaxIter"]):
                        self.step(self.t)
                        #print self.t
                        #self.log.append(self.loginstant(iter))
                        f.write(self.loginstant(iter)+"\n")
                        # Do logging.
                        self.t = self.t + self.params["dt"]
                        if iter%self.params["StatusEvery"] ==0 or iter == self.params["MaxIter"]-1:
                                end = time.time()
                                logger.log(self, '%f hr/ps', (end - start)/(60*60*self.t * FsPerAu * 0.001))
                        iter = iter + 1

                f.close()
