# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 3. Use proper functions provided by PySCF
#   * Switch between df.incore and df.outcore according to system memory
#   *   (Koh: Is there an identical function in outcore? which one, incore or outcore, is used when need of more memory?)
#   * Use get_veff of scf object instead of get_vxc
#   *   (Koh: get_vxc cannot generate correct J,K matrix from complex density matrix)
#

import numpy as np
import scipy, time
import scipy.linalg
from pyscf import gto, dft, df
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import diis
import sys

FSPERAU = 0.0241888

def transmat(M,U,inv = 1):
    if inv == 1:
        # U.t() * M * U
        Mtilde = np.dot(np.dot(U.T.conj(),M),U)
    elif inv == -1:
        # U * M * U.t()
        Mtilde = np.dot(np.dot(U,M),U.T.conj())
    return Mtilde

def trdot(A,B):
    C = np.trace(np.dot(A,B))
    return C

def matrixpower(A,p,PrintCondition=False):
    """ Raise a Hermitian Matrix to a possibly fractional power. """
    u,s,v = np.linalg.svd(A)
    if (PrintCondition):
        print("matrixpower: Minimal Eigenvalue =", np.min(s))
    for i in range(len(s)):
        if (abs(s[i]) < np.power(10.0,-14.0)):
            s[i] = np.power(10.0,-14.0)
    return np.dot(u,np.dot(np.diag(np.power(s,p)),v))



class RTTDSCF(lib.StreamObject):
    """
    RT-TDSCF base object.
    Other types of propagations may inherit from this.
    Calling this class starts the propagation
    Attributes:
        verbose: int
            Print level.  Default value equals to :class:`ks.verbose`
        conv_tol: float
            converge threshold.  Default value equals to :class:`ks.conv_tol`
        auxbas: str
            auxilliary basis for 2c/3c eri. Default is weigend
        prm: str
            string object with |variable    value| on each line
    Saved results

        output: str
            name of the file with result of propagation


    """
    def __init__(self,ks,prm=None,output = "log.dat", auxbas = "weigend"):
        self.stdout = sys.stdout
        self.verbose = ks.verbose
        self.enuc = ks.energy_nuc()
        self.conv_tol = ks.conv_tol
        self.auxbas = auxbas
        self.hyb = ks._numint.hybrid_coeff(ks.xc, spin=(ks.mol.spin>0)+1)
        self.adiis = None
        self.ks  = ks
        self.eri3c = None
        self.eri2c = None
        self.s = ks.mol.intor_symmetric('int1e_ovlp')
        self.x = matrixpower(self.s,-1./2.)
        self._keys = set(self.__dict__.keys())

        fmat, c_am, v_lm, rho = self.initialcondition(prm)
        start = time.time()
        self.prop(fmat, c_am, v_lm, rho, output)
        end = time.time()
        logger.info(self,"Propagation time: %f", end-start)

        logger.warn(self, 'RT-TDSCF is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')


    def auxmol_set(self, mol, auxbas = "weigend"):
        """
        Generate 2c/3c electron integral (eri2c,eri3c)
        Generate ovlp matrix (S), and AO to Lowdin AO matrix transformation matrix (X)

        Args:
            mol: Mole class
                Default is ks.mol

        Kwargs:
            auxbas: str
                auxilliary basis for 2c/3c eri. Default is weigend

        Returns:
            eri3c: float
                3 center eri. shape: (AO,AO,AUX)
            eri2c: float
                2 center eri. shape: (AUX,AUX)

        """
        auxmol = gto.Mole()
        auxmol.atom = mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        self.auxmol = auxmol
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm,\
        auxmol._bas, auxmol._env)
        eri3c = df.incore.aux_e2(mol, auxmol, intor="cint3c2e_sph", aosym="s1",\
        comp=1 )
        eri2c = df.incore.fill_2c2e(mol,auxmol)
        self.eri3c = eri3c.copy()
        self.eri2c = eri2c.copy()
        return eri3c, eri2c


    def fockbuild(self,dm_lao,it = -1):
        """
        Updates Fock matrix

        Args:
            dm_lao: float or complex
                Lowdin AO density matrix.
            it: int
                iterator for SCF DIIS

        Returns:
            fmat: float or complex
                Fock matrix in Lowdin AO basis
            jmat: float or complex
                Coulomb matrix in AO basis
            kmat: float or complex
                Exact Exchange in AO basis
        """
        if self.params["Model"] == "TDHF":
            Pt = 2.0*transmat(dm_lao,self.x,-1)
            jmat,kmat = self.get_jk(Pt)
            veff = 0.5*(jmat+jmat.T.conj()) - 0.5*(0.5*(kmat + kmat.T.conj()))
            if self.adiis and it > 0:
                return transmat(self.adiis.update(self.s,Pt,self.h + veff),\
                self.x), jmat, kmat
            else:
                return  transmat(self.h + veff,self.x), jmat, kmat
        elif self.params["Model"] == "TDDFT":
            Pt = 2 * transmat(dm_lao,self.x,-1)
            jmat = self.J = self.get_j(Pt)
            Veff = self.J.astype(complex)
            Vxc, excsum, kmat = self.get_vxc(Pt)
            Veff += Vxc
            if self.adiis and it > 0:
                return transmat(self.adiis.update(self.s,Pt,self.h + Veff),\
                self.x), jmat, kmat
            else:
                return transmat(self.h + Veff,self.x), jmat, kmat


    def get_vxc(self,dm):
        """
        Update exchange matrices and energy
        Args:
            dm: float or complex
                AO density matrix.

        Returns:
            vxc: float or complex
                exchange-correlation matrix in AO basis
            excsum: float
                exchange-correlation energy
            kmat: float or complex
                Exact Exchange in AO basis
        """

        nelec, excsum, vxc = self.ks._numint.nr_vxc(self.ks.mol, \
        self.ks.grids, self.ks.xc, dm)
        self.exc = excsum
        vxc  = vxc.astype(complex)
        if(self.hyb > 0.01):
            kmat = self.get_k(dm)
            vxc += -0.5 * self.hyb * kmat
        else:
            kmat = None
        return vxc, excsum, kmat

    def get_jk(self, dm):
        """
        Update Coulomb and Exact Exchange Matrix

        Args:
            dm: float or complex
                AO density matrix.
        Returns:
            jmat: float or complex
                Coulomb matrix in AO basis
            kmat: float or complex
                Exact Exchange in AO basis
        """
        jmat = self.get_j(dm)
        kmat = self.get_k(dm)
        return jmat, kmat

    def get_j(self,dm):
        """
        Update Coulomb Matrix

        Args:
            dm: float or complex
                AO density matrix.
        Returns:
            jmat: float or complex
                Coulomb matrix in AO basis
        """
        rho = np.einsum("ijp,ij->p", self.eri3c, dm)
        rho = np.linalg.solve(self.eri2c, rho)
        jmat = np.einsum("p,ijp->ij", rho, self.eri3c)
        return jmat

    def get_k(self,dm):
        """
        Update Exact Exchange Matrix

        Args:
            dm: float or complex
                AO density matrix.
        Returns:
            kmat: float or complex
                Exact Exchange in AO basis
        """
        naux = self.auxmol.nao_nr()
        nao = self.ks.mol.nao_nr()
        kpj = np.einsum("ijp,jk->ikp", self.eri3c, dm)
        pik = np.linalg.solve(self.eri2c, kpj.reshape(-1,naux).T.conj())
        kmat = np.einsum("pik,kjp->ij", pik.reshape(naux,nao,nao), self.eri3c)
        return kmat

    def initialcondition(self,prm):
        """
        Prepare the variables/Matrices needed for propagation
        The SCF is done here to make matrices that are not accessable from pyscf.scf
        Args:
            prm: str
                string object with |variable    value| on each line
        Returns:
            fmat: float or complex
                Fock matrix in Lowdin AO basis
            c_am: float
                Transformation Matrix |AO><MO|
            v_lm: float
                Transformation Matrix |LAO><MO|
            rho: float or complex
                Initial MO density matrix.

        """
        from pyscf.rt import tdfields
        self.auxmol_set(self.ks.mol, auxbas = self.auxbas)
        self.params = dict()

        logger.log(self,"""
            ===================================
            |  Realtime TDSCF module          |
            ===================================
            | J. Parkhill, T. Nguyen          |
            | J. Koh, J. Herr,  K. Yao        |
            ===================================
            | Refs: 10.1021/acs.jctc.5b00262  |
            |       10.1063/1.4916822         |
            ===================================
            """)
        n_ao = self.ks.mol.nao_nr()
        n_occ = int(sum(self.ks.mo_occ)/2)
        logger.log(self,"n_ao: %d        n_occ: %d", n_ao,\
        n_occ)
        self.readparams(prm)
        fmat, c_am, v_lm = self.initfockbuild() # updates self.C
        rho = 0.5*np.diag(self.ks.mo_occ).astype(complex)
        self.field = tdfields.FIELDS(self, self.params)
        self.field.initializeexpectation(rho, c_am)
        return fmat, c_am, v_lm, rho

    def readparams(self,prm):
        """
        Set Defaults, Read the file and fill the params dictionary

        Args:
            prm: str
                string object with |variable    value| on each line
        """
        self.params["Model"] = "TDDFT"
        self.params["Method"] = "MMUT"
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
                    if s[0] == "MaxIter" or s[0] == str("ApplyImpulse") or \
                    s[0] == str("ApplyCw") or s[0] == str("StatusEvery"):
                        self.params[s[0]] = int(s[1])
                    elif s[0] == "Model" or s[0] == "Method":
                        self.params[s[0]] = s[1].upper()
                    else:
                        self.params[s[0]] = float(s[1])

        logger.log(self,"=============================")
        logger.log(self,"         Parameters")
        logger.log(self,"=============================")
        logger.log(self,"Model: " + self.params["Model"].upper())
        logger.log(self,"Method: "+ self.params["Method"].upper())
        logger.log(self,"dt: %.2f", self.params["dt"])
        logger.log(self,"MaxIter: %d", self.params["MaxIter"])
        logger.log(self,"ExDir: %.2f", self.params["ExDir"])
        logger.log(self,"EyDir: %.2f", self.params["EyDir"])
        logger.log(self,"EzDir: %.2f", self.params["EzDir"])
        logger.log(self,"FieldAmplitude: %.4f", self.params["FieldAmplitude"])
        logger.log(self,"FieldFreq: %.4f", self.params["FieldFreq"])
        logger.log(self,"Tau: %.2f", self.params["Tau"])
        logger.log(self,"tOn: %.2f", self.params["tOn"])
        logger.log(self,"ApplyImpulse: %d", self.params["ApplyImpulse"])
        logger.log(self,"ApplyCw: %d", self.params["ApplyCw"])
        logger.log(self,"StatusEvery: %d", self.params["StatusEvery"])
        logger.log(self,"=============================\n\n")

        return

    def initfockbuild(self):
        """
        Using Roothan's equation to build a Initial Fock matrix and
        Transformation Matrices

        Returns:
            fmat: float or complex
                Fock matrix in Lowdin AO basis
            c_am: float
                Transformation Matrix |AO><MO|
            v_lm: float
                Transformation Matrix |LAO><MO|
        """
        start = time.time()
        n_occ = int(sum(self.ks.mo_occ)/2)
        err = 100
        it = 0
        self.h = self.ks.get_hcore()
        s = self.s.copy()
        x = self.x.copy()
        sx = np.dot(s,x)
        dm_lao = 0.5*transmat(self.ks.get_init_guess(self.ks.mol, \
        self.ks.init_guess), sx).astype(complex)

        if isinstance(self.ks.diis, lib.diis.DIIS):
            self.adiis = self.ks.diis
        elif self.ks.diis:
            self.adiis = diis.SCF_DIIS(self.ks, self.ks.diis_file)
            self.adiis.space = self.ks.diis_space
            self.adiis.rollback = self.ks.diis_space_rollback
        else:
            self.adiis = None

        fmat, jmat, kmat = self.fockbuild(dm_lao)
        dm_lao_old = dm_lao
        etot = self.energy(dm_lao,fmat, jmat, kmat)+ self.enuc

        while (err > self.conv_tol):
            # Diagonalize F in the lowdin basis
            eigs, v_lm = np.linalg.eig(fmat)
            idx = eigs.argsort()
            eigs.sort()
            v_lm = v_lm[:,idx].copy()
            # Fill up the density in the MO basis and then Transform back
            rho = 0.5*np.diag(self.ks.mo_occ).astype(complex)
            dm_lao = transmat(rho,v_lm,-1)
            etot_old = etot
            etot = self.energy(dm_lao,fmat, jmat, kmat)
            fmat, jmat, kmat = self.fockbuild(dm_lao,it)
            err = abs(etot-etot_old)
            logger.debug(self, "Ne: %f", np.trace(rho))
            logger.debug(self, "Iteration: %d         Energy: %.11f      \
            Error = %.11f", it, etot, err)
            it += 1
            if it > self.ks.max_cycle:
                logger.log(self, "Max cycle of SCF reached: %d\n Exiting TDSCF. Please raise ks.max_cycle", it)
                quit()
        rho = 0.5*np.diag(self.ks.mo_occ).astype(complex)
        dm_lao = transmat(rho,v_lm,-1)
        c_am = np.dot(self.x,v_lm)
        logger.log(self, "Ne: %f", np.trace(rho))
        logger.log(self, "Converged Energy: %f", etot)
        # logger.log(self, "Eigenvalues: %f", eigs.real)
        # print "Eigenvalues: ", eigs.real
        end = time.time()
        logger.info(self, "Initial Fock Built time: %f", end-start)
        return fmat, c_am, v_lm

    def split_rk4_step_mmut(self, w, v , oldrho , tnow, dt ,IsOn):
        Ud = np.exp(w*(-0.5j)*dt);
        U = transmat(np.diag(Ud),v,-1)
        RhoHalfStepped = transmat(oldrho,U,-1)
        # If any TCL propagation occurs...
        # DontDo=
        # SplitLiouvillian( RhoHalfStepped, k1,tnow,IsOn);
        # v2 = (dt/2.0) * k1;
        # v2 += RhoHalfStepped;
        # SplitLiouvillian(  v2, k2,tnow+(dt/2.0),IsOn);
        # v3 = (dt/2.0) * k2;
        # v3 += RhoHalfStepped;
        # SplitLiouvillian(  v3, k3,tnow+(dt/2.0),IsOn);
        # v4 = (dt) * k3;
        # v4 += RhoHalfStepped;
        # SplitLiouvillian(  v4, k4,tnow+dt,IsOn);
        # newrho = RhoHalfStepped;
        # newrho += dt*(1.0/6.0)*k1;
        # newrho += dt*(2.0/6.0)*k2;
        # newrho += dt*(2.0/6.0)*k3;
        # newrho += dt*(1.0/6.0)*k4;
        # newrho = U*newrho*U.t();
        #
        newrho = transmat(RhoHalfStepped,U,-1)

        return newrho

    def tddftstep(self,fmat, c_am, v_lm, rho, rhom12, tnow):
        """
        Take dt step in propagation
        updates matrices and rho to next timestep
        Args:
            fmat: float or complex
                Fock matrix in Lowdin AO basis
            c_am: float or complex
                Transformation Matrix |AO><MO|
            v_lm: float or complex
                Transformation Matrix |LAO><MO|
            rho: complex
                MO density matrix.
            rhom12: complex
            tnow: float
                current time in A.U.
        Returns:
            n_rho: complex
                MO density matrix.
            n_rhom12: complex
            n_c_am: complex
                Transformation Matrix |AO><MO|
            n_v_lm: complex
                Transformation Matrix |LAO><MO|
            n_fmat: complex
                Fock matrix in Lowdin AO basis
            n_jmat: complex
                Coulomb matrix in AO basis
            n_kmat: complex
                Exact Exchange in AO basis
        """
        if (self.params["Method"] == "MMUT"):
            fmat, n_jmat, n_kmat = self.fockbuild(transmat(rho,v_lm,-1))
            n_fmat = fmat.copy()
            fmat_c = np.conj(fmat)
            fmat_prev = transmat(fmat_c, v_lm)
            eigs, rot = np.linalg.eig(fmat_prev)
            idx = eigs.argsort()
            eigs.sort()
            rot = rot[:,idx].copy()
            rho = transmat(rho, rot)
            rhoM12 = transmat(rhom12, rot)
            v_lm = np.dot(v_lm , rot)
            c_am = np.dot(self.x , v_lm)
            n_v_lm = v_lm.copy()
            n_c_am = c_am.copy()
            fmat_mo = np.diag(eigs).astype(complex)
            fmatfield, IsOn = self.field.applyfield(fmat_mo,c_am,tnow)
            w,v = scipy.linalg.eig(fmatfield)
            NewRhoM12 = self.split_rk4_step_mmut(w, v, rhom12, tnow, \
            self.params["dt"], IsOn)
            NewRho = self.split_rk4_step_mmut(w, v, NewRhoM12, tnow,\
            self.params["dt"]/2.0, IsOn)
            n_rho = 0.5*(NewRho+(NewRho.T.conj()));
            n_rhom12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))
            return n_rho, n_rhom12, n_c_am, n_v_lm, n_fmat, n_jmat, n_kmat
        else:
            raise Exception("Unknown Method...")
        return


    def dipole(self, rho, c_am):
        """
        Args:
            c_am: float or complex
                Transformation Matrix |AO><MO|
            rho: complex
                MO density matrix.
        Returns:
            dipole: float
                xyz component of dipole of a molecule. [x y z]
        """
        return self.field.expectation(rho, c_am)

    def energy(self,dm_lao,fmat,jmat,kmat):
        """
        Args:
            dm_lao: complex
                Density in LAO basis.
            fmat: complex
                Fock matrix in Lowdin AO basis
            jmat: complex
                Coulomb matrix in AO basis
            kmat: complex
                Exact Exchange in AO basis
        Returns:
            e_tot: float
                Total Energy of a system
        """
        if (self.params["Model"] == "TDHF"):
            hlao = transmat(self.h,self.x)
            e_tot = (self.enuc+np.trace(np.dot(dm_lao,hlao+fmat))).real
            return e_tot
        elif self.params["Model"] == "TDDFT":
            dm = transmat(dm_lao,self.x,-1)
            exc = self.exc
            if(self.hyb > 0.01):
                exc -= 0.5 * self.hyb * trdot(dm,kmat)
            # if not using auxmol
            eh = trdot(dm,2*self.h)
            ej = trdot(dm,jmat)
            e_tot = (eh + ej + exc + self.enuc).real
            return e_tot

    def loginstant(self, rho, c_am, v_lm, fmat, jmat, kmat, tnow, it):
        """
        time is logged in atomic units.
        Args:
            rho: complex
                MO density matrix.
            c_am: complex
                Transformation Matrix |AO><MO|
            v_lm: complex
                Transformation Matrix |LAO><MO|
            fmat: complex
                Fock matrix in Lowdin AO basis
            jmat: complex
                Coulomb matrix in AO basis
            kmat: complex
                Exact Exchange in AO basis
            tnow: float
                Current time in propagation in A.U.
            it: int
                Number of iteration of propagation
        Returns:
            tore: str
                |t, dipole(x,y,z), energy|

        """
        np.set_printoptions(precision = 7)
        tore = str(tnow)+" "+str(self.dipole(rho, c_am).real).rstrip("]").lstrip("[")+\
         " " +str(self.energy(transmat(rho,v_lm,-1),fmat, jmat, kmat))

        if it%self.params["StatusEvery"] ==0 or it == self.params["MaxIter"]-1:
            logger.log(self, "t: %f fs    Energy: %f a.u.   Total Density: %f",\
            tnow*FSPERAU,self.energy(transmat(rho,v_lm,-1),fmat, jmat, kmat), \
            2*np.trace(rho))
            logger.log(self, "Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f",\
             self.dipole(rho, c_am).real[0],self.dipole(rho, c_am).real[1],\
             self.dipole(rho, c_am).real[2])
        return tore

    def prop(self, fmat, c_am, v_lm, rho, output):
        """
        The main tdscf propagation loop.
        Args:
            fmat: complex
                Fock matrix in Lowdin AO basis
            c_am: complex
                Transformation Matrix |AO><MO|
            v_lm: complex
                Transformation Matrix |LAO><MO|
            rho: complex
                MO density matrix.
            output: str
                name of the file with result of propagation
        Saved results:
            f: file
                output file with |t, dipole(x,y,z), energy|
        """
        it = 0
        tnow = 0
        rhom12 = rho.copy()
        n_occ = int(sum(self.ks.mo_occ)/2)
        f = open(output,"a")
        logger.log(self,"\n\nPropagation Begins")
        start = time.time()
        while (it<self.params["MaxIter"]):
            rho, rhom12, c_am, v_lm, fmat, jmat, kmat = self.tddftstep(fmat, c_am, v_lm, rho, rhom12, tnow)
            # rho = newrho.copy()
            # rhom12 = newrhom12.copy()
            #self.log.append(self.loginstant(it))
            f.write(self.loginstant(rho, c_am, v_lm, fmat, jmat, kmat, tnow, it)+"\n")
            # Do logging.
            tnow = tnow + self.params["dt"]
            if it%self.params["StatusEvery"] ==0 or \
            it == self.params["MaxIter"]-1:
                end = time.time()
                logger.log(self, "%f hr/ps", \
                (end - start)/(60*60*tnow * FSPERAU * 0.001))
            it = it + 1

        f.close()
