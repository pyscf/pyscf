# TODO: refactor the code before adding to FEATURES list by PySCF-1.5 release
# 1. code style
#   * Indent: 8 -> 4
#   * The statement "from import *"
#   * Class name should be capitalized
#   * Function/method should be all lowercase
#   * Inherite class from object or lib.StreamObject
#   * Remove the unused modules: func, os, ...
#   * Line wrap around 80 columns
#   * Use either double quote or single quote, not mix
#   * Avoid python builtin keyword: time, ...
# 
# 2. Conventions required by PySCF
#   * Class attributes should be immutable.
#   * Class should not hold intermediate states.
#   * Suffix _ only be used for functions with side effect
#   * Add attribute ._keys for sanity check
#   * Meaningful return value for methods
#   * Class attributes should be all lowercase
#   * Remove unused arguments in function definition
#

import numpy as np
from cmath import *
from func import *
import scipy
import scipy.linalg

class fields:
        """
        A class which manages field perturbations. Mirrors TCL_FieldMatrices.h
        """
        def __init__(self,the_scf_, params_):
                self.dip_ints = None # AO dipole integrals.
                self.dip_ints_bo = None
                self.nuc_dip = None
                self.dip_mo = None # Nuclear dipole (AO)
                self.Generate(the_scf_)
                self.fieldAmplitude = params_["FieldAmplitude"]
                self.tOn = params_["tOn"]
                self.Tau = params_["Tau"]
                self.FieldFreq = params_["FieldFreq"]
                self.pol = np.array([params_["ExDir"],params_["EyDir"],params_["EzDir"]])
                self.pol0 = None
                self.pol0AA = None
                return

        def Generate(self,the_scf):
                """
                Performs the required PYSCF calls to generate the AO basis dipole matrices.
                """
                self.dip_ints = the_scf.mol.intor('cint1e_r_sph', comp=3) # component,ao,ao.
                charges = the_scf.mol.atom_charges()
                coords  = the_scf.mol.atom_coords()
                self.nuc_dip = np.einsum('i,ix->x', charges, coords)
                return

        def ImpulseAmp(self,time):
                amp = self.fieldAmplitude*np.sin(self.FieldFreq*time)*(1.0/sqrt(2.0*3.1415*self.Tau*self.Tau))*np.exp(-1.0*np.power(time-self.tOn,2.0)/(2.0*self.Tau*self.Tau));
                IsOn = False
                if (np.abs(amp)>pow(10.0,-9.0)):
                        IsOn = True
                return amp,IsOn

        def InitializeExpectation(self,rho0_, C_,nA = None):
                self.pol0 = self.Expectation(rho0_,C_)
                if nA != None:
                        self.dip_ints_bo = self.dip_ints.copy()
                        for i in range(3):
                                self.dip_ints_bo[i] = TransMat(self.dip_ints[i],C_)
                        self.pol0AA = self.Expectation(rho0_,C_,True,nA)


        def ApplyField(self, a_mat, time):
                """
                Args:
                        a_mat: an AO matrix to which the field is added.
                        time: current time.
                Returns:
                        a_mat + dipole field at this time.
                        IsOn
                """
                amp, IsOn = self.ImpulseAmp(time)
                mpol = self.pol * amp
                if (IsOn):
                        #print "Field on"
                        return a_mat + 2.0*np.einsum("kij,k->ij",self.dip_ints,mpol), True
                else:
                        return a_mat, False

        def ApplyField(self, a_mat, c_mat, time):
                """
                Args:
                        a_mat: an MO matrix to which the field is added.
                        c_mat: a AO=>MO coefficient matrix.
                        time: current time.
                Returns:
                        a_mat + dipole field at this time.
                        IsOn
                """
                amp, IsOn = self.ImpulseAmp(time)
                mpol = self.pol * amp
                if (IsOn):
                        return a_mat + 2.0*TransMat(np.einsum("kij,k->ij",self.dip_ints,mpol),c_mat), True
                else :
                        return a_mat, False

        def Expectation(self, rho_, C_, AA = False, nA = None,U = None):
                """
                Args:
                        rho_: current MO density.
                        C_: current AO=> Mo Transformation. (ao X mo)
                Returns:
                        [<Mux>,<Muy>,<Muz>]
                """
                # At this point convert both into MO and then calculate the dipole...
                rhoAO = TransMat(rho_,C_,-1)
                if (AA):
                        # first try in AO basis, if it does not work then in BO
                        e_dip = np.einsum('xij,ji->x', self.dip_ints[:,:nA,:nA], rhoAO[:nA,:nA])
                        if (np.any(self.pol0AA) != None):
                                return e_dip - self.pol0AA
                        else:
                                return e_dip
                else:
                        mol_dip = np.einsum('xij,ji->x', self.dip_ints, rhoAO)
                        if (np.any(self.pol0) != None):
                                return mol_dip - self.pol0
                        else:
                                return mol_dip
