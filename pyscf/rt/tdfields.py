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

import pyscf
from pyscf import lib
import numpy as np
import scipy
import scipy.linalg
import math
from pyscf.rt import tdscf

class FIELDS(lib.StreamObject):
    """
    A class which manages field perturbations.

    Attributes:
        dip_ints: float
            Dipole matrix of a molecule
        pert_xyz: float
            Strength of perturbation in |x,y,z| direction
        dip0: float
            Initial Dipole moment of a molecule in |x y z| direction
        field_tol: float
            Field amplitude value at which it is considered no field

    """
    def __init__(self,rt, prm, field_tol = 10**-9.0):
        self.dip_ints = rt.ks.mol.intor("cint1e_r_sph", comp=3) # AO dipole integrals.
        self.field_tol = field_tol
        self.fieldamp = prm["FieldAmplitude"]
        self.tOn = prm["tOn"]
        self.tau = prm["Tau"]
        self.fieldfreq = prm["FieldFreq"]
        self.pert_xyz = np.array([prm["ExDir"],prm["EyDir"],prm["EzDir"]])
        self.dip0 = None
        self._keys = set(self.__dict__.keys())
        return

    def impulseamp(self,tnow):
        """
        Apply impulsive wave to the system
        Args:
            tnow: float
                Current time in A.U.
        Returns:
            amp: float
                Amplitude of field at time
            ison: bool
                On whether field is on or off

        """
        amp = self.fieldamp*np.sin(self.fieldfreq*tnow)*\
        (1.0/math.sqrt(2.0*3.1415*self.tau*self.tau))*\
        np.exp(-1.0*np.power(tnow-self.tOn,2.0)/(2.0*self.tau*self.tau))
        ison = False
        if (np.abs(amp)>self.field_tol):
            ison = True
        return amp,ison

    def initializeexpectation(self,rho0, c_am):
        """
        Calculate the initial dipole moment
        """
        self.dip0 = self.expectation(rho0,c_am)

    def applyfield(self, a_mat, tnow):
        """
        Args:
            a_mat: float or complex
                an AO matrix
            tnow: float
                current time.
        Returns:
            a_mat_field: float or complex
                an AO matrix with the field added
            ison: bool
                On whether field is on or off
        """
        amp, ison = self.impulseamp(tnow)
        mpol = self.pert_xyz * amp
        if (ison):
            a_mat_field = a_mat + 2.0*np.einsum("kij,k->ij",self.dip_ints,mpol)
            return a_mat_field, True
        else:
            a_mat_field = a_mat.copy()
            return a_mat, False

    def applyfield(self, a_mat, c_am, tnow):
        """
        Args:
            a_mat: float or complex
                an MO matrix
            c_am: float or complex
                Transformation Matrix |AO><MO|
            tnow: float
                current time.
        Returns:
            a_mat_field: float or complex
                an MO matrix with the field added
            ison: bool
                On whether field is on or off
        """
        amp, ison = self.impulseamp(tnow)
        mpol = self.pert_xyz * amp
        if (ison):
            a_mat_field = a_mat + 2.0*tdscf.transmat(\
            np.einsum("kij,k->ij",self.dip_ints,mpol),c_am)
            return a_mat_field, True
        else:
            a_mat_field = a_mat.copy()
            return a_mat, False

    def expectation(self, rho, c_am):
        """
        Args:
            rho: float or complex
                current MO density.
            c_am: float or complex
                Transformation Matrix |AO><MO|
        Returns:
            mu: float or complex
                dipole moment in |x y z| direction
        """
        rhoAO = tdscf.transmat(rho,c_am,-1)
        mol_dip = np.einsum('xij,ji->x', self.dip_ints, rhoAO)
        if (np.any(self.dip0) != None):
            mu = mol_dip - self.dip0
            return mu
        else:
            mu = mol_dip.copy()
            return mu
