# Copyright (c) 2020, Susi Lehtola
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
# names of its contributors may be used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# Routines for the implementation of the superposition of atomic
# potentials guess for electronic structure calculations, see
#
# S. Lehtola, "Assessment of Initial Guesses for Self-Consistent Field
# Calculations. Superposition of Atomic Potentials: Simple yet
# Efficient", J. Chem. Theory Comput. 15, 1593 (2019).
# DOI: 10.1021/acs.jctc.8b01089
#
# This function evaluates the effective charge of a neutral atom,
# given by exchange-only LDA on top of spherically symmetric
# unrestricted Hartree-Fock calculations as described in
#
# S. Lehtola, L. Visscher, E. Engel, Efficient implementation of the
# superposition of atomic potentials initial guess for electronic
# structure calculations in Gaussian basis sets, J. Chem. Phys., in
# press (2020).
#
# The potentials have been calculated for the ground-states of
# spherically symmetric atoms at the non-relativistic level of theory
# as described in
#
# S. Lehtola, "Fully numerical calculations on atoms with fractional
# occupations and range-separated exchange functionals", Phys. Rev. A
# 101, 012516 (2020). DOI: 10.1103/PhysRevA.101.012516
#
# using accurate finite-element calculations as described in
#
# S. Lehtola, "Fully numerical Hartree-Fock and density functional
# calculations. I. Atoms", Int. J. Quantum Chem. e25945 (2019).
# DOI: 10.1002/qua.25945

import numpy
from pyscf.dft.sap_data import sap_Zeff

def sap_effective_charge(Z, r):
    '''
    Calculates the effective charge for the superposition of atomic potentials.

    S. Lehtola, "Assessment of Initial Guesses for Self-Consistent Field
    Calculations. Superposition of Atomic Potentials: Simple yet
    Efficient", J. Chem. Theory Comput. 15, 1593 (2019).
    DOI: 10.1021/acs.jctc.8b01089

    This function evaluates the effective charge of a neutral atom,
    given by exchange-only LDA on top of spherically symmetric
    unrestricted Hartree-Fock calculations as described in

    S. Lehtola, L. Visscher, E. Engel, Efficient implementation of the
    superposition of atomic potentials initial guess for electronic
    structure calculations in Gaussian basis sets, J. Chem. Phys., in
    press (2020).

    The potentials have been calculated for the ground-states of
    spherically symmetric atoms at the non-relativistic level of theory
    as described in

    S. Lehtola, "Fully numerical calculations on atoms with fractional
    occupations and range-separated exchange functionals", Phys. Rev. A
    101, 012516 (2020). DOI: 10.1103/PhysRevA.101.012516

    using accurate finite-element calculations as described in

    S. Lehtola, "Fully numerical Hartree-Fock and density functional
    calculations. I. Atoms", Int. J. Quantum Chem. e25945 (2019).
    DOI: 10.1002/qua.25945

    Input:
       Z: atomic charge
       r: distance from nucleus
    Output:
       Z(r): screened charge
    '''

    if Z < 1:
        return 0.0
    if Z >= sap_Zeff.shape[1]:
        raise ValueError('Atoms beyond Og are not supported')
    if numpy.any(r < 0.0):
        raise ValueError('Distance cannot be negative')

    if r.ndim == 0:
        if r >= sap_Zeff.shape[0]:
            return 0.0
        # Linear interpolation
        return numpy.interp(r, sap_Zeff[0,:], sap_Zeff[Z,:])
    else:
        v = numpy.interp(r, sap_Zeff[0,:], sap_Zeff[Z,:])
        v[r >= sap_Zeff.shape[0]] = 0.
        return v
