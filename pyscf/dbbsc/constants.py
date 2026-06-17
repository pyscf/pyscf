#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

"""Numerical constants for DBBSC ECMD."""

import numpy

# Density threshold below which ECMD local quantities are set to zero.
RHO_CUTOFF = 1e-15
# On-top pair-density threshold below which the local interaction is skipped.
ONTOP_CUTOFF = 1e-30
# Maximum memory, in MB, used to size ECMD integration grid blocks.
DBBSC_BLOCK_MEMORY = 256
# Prefactor in mu_B(r) = sqrt(pi) W_B(r,r) / 2.
MU_PREFACTOR = 0.5 * numpy.sqrt(numpy.pi)
# Prefactor in the ECMD short-range interpolation beta parameter.
ECMD_BETA_PREFACTOR = 3.0 / (2.0 * numpy.sqrt(numpy.pi) * (1.0 - numpy.sqrt(2.0)))

# UEG on-top pair-density fit from Eq. 46 of 10.1103/PhysRevA.73.032506.
UEG_A_HD = -0.36583
UEG_C = 0.08193
UEG_D = -0.01277
UEG_E = 0.001859
UEG_EXPONENT = 0.7524
UEG_B = -2.0 * UEG_A_HD - UEG_EXPONENT
