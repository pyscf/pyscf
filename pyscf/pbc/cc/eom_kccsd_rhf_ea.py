#!/usr/bin/env python
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
#
# Author: Artem Pulkin
from pyscf.pbc.lib.kpts_helper import nested_to_vector, vector_to_nested


def ea_vector_desc(cc):
    """Description of the EA vector."""
    nvir = cc.nmo - cc.nocc
    return [(nvir,), (cc.nkpts, cc.nkpts, cc.nocc, nvir, nvir)]


def ea_amplitudes_to_vector(cc, t1, t2):
    """Ground state amplitudes to a vector."""
    return nested_to_vector((t1, t2))[0]


def ea_vector_to_amplitudes(cc, vec):
    """Ground state vector to apmplitudes."""
    return vector_to_nested(vec, ea_vector_desc(cc))