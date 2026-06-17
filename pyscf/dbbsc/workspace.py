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

"""Reusable scratch arrays for DBBSC grid blocks."""

import numpy


class _GridBlockWorkspace:
    def __init__(self):
        self._arrays = {}

    def empty(self, name, shape):
        out = self._arrays.get(name)
        if out is None or out.shape != shape:
            out = numpy.empty(shape)
            self._arrays[name] = out
        return out

    def pair_values(self, name, occ_g, mo_g):
        out = self.empty(name, (occ_g.shape[0], occ_g.shape[1] * mo_g.shape[1]))
        numpy.multiply(
            occ_g[:, :, None],
            mo_g[:, None, :],
            out=out.reshape(occ_g.shape[0], occ_g.shape[1], mo_g.shape[1]),
        )
        return out
