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

"""Backward-compatible private imports for DBBSC ECMD helpers."""

from pyscf.dbbsc.functional import FunctionalSettings, _resolve_functional
from pyscf.dbbsc.reference import (
    ReferenceKind,
    _check_active_mask,
    _frozen_orbital_mask,
    _get_scf_method,
    _reference_kind,
)
from pyscf.dbbsc.workspace import _GridBlockWorkspace

__all__ = [
    'FunctionalSettings',
    'ReferenceKind',
    '_GridBlockWorkspace',
    '_check_active_mask',
    '_frozen_orbital_mask',
    '_get_scf_method',
    '_reference_kind',
    '_resolve_functional',
]
