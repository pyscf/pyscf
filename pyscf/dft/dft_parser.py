
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

'''
unified dft parser for coordinating dft protocols with
1. xc functionals
2. dispersion corrections / nonlocal correction
3. GTO basis (TODO)
4. geometrical counterpoise (gCP) correction (TODO)
'''

from functools import lru_cache
import warnings

# supported dispersion corrections
DISP_VERSIONS = ['d3bj', 'd3zero', 'd3bjm', 'd3zerom', 'd3op', 'd4']

@lru_cache(128)
def parse_dft(dft_method):
    ''' conventional dft method ->
    (xc, enable nlc, (xc for dftd3, dispersion version, with 3-body dispersion))
    '''
    if not isinstance(dft_method, str):
        return dft_method, None, (dft_method, None, False)
    method_lower = dft_method.lower()

    # special cases:
    # - wb97x-d is not supported yet
    # - wb97*-d3bj is wb97*-v with d3bj
    # - wb97x-d3 is not supported yet
    # - 3c method is not supported yet

    if method_lower == 'wb97x-d':
        raise NotImplementedError('wb97x-d is not supported yet.')

    if method_lower == 'wb97m-d3bj':
        return 'wb97m-v', False, ('wb97m', 'd3bj', False)
    if method_lower == 'b97m-d3bj':
        return 'b97m-v', False, ('b97m', 'd3bj', False)
    if method_lower == 'wb97x-d3bj':
        return 'wb97x-v', False, ('wb97x', 'd3bj', False)

    # J. Chem. Theory Comput. 2013, 9, 1, 263-272
    if method_lower in ['wb97x-d3']:
        raise NotImplementedError('wb97x-d3 is not supported yet.')

    if method_lower.endswith('-3c'):
        raise NotImplementedError('*-3c methods are not supported yet.')

    xc = dft_method
    disp = None
    for d in DISP_VERSIONS:
        if method_lower.endswith(d):
            disp = d
            with_3body = False
            xc = method_lower.replace(f'-{d}','')
        elif method_lower.endswith(d+'2b'):
            disp = d
            with_3body = False
            xc = method_lower.replace(f'-{d}2b', '')
        elif method_lower.endswith(d+'atm'):
            disp = d
            with_3body = True
            xc = method_lower.replace(f'-{d}atm', '')

        if disp is not None:
            if xc in ('b97m', 'wb97m'):
                warnings.warn(
                    f'{dft_method} is not a well-defined functional. '
                    'The XC part is changed to {xc}-v')
                return xc+'-v', False, (xc, disp, with_3body)
            else:
                return xc, None, (xc, disp, with_3body)

    return xc, None, (xc, None, False)
