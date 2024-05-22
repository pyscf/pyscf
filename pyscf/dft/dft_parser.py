
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

# special cases:
# - wb97x-d is not supported yet
# - wb97*-d3bj is wb97*-v with d3bj
# - wb97x-d3 is not supported yet
# - 3c method is not supported yet

# These xc functionals need special treatments
white_list = {
    'wb97m-d3bj': ('wb97m-v', False, 'd3bj'),
    'b97m-d3bj': ('b97m-v', False, 'd3bj'),
    'wb97x-d3bj': ('wb97x-v', False, 'd3bj'),
}

# These xc functionals are not supported yet
black_list = {
    'wb97x-d', 'wb97x-d3',
    'wb97m-d3bj2b', 'wb97m-d3bjatm',
    'b97m-d3bj2b', 'b97m-d3bjatm',
}

@lru_cache(128)
def parse_dft(dft_method):
    ''' conventional dft method -> (xc, nlc, disp)
    '''
    if not isinstance(dft_method, str):
        return dft_method, '', None
    method_lower = dft_method.lower()

    if method_lower in black_list:
        raise NotImplementedError(f'{method_lower} is not supported yet.')

    if method_lower in white_list:
        return white_list[method_lower]

    if method_lower.endswith('-3c'):
        raise NotImplementedError('*-3c methods are not supported yet.')

    if '-d3' in method_lower or '-d4' in method_lower:
        xc, disp = method_lower.split('-')
    else:
        xc, disp = method_lower, None

    return xc, '', disp
