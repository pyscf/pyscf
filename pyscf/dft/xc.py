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

'''
Python XC functional implementation, the backup module if neither libxc nor xcfun
libraries are not available
'''

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    r'''XC functional, potential and functional derivatives.
    '''
    raise NotImplementedError
