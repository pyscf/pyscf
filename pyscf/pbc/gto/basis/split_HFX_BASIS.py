#!/usr/bin/python
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

import os
import re
import sys
from collections import OrderedDict

def main():

    if len(sys.argv) > 1:
        file_GTH = sys.argv[1]
    else:
        file_GTH = 'HFX_BASIS'

    basis_sets = OrderedDict()
    with open(file_GTH,'r') as searchfile:
        for line in searchfile:
            if line.startswith('#'):
                continue
            elif 'GTH' in line:
                bas_type = line.split()[1]
                if bas_type not in basis_sets:
                    basis_sets[bas_type] = []
                basis_sets[bas_type].append(line)
            else: 
                basis_sets[bas_type].append(line)

    for basis_set in basis_sets:
        with open('gth-%s.dat'%(basis_set.lower().replace('-gth','')),'w') as f:
            lines = basis_sets[basis_set]
            for line in lines:
                if 'GTH' in line:
                    f.write('#BASIS SET\n')
                f.write(line)
            f.write('END\n')
        f.close()

if __name__ == '__main__':
    main()
