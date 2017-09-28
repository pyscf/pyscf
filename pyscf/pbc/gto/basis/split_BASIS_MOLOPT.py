#!/usr/bin/python

import os
import re
import sys
from collections import OrderedDict

def main():

    file_GTH = 'BASIS_MOLOPT'

    basis_sets = OrderedDict()
    with open(file_GTH,'r') as searchfile:
        for line in searchfile:
            if line[0] == '#':
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
