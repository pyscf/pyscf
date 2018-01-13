#!/usr/bin/python

import os
import re
import sys

def main():

    file_GTH = 'GTH_POTENTIALS'

    header = []
    is_header = True
    is_footer = False
    xcs = []
    all_pseudos = []
    current_pseudo = []
    with open(file_GTH,'r') as searchfile:
        for line in searchfile:
            if 'functional' in line:
                xc = line.split()[1]
                xcs.append(xc)
                if len(current_pseudo) > 0:
                    all_pseudos.append(current_pseudo)
                current_pseudo = []
                current_pseudo.append(line)
                banner_count = 1
                is_header = False
                is_footer = False
            else: 
                if is_header:
                    header.append(line)
                else:
                    if banner_count > 3 and '#####' in line:
                        current_pseudo.pop()
                        is_footer = True
                    if banner_count < 3 or is_footer:
                        current_pseudo.append(line)
                    else:
                        current_pseudo.append(line.replace('#','#PSEUDOPOTENTIAL'))
                    banner_count += 1
        # The last one:
        all_pseudos.append(current_pseudo)

    print("Found", len(xcs), "XC pseudopotentials.")

#    for line in header:
#        print(line)
#
    for xc, pseudo in zip(xcs, all_pseudos):
        with open('gth-%s.dat'%(xc.lower()),'w') as f:
            for line in header:
                f.write(line)
            for line in pseudo:
                f.write(line)
        f.close()

if __name__ == '__main__':
    main()
