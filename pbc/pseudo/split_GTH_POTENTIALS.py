#!/usr/bin/python

import os
import re
import sys

def main():

    file_GTH = 'GTH_POTENTIALS'

    header = []
    is_header = True
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
                is_header = False
            else: 
                if is_header:
                    header.append(line)
                else:
                    current_pseudo.append(line)
        # The last one:
        all_pseudos.append(current_pseudo)

    print "Len of xcs =", len(xcs)
    print "Len of pseudos =", len(all_pseudos)

#    for line in header:
#        print line
#
    for xc, pseudo in zip(xcs, all_pseudos):
        with open('GTH_%s.dat'%(xc),'w') as f:
            for line in header:
                f.write(line)
            for line in pseudo:
                f.write(line)
        f.close()

if __name__ == '__main__':
    main()
