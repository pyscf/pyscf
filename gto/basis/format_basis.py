#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''download the basis sets TX93(optimized general contractions) from EMSL
    https://bse.pnl.gov/bse/portal
and format them to the proper format'''

__author__ = "Qiming Sun <osirpt.sun@gmail.com>"
__version__ = "$ 0.2 $"

import os, sys

def cmd_args():
    """
    get input from cmdline
    """
    import optparse
    usage = "Usage: %prog [options] arg1 arg2"
    parser = optparse.OptionParser(usage=usage, version=__version__)
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose",
                      help="make lots of noise [default]")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose",
                      help="be very quiet")
    parser.add_option("-i", "--input",
                      dest="fin", metavar="FILE", help="input should be in TX93 format")
    parser.add_option("-o", "--output",
                      dest="fout", metavar="FILE", help="write output to FILE")

    (opts_args, args_left) = parser.parse_args()

    check_args(opts_args.fin, opts_args.fout)
    return opts_args.fin, opts_args.fout

def check_args(fin, fout):
    assert(os.path.lexists(fin))
    try:
        if os.path.lexists(fout):
            print 'backup file  %s  as  %s.bak' % (fout, fout)
            os.rename(fout, fout + '.bak')
    except:
        pass
    return fin, fout

def dump(fin, fout):
    MAPSPDF = {'S': '0,', \
               'P': '1,', \
               'D': '2,', \
               'F': '3,', \
               'G': '4,', \
               'H': '5,'}
    fdin = open(fin, 'r')
    if fout is None:
        fdout = sys.stdout
    else:
        fdout = open(fout, 'w')
    fdout.write('# data from EMSL:  https://bse.pnl.gov/bse/portal\n\n')
    l = fdin.readline()
    while l:
        if l[0] == '!':
            fdout.write('#%s' % l[1:])
            l = fdin.readline()
        elif l == '\n':
            fdout.write(l)
            l = fdin.readline()
        elif l[0:3] == 'FOR':
            elem = l.split()[1]
            fdout.write('\'%s\':[' % elem)
            is_last_coeff = False
            l = fdin.readline()
            while l and l[0:3] != 'FOR':
                ls = l.split()
                if ls[0] in 'SPDFGH':
                    if is_last_coeff:
                        fdout.write('],\n      [%s' % MAPSPDF[ls[0]])
                    else:
                        fdout.write('[%s' % MAPSPDF[ls[0]])
                        is_last_coeff = True
                    d = ', '.join(ls[1:])
                else:
                    d = ', '.join(ls)
                fdout.write('\n        (%s),' % d.replace('D', 'e'))
                l = fdin.readline()
            fdout.write(']],\n')
    fdin.close()
    fdout.close()

if __name__ == "__main__":
    fin, fout = cmd_args()
    dump(fin, fout)
