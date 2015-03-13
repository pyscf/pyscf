#!/usr/bin/env python

def dump_tri(stdout, c, label=None, ncol=5, digits=5, start=0):
    nc = c.shape[1]
    for ic in range(0, nc, ncol):
        dc = c[:,ic:ic+ncol]
        m = dc.shape[1]
        fmt = (' %%%d.%df'%(digits+4,digits))*m + '\n'
        if label is None:
            stdout.write(((' '*(digits+3))+'%s\n') % \
                         (' '*(digits)).join(['#%-4d'%i for i in range(start+ic,start+ic+m)]))
            for k, v in enumerate(dc[ic:ic+m]):
                fmt = (' %%%d.%df'%(digits+4,digits))*(k+1) + '\n'
                stdout.write(('%-5d' % (ic+k+start)) + (fmt % tuple(v[:k+1])))
            for k, v in enumerate(dc[ic+m:]):
                stdout.write(('%-5d' % (ic+m+k+start)) + (fmt % tuple(v)))
        else:
            stdout.write(((' '*(digits+10))+'%s\n') % \
                         (' '*(digits)).join(['#%-4d'%i for i in range(start+ic,start+ic+m)]))
            #stdout.write('           ')
            #stdout.write(((' '*(digits)+'#%-5d')*m) % tuple(range(ic+start,ic+m+start)) + '\n')
            for k, v in enumerate(dc[ic:ic+m]):
                fmt = (' %%%d.%df'%(digits+4,digits))*(k+1) + '\n'
                stdout.write(('%12s' % label[ic+k]) + (fmt % tuple(v[:k+1])))
            for k, v in enumerate(dc[ic+m:]):
                stdout.write(('%12s' % label[ic+m+k]) + (fmt % tuple(v)))

def dump_rec(stdout, c, label=None, label2=None, ncol=5, digits=5, start=0):
    nc = c.shape[1]
    if label2 is None:
        fmt = '#%%-%dd' % (digits+3)
        label2 = [fmt%i for i in range(start,nc+start)]
    else:
        fmt = '%%-%ds' % (digits+4)
        label2 = [fmt%i for i in label2]
    for ic in range(0, nc, ncol):
        dc = c[:,ic:ic+ncol]
        m = dc.shape[1]
        fmt = (' %%%d.%df'%(digits+4,digits))*m + '\n'
        if label is None:
            stdout.write(((' '*(digits+3))+'%s\n') % ' '.join(label2[ic:ic+m]))
            for k, v in enumerate(dc):
                stdout.write(('%-5d' % (k+start)) + (fmt % tuple(v)))
        else:
            stdout.write(((' '*(digits+10))+'%s\n') % ' '.join(label2[ic:ic+m]))
            for k, v in enumerate(dc):
                stdout.write(('%12s' % label[k]) + (fmt % tuple(v)))

def dump_mo(mol, c):
    label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
    dump_rec(mol.stdout, c, label, start=1)


if __name__ == '__main__':
    import sys
    import numpy
    c = numpy.random.random((16,16))
    label = ['A%5d' % i for i in range(16)]
    dump_tri(sys.stdout, c, label, 10, 2, 1)
    dump_rec(sys.stdout, c, None, label, start=1)
