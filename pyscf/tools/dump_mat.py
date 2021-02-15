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

from pyscf.lib.parameters import OUTPUT_DIGITS, OUTPUT_COLS
from pyscf import __config__

BASE = getattr(__config__, 'BASE', 0)

def dump_tri(stdout, c, label=None,
             ncol=OUTPUT_COLS, digits=OUTPUT_DIGITS, start=BASE):
    ''' Format print for the lower triangular part of an array

    Args:
        stdout : file object
            eg sys.stdout, or stdout = open('/path/to/file') or
            mol.stdout if mol is an object initialized from :class:`gto.Mole`
        c : numpy.ndarray
            coefficients

    Kwargs:
        label : list of strings
            Row labels (default is 1,2,3,4,...)
        ncol : int
            Number of columns in the format output (default 5)
        digits : int
            Number of digits of precision for floating point output (default 5)
        start : int
            The number to start to count the index (default 0)

    Examples:

        >>> import sys, numpy
        >>> dm = numpy.eye(3)
        >>> dump_tri(sys.stdout, dm)
                #0        #1        #2   
        0       1.00000
        1       0.00000   1.00000
        2       0.00000   0.00000   1.00000
        >>> from pyscf import gto
        >>> mol = gto.M(atom='C 0 0 0')
        >>> dm = numpy.eye(mol.nao_nr())
        >>> dump_tri(sys.stdout, dm, label=mol.ao_labels(), ncol=9, digits=2)
                    #0     #1     #2     #3     #4     #5     #6     #7     #8   
        0  C 1s     1.00
        0  C 2s     0.00   1.00
        0  C 3s     0.00   0.00   1.00
        0  C 2px    0.00   0.00   0.00   1.00
        0  C 2py    0.00   0.00   0.00   0.00   1.00
        0  C 2pz    0.00   0.00   0.00   0.00   0.00   1.00
        0  C 3px    0.00   0.00   0.00   0.00   0.00   0.00   1.00
        0  C 3py    0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.00
        0  C 3pz    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.00
    '''
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

def dump_rec(stdout, c, label=None, label2=None,
             ncol=OUTPUT_COLS, digits=OUTPUT_DIGITS, start=BASE):
    ''' Print an array in rectangular format

    Args:
        stdout : file object
            eg sys.stdout, or stdout = open('/path/to/file') or
            mol.stdout if mol is an object initialized from :class:`gto.Mole`
        c : numpy.ndarray
            coefficients

    Kwargs:
        label : list of strings
            Row labels (default is 1,2,3,4,...)
        label2 : list of strings
            Col labels (default is 1,2,3,4,...)
        ncol : int
            Number of columns in the format output (default 5)
        digits : int
            Number of digits of precision for floating point output (default 5)
        start : int
            The number to start to count the index (default 0)

    Examples:

        >>> import sys, numpy
        >>> dm = numpy.eye(3)
        >>> dump_rec(sys.stdout, dm)
                #0        #1        #2   
        0       1.00000   0.00000   0.00000
        1       0.00000   1.00000   0.00000
        2       0.00000   0.00000   1.00000
        >>> from pyscf import gto
        >>> mol = gto.M(atom='C 0 0 0')
        >>> dm = numpy.eye(mol.nao_nr())
        >>> dump_rec(sys.stdout, dm, label=mol.ao_labels(), ncol=9, digits=2)
                    #0     #1     #2     #3     #4     #5     #6     #7     #8   
        0  C 1s     1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
        0  C 2s     0.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
        0  C 3s     0.00   0.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00
        0  C 2px    0.00   0.00   0.00   1.00   0.00   0.00   0.00   0.00   0.00
        0  C 2py    0.00   0.00   0.00   0.00   1.00   0.00   0.00   0.00   0.00
        0  C 2pz    0.00   0.00   0.00   0.00   0.00   1.00   0.00   0.00   0.00
        0  C 3px    0.00   0.00   0.00   0.00   0.00   0.00   1.00   0.00   0.00
        0  C 3py    0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.00   0.00
        0  C 3pz    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.00
    '''
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

def dump_mo(mol, c, label=None,
            ncol=OUTPUT_COLS, digits=OUTPUT_DIGITS, start=BASE):
    ''' Format print for orbitals

    Args:
        stdout : file object
            eg sys.stdout, or stdout = open('/path/to/file') or
            mol.stdout if mol is an object initialized from :class:`gto.Mole`
        c : numpy.ndarray
            Orbitals, each column is an orbital

    Kwargs:
        label : list of strings
            Row labels (default is AO labels)

    Examples:

        >>> from pyscf import gto
        >>> mol = gto.M(atom='C 0 0 0')
        >>> mo = numpy.eye(mol.nao_nr())
        >>> dump_mo(mol, mo)
                    #0     #1     #2     #3     #4     #5     #6     #7     #8   
        0  C 1s     1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
        0  C 2s     0.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
        0  C 3s     0.00   0.00   1.00   0.00   0.00   0.00   0.00   0.00   0.00
        0  C 2px    0.00   0.00   0.00   1.00   0.00   0.00   0.00   0.00   0.00
        0  C 2py    0.00   0.00   0.00   0.00   1.00   0.00   0.00   0.00   0.00
        0  C 2pz    0.00   0.00   0.00   0.00   0.00   1.00   0.00   0.00   0.00
        0  C 3px    0.00   0.00   0.00   0.00   0.00   0.00   1.00   0.00   0.00
        0  C 3py    0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.00   0.00
        0  C 3pz    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.00
    '''
    if label is None:
        label = mol.ao_labels()
    dump_rec(mol.stdout, c, label, None, ncol, digits, start)

del(BASE)


if __name__ == '__main__':
    import sys
    import numpy
    c = numpy.random.random((16,16))
    label = ['A%5d' % i for i in range(16)]
    dump_tri(sys.stdout, c, label, 10, 2, 1)
    dump_rec(sys.stdout, c, None, label, start=1)
