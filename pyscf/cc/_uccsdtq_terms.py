# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

"""Connected spin-orbital CCSDTQ seeds for T1-dressed F and V.

All terms require the UNNORMALIZED external antisymmetrizer A_occ A_vir.
Integrals are g[p,q,r,s] = <pq||rs>, with pair antisymmetry only.  No
Hermitian pair interchange is assumed, since T1 dressing destroys it.
Amplitudes have occupied indices first.  i,j,k,l,m,n are occupied;
a,b,c,d,e,f are virtual.  Names identify diagram families and exact rational
coefficients.  Wick pairing counts are listed below the seeds.  The
determinant-space BCH reference in ``pyscf.cc.test.uccsdtq_reference``
independently checks the resulting residuals.
"""

from itertools import permutations

import numpy as np

TERMS = {2: (('V*T4/01', 1, 16, (('g', 'mnef'), ('t4', 'ijmnabef'))),),
    3: (('F*T4/01', 1, 36, (('f', 'me'), ('t4', 'ijkmabce'))),
     ('V*T4/01', 1, 24, (('g', 'amef'), ('t4', 'ijkmbcef'))),
     ('V*T4/02', 1, 24, (('g', 'mnei'), ('t4', 'jkmnabce')))),
    4: (('F*T4/01', -1, 144, (('f', 'ae'), ('t4', 'ijklbcde'))),
     ('F*T4/02', 1, 144, (('f', 'mi'), ('t4', 'jklmabcd'))),
     ('F*T2*T3/01', -1, 24, (('f', 'me'), ('t2', 'ijae'), ('t3', 'klmbcd'))),
     ('F*T2*T3/02', -1, 24, (('f', 'me'), ('t2', 'imab'), ('t3', 'jklcde'))),
     ('V*T3/01', -1, 24, (('g', 'abei'), ('t3', 'jklcde'))),
     ('V*T3/02', -1, 24, (('g', 'amij'), ('t3', 'klmbcd'))),
     ('V*T4/01', 1, 192, (('g', 'abef'), ('t4', 'ijklcdef'))),
     ('V*T4/02', -1, 36, (('g', 'amei'), ('t4', 'jklmbcde'))),
     ('V*T4/03', 1, 192, (('g', 'mnij'), ('t4', 'klmnabcd'))),
     ('V*T2*T2/01', -1, 16, (('g', 'abef'), ('t2', 'ijce'), ('t2', 'kldf'))),
     ('V*T2*T2/02', -1, 4, (('g', 'amei'), ('t2', 'jkbe'), ('t2', 'lmcd'))),
     ('V*T2*T2/03', -1, 16, (('g', 'mnij'), ('t2', 'kmab'), ('t2', 'lncd'))),
     ('V*T2*T3/01', -1, 8, (('g', 'amef'), ('t2', 'ijbe'), ('t3', 'klmcdf'))),
     ('V*T2*T3/02', -1, 48, (('g', 'amef'), ('t2', 'ijef'), ('t3', 'klmbcd'))),
     ('V*T2*T3/03', 1, 24, (('g', 'amef'), ('t2', 'imbc'), ('t3', 'jkldef'))),
     ('V*T2*T3/04', 1, 12, (('g', 'amef'), ('t2', 'imbe'), ('t3', 'jklcdf'))),
     ('V*T2*T3/05', 1, 24, (('g', 'mnei'), ('t2', 'jkae'), ('t3', 'lmnbcd'))),
     ('V*T2*T3/06', -1, 8, (('g', 'mnei'), ('t2', 'jmab'), ('t3', 'klncde'))),
     ('V*T2*T3/07', 1, 12, (('g', 'mnei'), ('t2', 'jmae'), ('t3', 'klnbcd'))),
     ('V*T2*T3/08', -1, 48, (('g', 'mnei'), ('t2', 'mnab'), ('t3', 'jklcde'))),
     ('V*T2*T4/01', -1, 48, (('g', 'mnef'), ('t2', 'ijae'), ('t4', 'klmnbcdf'))),
     ('V*T2*T4/02', 1, 384, (('g', 'mnef'), ('t2', 'ijef'), ('t4', 'klmnabcd'))),
     ('V*T2*T4/03', -1, 48, (('g', 'mnef'), ('t2', 'imab'), ('t4', 'jklncdef'))),
     ('V*T2*T4/04', 1, 36, (('g', 'mnef'), ('t2', 'imae'), ('t4', 'jklnbcdf'))),
     ('V*T2*T4/05', -1, 288, (('g', 'mnef'), ('t2', 'imef'), ('t4', 'jklnabcd'))),
     ('V*T2*T4/06', 1, 384, (('g', 'mnef'), ('t2', 'mnab'), ('t4', 'ijklcdef'))),
     ('V*T2*T4/07', -1, 288, (('g', 'mnef'), ('t2', 'mnae'), ('t4', 'ijklbcdf'))),
     ('V*T3*T3/01', 1, 48, (('g', 'mnef'), ('t3', 'ijkabe'), ('t3', 'lmncdf'))),
     ('V*T3*T3/02', 1, 144, (('g', 'mnef'), ('t3', 'ijkaef'), ('t3', 'lmnbcd'))),
     ('V*T3*T3/03', 1, 48, (('g', 'mnef'), ('t3', 'ijmabc'), ('t3', 'klndef'))),
     ('V*T3*T3/04', 1, 32, (('g', 'mnef'), ('t3', 'ijmabe'), ('t3', 'klncdf'))),
     ('V*T2*T2*T2/01', -1, 32, (('g', 'mnef'), ('t2', 'ijae'), ('t2', 'klbf'), ('t2', 'mncd'))),
     ('V*T2*T2*T2/02', 1, 4, (('g', 'mnef'), ('t2', 'ijae'), ('t2', 'kmbc'), ('t2', 'lndf'))),
     ('V*T2*T2*T2/03', -1, 32, (('g', 'mnef'), ('t2', 'ijef'), ('t2', 'kmab'), ('t2', 'lncd'))))}
# (rank, family, raw Wick pairings, tensor-normalized seeds, canonical seeds)
# (2, 'V*T4', 144, 1, 1)
# (3, 'F*T4', 16, 1, 1)
# (3, 'V*T4', 192, 2, 2)
# (4, 'F*T4', 8, 2, 2)
# (4, 'F*T2*T3', 12, 2, 2)
# (4, 'V*T3', 12, 2, 2)
# (4, 'V*T4', 88, 3, 3)
# (4, 'V*T2*T2', 48, 6, 3)
# (4, 'V*T2*T3', 312, 12, 8)
# (4, 'V*T2*T4', 752, 14, 7)
# (4, 'V*T3*T3', 828, 14, 4)
# (4, 'V*T2*T2*T2', 480, 36, 3)


def _parity(order):
    return (-1) ** sum(x > y for k, x in enumerate(order) for y in order[k+1:])


def antisymmetrize(tensor, rank):
    """Unnormalized separate external hole and particle antisymmetrizers."""
    # Apply the two sums successively: 2*n! transposes rather than (n!)**2.
    result = np.zeros_like(tensor)
    for p in permutations(range(rank)):
        result += _parity(p) * tensor.transpose(p + tuple(range(rank, 2*rank)))
    output = np.zeros_like(tensor)
    for p in permutations(range(rank)):
        output += _parity(p) * result.transpose(tuple(range(rank)) + tuple(rank+x for x in p))
    return output


def residuals(f, g, t2, t3, t4, ranks=(2, 3, 4), families=None):
    """Reference contractions: T4 feedback to R2/R3 and complete R4.

    f/g are the full T1-dressed, reference-normal-ordered coefficients.
    Optional family prefixes select diagram families for algebra tests.
    No iteration denominators, diagonal removal, or level shift are applied.
    """
    no, nv = t2.shape[0], t2.shape[2]
    tensors = {'f': f, 'g': g, 't2': t2, 't3': t3, 't4': t4}
    output = []
    for rank in ranks:
        unprojected = np.zeros((no,)*rank + (nv,)*rank)
        external = 'ijkl'[:rank] + 'abcd'[:rank]
        for name, numerator, denominator, factors in TERMS[rank]:
            if families is not None and name.split('/')[0] not in families:
                continue
            operands, labels = [], []
            for kind, indices in factors:
                operand = tensors[kind]
                if kind in ('f', 'g'):
                    slices = tuple(slice(0, no) if c in 'ijklmn' else slice(no, None)
                                   for c in indices)
                    operand = operand[slices]
                operands.append(operand)
                labels.append(indices)
            expression = ','.join(labels) + '->' + external
            unprojected += (numerator / denominator) * np.einsum(
                expression, *operands, optimize=True)
        output.append(antisymmetrize(unprojected, rank))
    return tuple(output)
