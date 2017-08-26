#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import copy
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo

# Obtained from http://www.psicode.org/psi4manual/master/basissets_byfamily.html
DEFAULT_AUXBASIS = {
# AO basis JK-fit MP2-fit
'ccpvdz'      : ('ccpvdzjkfit'        , 'ccpvdzri'     ),
'ccpvdpdz'    : ('ccpvdzjkfit'        , 'ccpvdzri'     ),
'augccpvdz'   : ('augccpvdzjkfit'     , 'augccpvdzri'  ),
'augccpvdpdz' : ('augccpvdzjkfit'     , 'augccpvdzri'  ),
'ccpvtz'      : ('ccpvtzjkfit'        , 'ccpvtzri'     ),
'augccpvtz'   : ('augccpvtzjkfit'     , 'augccpvtzri'  ),
'ccpvqz'      : ('ccpvqzjkfit'        , 'ccpvqzri'     ),
'augccpvqz'   : ('augccpvqzjkfit'     , 'augccpvqzri'  ),
'ccpv5z'      : ('ccpv5zjkfit'        , 'ccpv5zri'     ),
'augccpv5z'   : ('augccpv5zjkfit'     , 'augccpv5zri'  ),
'def2svp'     : ('def2svpjkfit'       , 'def2svpri'    ),
'def2svp'     : ('def2svpjkfit'       , 'def2svpri'    ),
'def2svpd'    : ('def2svpjkfit'       , 'def2svpdri'   ),
'def2tzvp'    : ('def2tzvpjkfit'      , 'def2tzvpri'   ),
'def2tzvpd'   : ('def2tzvpjkfit'      , 'def2tzvpdri'  ),
'def2tzvpp'   : ('def2tzvppjkfit'     , 'def2tzvppri'  ),
'def2tzvppd'  : ('def2tzvppjkfit'     , 'def2tzvppdri' ),
'def2qzvp'    : ('def2qzvpjkfit'      , 'def2qzvpri'   ),
'def2qzvpd'   : ('def2qzvpjkfit'      , None           ),
'def2qzvpp'   : ('def2qzvppjkfit'     , 'def2qzvppri'  ),
'def2qzvppd'  : ('def2qzvppjkfit'     , 'def2qzvppdri' ),
'sto3g'       : ('def2svpjkfit'       , 'def2svprifit'     ),
'321g'        : ('def2svpjkfit'       , 'def2svprifit'     ),
'631g'        : ('ccpvdzjkfit'        , 'ccpvdzri'         ),
'631+g'       : ('heavyaugccpvdzjkfit', 'heavyaugccpvdzri' ),
'631++g'      : ('augccpvdzjkfit'     , 'augccpvdzri'      ),
'6311g'       : ('ccpvtzjkfit'        , 'ccpvtzri'         ),
'6311+g'      : ('heavyaugccpvtzjkfit', 'heavyaugccpvtzri' ),
'6311++g'     : ('augccpvtzjkfit'     , 'augccpvtzri'      ),
}

class load(ao2mo.load):
    '''load 3c2e integrals from hdf5 file
    Usage:
        with load(cderifile) as eri:
            print eri.shape
    '''
    def __init__(self, eri, dataname='j3c'):
        ao2mo.load.__init__(self, eri, dataname)


def aug_etb_for_dfbasis(mol, dfbasis='weigend', beta=2.3, start_at='Rb'):
    '''augment weigend basis with even tempered gaussian basis
    exps = alpha*beta^i for i = 1..N
    '''
    nuc_start = gto.mole._charge(start_at)
    uniq_atoms = set([a[0] for a in mol._atom])

    newbasis = {}
    for symb in uniq_atoms:
        nuc_charge = gto.mole._charge(symb)
        if nuc_charge < nuc_start:
            newbasis[symb] = dfbasis
        #?elif symb in mol._ecp:
        else:
            conf = lib.parameters.ELEMENTS[nuc_charge][2]
            max_shells = 4 - conf.count(0)
            emin_by_l = [1e99] * 8
            emax_by_l = [0] * 8
            for b in mol._basis[symb]:
                l = b[0]
                if l >= max_shells+1:
                    continue

                if isinstance(b[1], int):
                    e_c = numpy.array(b[2:])
                else:
                    e_c = numpy.array(b[1:])
                es = e_c[:,0]
                cs = e_c[:,1:]
                es = es[abs(cs).max(axis=1) > 1e-3]
                emax_by_l[l] = max(es.max(), emax_by_l[l])
                emin_by_l[l] = min(es.min(), emin_by_l[l])

            l_max = 8 - emax_by_l.count(0)
            emin_by_l = numpy.array(emin_by_l[:l_max])
            emax_by_l = numpy.array(emax_by_l[:l_max])
# Estimate the exponents ranges by geometric average
            emax = numpy.sqrt(numpy.einsum('i,j->ij', emax_by_l, emax_by_l))
            emin = numpy.sqrt(numpy.einsum('i,j->ij', emin_by_l, emin_by_l))
            liljsum = numpy.arange(l_max)[:,None] + numpy.arange(l_max)
            emax_by_l = [emax[liljsum==ll].max() for ll in range(l_max*2-1)]
            emin_by_l = [emin[liljsum==ll].min() for ll in range(l_max*2-1)]
            # Tune emin and emax
            emin_by_l = numpy.array(emin_by_l) * 2  # *2 for alpha+alpha on same center
            emax_by_l = numpy.array(emax_by_l) * 2  #/ (numpy.arange(l_max*2-1)*.5+1)

            ns = numpy.log((emax_by_l+emin_by_l)/emin_by_l) / numpy.log(beta)
            etb = [(l, max(n,1), emin_by_l[l], beta)
                   for l, n in enumerate(numpy.ceil(ns).astype(int))]
            newbasis[symb] = gto.expand_etbs(etb)

    return newbasis

def aug_etb(mol, beta=2.3):
    return aug_etb_for_dfbasis(mol, beta=beta, start_at=0)

def make_auxbasis(mol, mp2fit=False):
    '''Even-tempered Gaussians or the DF basis in DEFAULT_AUXBASIS'''
    uniq_atoms = set([a[0] for a in mol._atom])
    if isinstance(mol.basis, str):
        _basis = dict(((a, mol.basis) for a in uniq_atoms))
    elif 'default' in mol.basis:
        default_basis = mol.basis['default']
        _basis = dict(((a, default_basis) for a in uniq_atoms))
        _basis.update(mol.basis)
        del(_basis['default'])
    else:
        _basis = mol.basis

    auxbasis = {}
    for k in _basis:
        if isinstance(_basis[k], str):
            balias = gto.basis._format_basis_name(_basis[k])
            if gto.basis._is_pople_basis(balias):
                balias = balias.split('g')[0] + 'g'
            if balias in DEFAULT_AUXBASIS:
                if mp2fit:
                    auxb = DEFAULT_AUXBASIS[balias][1]
                else:
                    auxb = DEFAULT_AUXBASIS[balias][0]
                if auxb is not None:
                    auxbasis[k] = auxb

    if len(auxbasis) != len(_basis):
        # Some AO basis not found in DEFAULT_AUXBASIS
        auxbasis = aug_etb(mol).update(auxbasis)
    return auxbasis

def make_auxmol(mol, auxbasis='weigend+etb'):
    '''Generate a fake Mole object which uses the density fitting auxbasis as
    the basis sets
    '''
    pmol = copy.copy(mol)  # just need shallow copy

    if auxbasis is None:
        auxbasis = make_auxbasis(mol)
    elif '+etb' in auxbasis:
        dfbasis = auxbasis[:-4]
        auxbasis = aug_etb_for_dfbasis(mol, dfbasis)
    pmol.basis = auxbasis

    if isinstance(auxbasis, (str, unicode, list, tuple)):
        uniq_atoms = set([a[0] for a in mol._atom])
        _basis = dict([(a, auxbasis) for a in uniq_atoms])
    elif 'default' in auxbasis:
        uniq_atoms = set([a[0] for a in mol._atom])
        _basis = dict(((a, auxbasis['default']) for a in uniq_atoms))
        _basis.update(auxbasis)
        del(_basis['default'])
    else:
        _basis = auxbasis
    pmol._basis = pmol.format_basis(_basis)

    pmol._atm, pmol._bas, pmol._env = \
            pmol.make_env(mol._atom, pmol._basis, mol._env[:gto.PTR_ENV_START])
    pmol._built = True
    logger.debug(mol, 'aux basis %s, num shells = %d, num cGTOs = %d',
                 auxbasis, pmol.nbas, pmol.nao_nr())
    return pmol
