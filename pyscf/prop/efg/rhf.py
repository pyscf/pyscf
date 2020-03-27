#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Electric field gradients, nuclear quadrupolar coupling and Mossbauer
spectroscopy for non-relativistic (or sf-x2c) mean-field and post-HF methods.
(In testing)

Ref:

[1] J. Autschbach, S. Zheng, and R. Schurko. Concepts in Magnetic Resonance Part A, 36A, 126 (2010)

[2] H. Petrilli, P. Blochl, P. Blaha, and K. Schwarz. Phys. Rev. B, 57, 14690 (1998)

[3] S. Adiga, D. Aebi, and D. Bryce. Can. J. Chem, 85, 496 (2007)

[4] http://www.cmp.liv.ac.uk/frink/thesis/thesis/node18.html
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import x2c
from pyscf.data import nist
from pyscf.data import elements
from pyscf.data.nucprop import ISOTOPE_QUAD_MOMENT

def kernel(method, efg_nuc=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** EFG for non-relativistic methods (In testing) ********')
    mol = method.mol
    if efg_nuc is None:
        efg_nuc = range(mol.natm)

    dm = method.make_rdm1(ao_repr=True)
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF density matrix
        dm = dm[0] + dm[1]

    if isinstance(method, scf.hf.SCF):
        with_x2c = getattr(method, 'with_x2c', None)
    else:
        with_x2c = getattr(method._scf, 'with_x2c', None)
    if with_x2c:
        xmol, contr_coeff = with_x2c.get_xmol(mol)
        xmat = with_x2c.get_xmat(xmol)
        # rmat transforms integrals from \tilde{S} metric to S metric
        c = lib.param.LIGHT_SPEED
        s = xmol.intor_symmetric('int1e_ovlp')
        t = xmol.intor_symmetric('int1e_kin')
        s1 = s + reduce(numpy.dot, (xmat.T, t, xmat)) * (.5/c**2)
        rmat = x2c.x2c._get_r(s, s1)
        if contr_coeff is not None:
            rmat = numpy.dot(rmat, contr_coeff)

    log.info('\nElectric Field Gradient Tensor Results')
    efg = []
    for i, atm_id in enumerate(efg_nuc):
        # The electronic quadrupole operator (3 \vec{r} \vec{r} - r^2) / r^5
        # Note the difference to HFC tensor. Spin density matrix was used in
        # HFC tensor.
        h1 = _get_quadrupole_integrals(mol, atm_id)

        if with_x2c:
            # J. Autschbach, D. Peng, and R. Markus. JCTC, 8, 4239 (2012)
            h1SS = _get_sfx2c_quadrupole_integrals(xmol, atm_id)
            h1 += lib.einsum('xypq,pi,qj->xyij', h1SS, rmat, rmat) * (.25/c**2)

        efg_e = numpy.einsum('xyij,ji->xy', h1, dm)
        efg_nuc = _get_quad_nuc(mol, atm_id)
        v = efg_nuc - efg_e
        efg.append(v)

        _analyze(mol, atm_id, v, log)

    return numpy.asarray(efg)

def _analyze(mol, atm_id, v, log):
    Z = mol.atom_charge(atm_id)
    symb = mol.atom_symbol(atm_id)
    stdsymb = elements.ELEMENTS[Z]

    # Ground-state quadrupole moment is non zero only if the nucleus has I > 1/2.
    # CQ is nuclear quadruplar coupling constant. Q is nuclear electric
    # quadrupole moment.
    isotope, I, Q = ISOTOPE_QUAD_MOMENT[Z]
    if mol.nucprop:
        if atm_id+1 in mol.nucprop:
            prop = mol.nucprop[atm_id+1]
        elif symb in mol.nucprop:
            prop = mol.nucprop[symb]
        elif stdsymb in mol.nucprop:
            prop = mol.nucprop[stdsymb]
        else:
            prop = {}
        isotope = prop.get('isotope', isotope)
        I = prop.get('I', I)
        Q = prop.get('Q', Q)

    log.info('--\nEFG for %d %s: Isotope %s  I = %g  Q = %g barn',
             atm_id, symb, isotope, I, Q)

    # Principal axis system and asymmetry parameter
    e, pas = numpy.linalg.eigh(v)
    # The principle components are ordered Vzz > Vxx >= Vyy
    if abs(e[2]) > abs(e[0]):
        Vyy, Vxx, Vzz = e
    else:
        Vyy, Vxx, Vzz = e[::-1]
        pas = pas[:,::-1]

    log.info('EFG eigen (au)        PAS')
    log.info('Vxx %.9g  %s', Vxx, pas[:,0])
    log.info('Vyy %.9g  %s', Vyy, pas[:,1])
    log.info('Vzz %.9g  %s', Vzz, pas[:,2])

    eta = (Vxx - Vyy) / Vzz
    log.info('Quadrupolar asymmetry parameter %.9g', eta)

    # 1 barn = 1e-28 m^2
    au2MHz = nist.E_CHARGE * nist.AUEFG / nist.PLANCK * 1e-6 * 1e-28

    if stdsymb in ('Fe', 'Sn'):
        # Mossbauer spectroscopy
        # Eq 2.14 of http://www.cmp.liv.ac.uk/frink/thesis/thesis/node18.html
        qQ = 0.5 * Vzz * Q * au2MHz
        Delta = qQ * (1 + eta**2/3)**.5
        log.info('Quadrupolar coupling constant %.9g MHz', 2*qQ)
        log.info('e^2qQ/2 = %.9g MHz', qQ)
        log.info('Delta = e^2qQ/2(1+eta^2/3)^.5 = %.9g MHz', Delta)
    elif I > 0.5:
        # EFG has no effect on the I = 1/2 ground state
        CQ = Vzz * Q * au2MHz
        log.info('Quadrupolar coupling constant %.9g MHz', CQ)

# TODO: Euler angles analysis for EFG tensor with nuclear shield tensor
# Can. J. Chem, 85, 496 (2007)

def _get_quad_nuc(mol, atm_id):
    mask = numpy.ones(mol.natm, dtype=bool)
    mask[atm_id] = False  # exclude the contribution from atm_id
    dr = mol.atom_coords()[mask] - mol.atom_coord(atm_id)
    d = numpy.linalg.norm(dr, axis=1)
    rr = 3*numpy.einsum('ix,iy->ixy', dr, dr)
    for i in range(3):
        rr[:,i,i] -= d**2
    z = mol.atom_charges()[mask]
    efg_nuc = numpy.einsum('i,ixy->xy', z/d**5, rr)
    return efg_nuc

def _get_quadrupole_integrals(mol, atm_id):
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        # Compute the integrals of quadrupole operator 
        # (3 \vec{r} \vec{r} - r^2) / r^5
        ipipv = mol.intor('int1e_ipiprinv', 9).reshape(3,3,nao,nao)
        ipvip = mol.intor('int1e_iprinvip', 9).reshape(3,3,nao,nao)
        h1ao = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1ao = h1ao + h1ao.transpose(0,1,3,2)

    coords = mol.atom_coord(atm_id).reshape(1, 3)
    ao = mol.eval_gto('GTOval', coords)
    fc = 4*numpy.pi/3 * numpy.einsum('ip,iq->pq', ao, ao)

    h1ao[0,0] += fc
    h1ao[1,1] += fc
    h1ao[2,2] += fc
    return h1ao

def _get_sfx2c_quadrupole_integrals(mol, atm_id):
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        # Compute the integrals of quadrupole operator 
        # < sigma dot p | (3 \vec{r} \vec{r} - r^2) / r^5 | sigma dot p >
        ipipv = mol.intor('int1e_ipipprinvp', 9).reshape(3,3,nao,nao)
        ipvip = mol.intor('int1e_ipprinvpip', 9).reshape(3,3,nao,nao)
        h1ao = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1ao = h1ao + h1ao.transpose(0,1,3,2)

    coords = mol.atom_coord(atm_id).reshape(1, 3)
    ao = mol.eval_gto('GTOval_ip', coords, comp=3)
    fc = 4*numpy.pi/3 * numpy.einsum('dip,diq->pq', ao, ao)

    h1ao[0,0] += fc
    h1ao[1,1] += fc
    h1ao[2,2] += fc
    return h1ao

EFG = kernel

scf.hf.RHF.EFG = scf.rohf.ROHF.EFG = scf.uhf.UHF.EFG = lib.class_as_method(EFG)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = 'ccpvdz'
    mol.unit = 'B'
    mol.build()

    mf = scf.UHF(mol).x2c().run()
    mf.EFG()
