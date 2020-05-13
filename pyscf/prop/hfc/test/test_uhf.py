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

from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto, scf
from pyscf.scf import ucphf
from pyscf.scf import _response_functions
from pyscf.prop.hfc import uhf as uhf_hfc
from pyscf.prop.ssc import uhf as uhf_ssc
from pyscf.ao2mo import _ao2mo
from pyscf.prop.ssc.rhf import _dm1_mo2ao
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

# Test pso_soc function
# solve MO1 associated to a01 operator, then call contract to soc integral

# Note mo1 is the imaginary part of MO^1
def make_pso_soc(hfcobj, hfc_nuc=None):
    '''Spin-orbit coupling correction'''
    mol = hfcobj.mol
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)

    mf = hfcobj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    effspin = mol.spin * .5
    e_gyro = .5 * nist.G_ELECTRON
    nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**4 / 4 / effspin * e_gyro * au2MHz

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    orboa = mo_coeff[0][:, occidxa]
    orbva = mo_coeff[0][:,~occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbvb = mo_coeff[1][:,~occidxb]
    nocca = orboa.shape[1]
    nvira = orbva.shape[1]
    noccb = orbob.shape[1]
    nvirb = orbvb.shape[1]
    mo1a, mo1b = solve_mo1_pso(hfcobj, hfc_nuc)
    mo1a = mo1a.reshape(len(hfc_nuc),3,nvira,nocca)
    mo1b = mo1b.reshape(len(hfc_nuc),3,nvirb,noccb)
    dm0 = hfcobj._scf.make_rdm1()
    h1a, h1b = uhf_hfc.make_h1_soc(hfcobj, dm0)
    h1a = numpy.asarray([reduce(numpy.dot, (orbva.T, x, orboa)) for x in h1a])
    h1b = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orbob)) for x in h1b])
    para = []
    for n, atm_id in enumerate(hfc_nuc):
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atm_id)) * nuc_mag
        e = numpy.einsum('xij,yij->xy', mo1a[n], h1a) * 2
        e+= numpy.einsum('xij,yij->xy', mo1b[n], h1b) * 2
        para.append(fac * nuc_gyro * e)
    return numpy.asarray(para)

def solve_mo1_pso(hfcobj, hfc_nuc=None, with_cphf=None):
    if hfc_nuc   is None: hfc_nuc  = hfcobj.hfc_nuc
    if with_cphf is None: with_cphf = hfcobj.cphf

    mf = hfcobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mol = hfcobj.mol
    h1a, h1b = uhf_ssc.make_h1_pso(mol, mo_coeff, mo_occ, hfc_nuc)
    h1a = numpy.asarray(h1a)
    h1b = numpy.asarray(h1b)

    if with_cphf:
        vind = gen_vind(mf, mo_coeff, mo_occ)
        mo1, mo_e1 = ucphf.solve(vind, mo_energy, mo_occ, (h1a,h1b), None,
                                 hfcobj.max_cycle_cphf, hfcobj.conv_tol)
    else:
        eai_aa = lib.direct_sum('i-a->ai', mo_energy[0][mo_occ[0]>0], mo_energy[0][mo_occ[0]==0])
        eai_bb = lib.direct_sum('i-a->ai', mo_energy[1][mo_occ[1]>0], mo_energy[1][mo_occ[1]==0])
        mo1 = (h1a * (1/eai_aa), h1b * (1/eai_bb))
    return mo1

def gen_vind(mf, mo_coeff, mo_occ):
    '''Induced potential'''
    vresp = mf.gen_response(with_j=False, hermi=0)
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    orboa = mo_coeff[0][:, occidxa]
    orbva = mo_coeff[0][:,~occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbvb = mo_coeff[1][:,~occidxb]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]
    nova = nocca * nvira
    novb = noccb * nvirb
    mo_va_oa = numpy.asarray(numpy.hstack((orbva,orboa)), order='F')
    mo_vb_ob = numpy.asarray(numpy.hstack((orbvb,orbob)), order='F')
    def vind(mo1):
        mo1a = mo1.reshape(-1,nova+novb)[:,:nova].reshape(-1,nvira,nocca)
        mo1b = mo1.reshape(-1,nova+novb)[:,nova:].reshape(-1,nvirb,noccb)
        nset = mo1a.shape[0]
        dm1a = _dm1_mo2ao(mo1a, orbva, orboa)
        dm1b = _dm1_mo2ao(mo1b, orbvb, orbob)
        dm1 = numpy.vstack([dm1a-dm1a.transpose(0,2,1),
                            dm1b-dm1b.transpose(0,2,1)])
        v1 = vresp(dm1)
        v1a = _ao2mo.nr_e2(v1[    :nset], mo_va_oa, (0,nvira,nvira,nvira+nocca))
        v1b = _ao2mo.nr_e2(v1[nset:    ], mo_vb_ob, (0,nvirb,nvirb,nvirb+noccb))
        v1mo = numpy.hstack((v1a.reshape(nset,-1), v1b.reshape(nset,-1)))
        return v1mo.ravel()
    return vind


mol = gto.M(atom='H 0 0 0; H 0 0 1.',
            basis='ccpvdz', spin=1, charge=-1, verbose=3)
mf = scf.UHF(mol)
mf.kernel()
hfc = uhf_hfc.HFC(mf)
hfc.cphf = True
hfc.with_so_eff_charge = False
p1 = make_pso_soc(hfc)
print(p1) # -1.64305772e-03
