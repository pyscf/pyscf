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
# Based on fcidump.py

'''
FCIDUMP functions (write, read) for complex Hamiltonian for periodic systems

'#P' indicates that these lines exist for parallelism.

Based on fcidump.py, but with additional symmetry specification for k-point momentum
as used by HANDE.

Currently just RHF and needs modified kccsd_rhf to store all eris.

It also calculates modified exchange integrals using the truncated exchange
treatment which differ from the exchange using the bare Coulomb interaction.
These can be written in the same format as an FCIDUMP file, but without a header.

'''
from functools import reduce
import copy
import numpy
import pyscf.pbc
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.kccsd_rhf import KRCCSD
try:  # P
    from mpi4py import MPI  # P
    mpi4py_avail = True  # P
except ImportError:  # P
    mpi4py_avail = False  # P

# [todo] - allow read in from __config__, too.
DEFAULT_FLOAT_FORMAT = ' (%.16g,%.16g)'
TOL = 1e-10
# [todo] - allow orbsyms?
MOLPRO_ORBSYM = False


def write_head(fout, nmo, nelec, ms, nprop, propbitlen, orbsym=None):
    '''Write header of FCIDUMP file.

    Args:
        fout : file
            FCIDUMP file.
        nmo : int
            Number of (molecular) orbitals.
        nelec : int or List of ints [todo]
            Number of electrons.
        ms : int
            Overall spin.
        nprop : List of int
            Number of k points in each dimension.  [todo] - check
        propbitlen : int
            Number of bits in an element of orbsym corresponding to
            a dimension of k.  It means that an element in orbsym
            (which is an integer) can represent k (which is a list of
            integers).  If k is three dimensional, each third of the
            bit representation of an element of orbsym belongs to an
            an element of k.  The lengths of this third, in number of
            bits, is propbitlen.
            [todo] - improve? HANDE has an example (by Charlie)
    Kwargs:
        orbsym : List of ints, optional
            Integers labelling symmetry of the orbitals.  If not
            supplied, will assign label '1' to all orbitals, assigning
            them the same symmetry.
    '''
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * (nmo-1) + '1'))
    fout.write('  NPROP=%4d %4d %4d\n' % tuple(nprop))
    fout.write('  PROPBITLEN=%4d\n' % propbitlen)
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')


def write_eri(fout, eri, kconserv, tol=TOL,
              float_format=DEFAULT_FLOAT_FORMAT):
    '''Write electron repulsion integrals (ERIs) to FCIDUMP.

    Args:
        fout : file
            FCIDUMP file.
        eri : numpy array
            Contains ERIs divided by number of k points, with indices
            [ka,kc,kb,a,c,b,d].
        kconserv : function
            Pass in three k point indices and get fourth one where
            overall momentum is conserved.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    output_format = float_format + ' %4d %4d %4d %4d\n'
    nkpts = eri.oooo.shape[0]
    no = eri.oooo.shape[3]
    nv = eri.vvvv.shape[3]
    nor = no+nv
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp, kq, kr]
                # The documentation in pyscf/pbc/lib/kpts_helper.py in
                # get_kconserv is inconsistent with the actual code (and
                # physics). [k*(1) l(1) | m*(2) n(2)] = <km|ln> is the
                # integral. kconserve gives n given klm, such that
                # l-k=n-m (not k-l=n-m)
                er = eri.eri
                for i in range(er.shape[3]):
                    for j in range(er.shape[5]):
                        for k in range(er.shape[4]):
                            for l in range(er.shape[6]):
                                # Stored as [ka,kc,kb,a,c,b,d] <- (ab|cd)
                                v = er[kp, kr, kq, i, k, j, l]
                                if abs(v) > tol:
                                    fout.write(output_format % (
                                        v.real, v.imag, nor*kp+i+1, nor*kq+j+1,
                                        nor*kr+k+1, nor*ks+l+1))


def write_exchange_integrals(fout, xints, ki, nkpts, nor, tol=TOL,
                             float_format=DEFAULT_FLOAT_FORMAT):
    '''Write extra exchange electron repulsion integrals to FCIDUMP.

    Args:
        fout : file
            FCIDUMP file.
        xints : numpy array, dim: (MO at k point, all MO, all MO)
            extra exchange integrals <pi|iq>_x
        ki : int
            k index for orb i in <pi|iq>_x
        nkpts : int
            Number of k points
        nor : int
            Number of orbitals at a k point.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    output_format = float_format + ' %4d %4d %4d %4d\n'
    for kj in range(nkpts):
        for kk in range(nkpts):
            for i in range(nor):
                for j in range(nor):
                    for k in range(nor):
                        v = xints[i, kj*nor+j, kk*nor+k]/nkpts
                        if abs(v) > tol:
                            fout.write(output_format % (
                                v.real, v.imag, nor*ki+i+1, nor*kj+j+1,
                                nor*kk+k+1, nor*ki+i+1))


def write_hcore(fout, h, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Write the <i|h|j> integrals to FCIDUMP file.

    Note that <i|h|j> == 0 unless i and j are on the same k point.

    Args:
        fout : file
            FCIDUMP file
        h : array with dimensions (nkpoints, nmo, nmo) [todo] - check
            The <i|h|j> integrals for each kpoint.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    # [todo] - is something like hk.reshape(nmo, nmo) required?
    output_format = float_format + ' %4d %4d  0  0\n'
    nmos = 1
    for hk in h:
        nmo = hk.shape[0]
        for i in range(nmo):
            for j in range(0, i+1):
                if abs(hk[i, j]) > tol:
                    fout.write(output_format %
                               (hk[i, j].real, hk[i, j].imag, i+nmos, j+nmos))
        nmos += nmo


def from_integrals(output, h1e, h2e, nmo, nelec, kconserv, nuc, ms, nprop,
                   npropbitlen, orbsym=None, tol=TOL,
                   float_format=DEFAULT_FLOAT_FORMAT):
    '''Use passed in integrals to write integrals to FCIDUMP.

    Args:
        output : str
            Name of FCIDUMP file.
        h1e : array with dimensions (nkpoints, nmo, nmo) [todo] - check
            The <i|h|j> integrals for each k point.
        h2e : numpy array
            Contains ERIs divided by number of k points, with indices
            [ka,kc,kb,a,c,b,d].
        nmo : int
            Number of (molecular) orbitals (sum over all kpoints).
        nelec : int or List of ints [todo]
            Number of electrons.
        kconserv : function
            Pass in three k point indices and get fourth one where
            overall momentum is conserved.

        nuc : float
            Constant, nuclear energy.  Note that it is scaled by number
            of k points.
        ms : int
            Overall spin.
        nprop : List of int
            Number of k points in each dimension.  [todo] - check
        propbitlen : int
            Number of bits in an element of orbsym corresponding to
            a dimension of k.  It means that an element in orbsym
            (which is an integer) can represent k (which is a list of
            integers).  If k is three dimensional, each third of the
            bit representation of an element of orbsym belongs to an
            an element of k.  The lengths of this third, in number of
            bits, is propbitlen.
            [todo] - improve? HANDE has an example (by Charlie)
    Kwargs:
        orbsym : List of ints, optional
            Integers labelling symmetry of the orbitals.
            Default is None, in which case write_head function will
            assign all orbitals the same symmetry, 1.
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    with open(output, 'w') as fout:
        write_head(fout, nmo, nelec, ms, nprop, npropbitlen, orbsym=orbsym)
        write_eri(fout, h2e, kconserv, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % (nuc, 0))


def _partition(part, rank, nmo, size, nkpts, ntot):  # p
    rem = nmo % size  # p
    dspls = [0]*(size+1)  # p
    for s in range(size):  # p
        if (rem != 0) and ((size - s) <= rem):  # p
            dspls[s+1] = dspls[s]+part+1  # p
        else:  # p
            dspls[s+1] = dspls[s]+part  # p
    l = range(dspls[rank], dspls[rank+1])  # p
    nar = len(l)  # p
    counts = [0]*size  # p
    for s in range(size+1):  # p
        dspls[s] = dspls[s]*2*ntot*ntot  # p
    for s in range(size):  # p
        counts[s] = dspls[s+1]-dspls[s]  # p
    dspls.pop(-1)  # p
    return nar, l, dspls, counts  # p


def exchange_integrals(comm, mf, nmo, kconserv, fout, kstart, kpts):
    '''Calculate <pi|iq>_x exchange integrals.

    Args:
        comm : MPI.COMM_WORLD or None
            Needed for MPI parallelism.
        mf : SCF calculation object
            Stores SCF calculation results.
        nmo : int
            Number of molecular orbital on a k point.
        kconserv : function
            Pass in three k point indices and get fourth one where
            overall momentum is conserved.
        fout : file
            FCIDUMP file
        kstart : int
            Next k point to calculation exchange integral for.
            (Calculation can take time, so this means that this can be
            split)
        kpts : (nkpts, 3) ndarray of floats
            Were found using cell.get_abs_kpts(scaled_kpts).
            Absolute k-points in 1/Bohr.
            [todo]
    '''
    if comm == None:  # P
        rank = 0
        size = 1
    else:  # P
        rank = comm.Get_rank()  # P
        size = comm.Get_size()  # P
    nkpts = len(kpts)
    ntot = nkpts*nmo
    if nmo < size:
        print("More processes than molecular orbitals")
    part = nmo // size
    nar, l, dspls, counts = _partition(part, rank, nmo, size, nkpts, ntot)
    xints_p = numpy.ndarray((nar, ntot, ntot), dtype=complex)
    xints_p.fill(0)
    # We wish to calculate <ip|qi> for all i,p,q.  We do this by iterating over
    # all possible orbitals i, and for each, create a density matrix.
    # From this density we create an exchange potential (v_k(G)), which is evaluated at
    # all G vectors.  To form <ip|qi> we calculate the codensity
    # of p and q (again in G space), and integrate (sum) this with the exchange
    # potential of i. i, p and q are different at each k-point, and so we need
    # to loop over k-points for them too, with the proviso of momentum
    # conservation, which is determined by the kconverv table.
    for ik in range(kstart, nkpts):
        for i in l:
            occ = numpy.zeros(nmo)
            occ[i] = 1
            occk = [numpy.zeros(nmo)]*nkpts
            occk[ik] = occ         # just occupy orbital i in kpoint ik
            # According to docs, dm_kpts is a (nkpts, nao, nao) ndarray.
            # dm_kpts is zero at all k points, except for k = ik case
            # where its (nao, nao) sized element contains C^i_m C^{*i}_n
            # entry at position mn.  C^i_m is the coefficient MO i has
            # at position m in AO space. (i, p, q are MO labels, m, n are
            # AO labels here).
            dm_kpts = mf.make_rdm1(mf.mo_coeff, occk)
            # 0 => might not be hermitian
            vk = mf.get_k(cell=None, dm_kpts=dm_kpts, hermi=0, kpts=kpts,
                          kpts_band=None, omega=None)
            # NB that vk is the exchange potential of electron ik,i, expressed
            # in the Bloch-AO basis and in general has components for all k
            # values.
            # [todo] - check:
            # vk is related to K in equation 11 of
            # McClain et al. JCTC, 2017, 13, 3, 1209–1218 but includes
            # contracted molecular coefficients of ik, i.
            for p in range(nmo):
                for pk in range(nkpts):
                    for q in range(nmo):
                        # The documentation in pyscf/pbc/lib/kpts_helper.py in
                        # get_kconserv is inconsistent with the actual code
                        # (and physics). [k*(1) l(1) | m*(2) n(2)] = <km|ln> is
                        # the integral. kconserve gives n given klm, such that
                        # l-k=n-m (not k-l=n-m).
                        # Since we want (iq|pi) = <ip|qi> = <pi|iq>, we use the
                        # following lookup
                        # Might need to check the ordering of this!
                        # It's definitely elec1,elec1,elec2 but whether the
                        # result is bra or ket is uncertain
                        qk = kconserv[pk, ik, ik]
                        if (qk != pk):
                            print("Error in exchange integrals: k point ",
                                  qk, "is matched to ", pk)
                        # [todo] - check order and einsum order here?
                        # dm_{mn} element is equal to C^{*p}_m C^q_n.
                        dm = numpy.outer(
                            mf.mo_coeff[pk].T[p].T.conj(),
                            mf.mo_coeff[qk].T[q])
                        intgrl = numpy.einsum('ij,ij', dm, vk[pk])
                        # nkpts needed for renormalization so integrals are for
                        # whole supercell not just unit cell
                        xints_p[i-l[0], qk*nmo+q, pk*nmo+p] = intgrl*nkpts
        if comm == None:  # P
            xints = xints_p
        else:  # P
            xints = numpy.ndarray(nmo*ntot*ntot, dtype=complex)  # P
            xints.fill(0)  # P
            comm.Gatherv([xints_p.flatten(), counts[rank], MPI.COMPLEX],  # P
                         [xints, tuple(counts), tuple(
                             dspls), MPI.COMPLEX],  # P
                         root=0)  # P
        if rank == 0:
            xints = xints.reshape((nmo, ntot, ntot))
            write_exchange_integrals(fout, xints, ik, nkpts, nmo)
            fout.flush()


def fcidump(fcid, mf, kgrid, scaled_kpts_in, MP, keep_exxdiv=False, resume=False,
            parallel=None):
    '''Dump constant term, orb energies, 1-e and 2-e integrals to file.

    Args:
        fcid : str
            Name of the file to write to. Exchange integrals will be
            written to fcid_X, the rest to fcid.
        mf : SCF calculation object
            Stores SCF calculation results.
        scaled_kpts_in : Numpy array of floats
            Scaled k-points, i.e. (.5,.5,.5) is at the very corner of
            the BZ.
        MP : bool
            True if the grid used is a Monkhorst-Pack grid.
    Kwargs:
        keep_exxdiv : bool
            If True, keep exxdiv treatment used for SCF calculation
            when evaluating two electron integrals.
            [todo] - check! Should this be an option?
            The default is False.
        resume : bool
            If True, the program will check for existing fcid_X file
            and resume the dumping of X integrals.
            The default is False.
        parallel : bool
            If True, MPI parallelization will be used.
            The default is False.
    '''

    if parallel:  # P
        if not mpi4py_avail:  # P
            raise ImportError("Chosen parallel but mpi4py not available!")  # P
        comm = MPI.COMM_WORLD  # P
    else:  # P
        comm = None  # P
    # scaled_kpts will be modified later so make copy.
    # [todo] - more deepcopying needed?
    scaled_kpts = copy.deepcopy(scaled_kpts_in)
    dummy_cc = KRCCSD(mf)
    # If keep_exxdiv, keeping exxdiv used in scf calculation.
    dummy_cc.keep_exxdiv = keep_exxdiv
    nprop = list(kgrid)

    if comm == None:  # P
        rank = 0
    else:  # P
        rank = comm.Get_rank()  # P
    kconserv = dummy_cc.khelper.kconserv
    nmo = len(mf.mo_coeff[0])
    fx = None
    kstart = None
    if rank == 0:
        if resume:
            fx = open(fcid+"_X", 'r')
            lines = fx.readlines()
            line = lines[-1].split()
            kstart = (int(line[1])-1)/nmo + 1
            print("Resuming dumping, starting at k point " + str(kstart))
            fx.close()
            fx = open(fcid+"_X", 'a')
        else:
            fx = open(fcid+"_X", 'w')
            kstart = 0
    if comm != None:
        kstart = comm.bcast(kstart, root=0)
    exchange_integrals(comm, mf, nmo, kconserv, fx, kstart, mf.kpts)
    if rank == 0:
        fx.close()
        # MP meshes with an even number of points in a dimension do not contain
        # the Gamma point.
        # Unfortunately this is not compatible with some symmetry
        # specifications, so if we multiply that dimension's kpoint grid by 2,
        # we get a grid which can contain both the MP mesh and the Gamma point
        # (even though we don't have any actual orbitals calculated at the
        # gamma point).
        if MP:
            for i in range(3):
                if nprop[i] % 2 == 0:
                    nprop[i] *= 2
        npropbitlen = 8
        for i, nk in enumerate(nprop):
            for j in range(scaled_kpts.shape[0]):
                scaled_kpts[j, i] = int(round(scaled_kpts[j, i]*nk)) % nk
        kps = len(mf.mo_coeff)
        # For each k-point, get the G-space core hamiltonian, and transform it
        # into the molecular orbital basis.
        # Different k-points don't couple.
        # h1es will contain a list with an MOxMO matrix for each k-point.
        h1es = [reduce(numpy.dot,
                       (numpy.asarray(mf.mo_coeff)[k].T.conj(),
                        mf.get_hcore()[k], numpy.asarray(mf.mo_coeff)[k]))
                for k in range(kps)]
        eris = dummy_cc.ao2mo()
        nel = sum(sum(mf.mo_occ))
        orbsym = []
        propsc = 2**npropbitlen
        for k in range(kps):
            n = scaled_kpts[k, 0]+propsc*scaled_kpts[k, 1] + \
                propsc*propsc*scaled_kpts[k, 2]
            orbsym += [int(n)]*nmo
        nkpts = kgrid[0]*kgrid[1]*kgrid[2]
        from_integrals(fcid, h1es, eris, kps*nmo, nel, kconserv,
                       nkpts*mf.mol.energy_nuc(), 0, nprop, npropbitlen,
                       orbsym=orbsym)
        # Write orbital energies to fcid file, too.
        f = open(fcid, 'a')
        n = 0
        for k in range(kps):
            for e in mf.mo_energy[k]:
                n += 1
                f.write(' (%.16g,%.16g) %4d %4d %4d %4d\n' %
                        (e.real, e.imag, n, 0, 0, 0))
        f.close()
