#!/usr/bin/env python
#
# Author: Zhendong Li <zhendongli2008@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import tempfile
import shutil
import subprocess
import numpy
from pyscf import lib
from pyscf import gto, scf, mcscf
from pyscf import df

'''
COSMO interface

Note the default COSMO integration grids may break the system symmetry.
'''

try:
    from pyscf.cosmo import settings
except ImportError:
    msg = '''settings.py not found.  Please create %s
''' % os.path.join(os.path.dirname(__file__), 'settings.py')
    sys.stderr.write(msg)
    raise ImportError

# Define a cosmo class to carry all the parameters
class COSMO(object):
    '''COSMO interface handler

    Attributes:
        verbose : int
            Print level
        output : str or None
            Output file, default is None which dumps msg to sys.stdout
        base
    '''
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.tmpdir = tempfile.mkdtemp(prefix='cosmotmp', dir=settings.COSMOSCRATCHDIR)
        #shutil.rmtree(self.tmpdir) # Note the circular dependence in cosmo_fock function
        self.base = 'iface'
        self.suffix = 'out'
        #toang = 0.529177249 # 0.52917721092d0
        self.eps     = -1.0  # Dielectric constant?
        self.rsolv   = 1.30  # Bohr
        self.routf   = 0.850
        self.disex   = 10.0
        self.nppa    = 1082
        self.nspa    = 92
        self.nradii  = 0

        self._dm_guess = None  # system density
        self.dm = None   # given system density, avoid potential being updated SCFly
        #self.x2c_correction = False
        self.casci_conv_tol = 1e-7
        self.casci_state_id = None
        self.casci_max_cycle = 50

##################################################
# don't modify the following private variables, they are not input options
        self.parfile = self.base+'.cosmo_par.'+self.suffix
        self.datfile = self.base+'.cosmo_dat.'+self.suffix
        self.segfile = self.base+'.cosmo_seg.'+self.suffix
        self.a1mfile = self.base+'.cosmo_a1m.'+self.suffix
        # segs
        self.nps = None
        self.npspher = None
        self.de = None
        self.qsum = None
        self.dq = None
        self.ediel = 0.0
        self.edielrs = None
        self.asso_cosurf = 0
        self.asso_qcos = 0
        self.asso_qcosrs = 0
        self.asso_ar = 0
        self.asso_phi = 0
        self.asso_phio = 0
        self.asso_phic = 0
        self.asso_qcosc = 0
        self.asso_iatsp = 0
        self.asso_gcos = 0
        self.qcos = None
        self.cosurf = None
        self.phi = None
        self.phio = None
        self._built = False

    def fepsi(self):
        if self.eps<0.0:
            fe = 1.0
        else:
            fe = (self.eps-1.0)/(self.eps+0.5)
        return fe

    def check(self):
        print('asso_cosurf= %s' % self.asso_cosurf)
        print('asso_qcos  = %s' % self.asso_qcos  )
        print('asso_qcosrs= %s' % self.asso_qcosrs)
        print('asso_ar    = %s' % self.asso_ar    )
        print('asso_phi   = %s' % self.asso_phi   )
        print('asso_phio  = %s' % self.asso_phio  )
        print('asso_phic  = %s' % self.asso_phic  )
        print('asso_qcosc = %s' % self.asso_qcosc )
        print('asso_iatsp = %s' % self.asso_iatsp )
        print('asso_gcos  = %s' % self.asso_gcos  )

    def build(self, mol=None):
        self.initialization(mol)

    def initialization(self, mol=None):
        if mol is None: mol = self.mol
        #
        # task-0
        #
        f1 = open(os.path.join(self.tmpdir, self.parfile),'w')
        f1.write('Task = 0\n')
        f1.write('eps = '+str(self.eps)+'\n')
        f1.write('rsolv = '+str(self.rsolv)+'\n')
        f1.write('routf = '+str(self.routf)+'\n')
        f1.write('disex = '+str(self.disex)+'\n')
        f1.write('nppa = '+str(self.nppa)+'\n' )
        f1.write('nspa = '+str(self.nspa)+'\n' )
        f1.write('natoms = '+str(mol.natm)+'\n')
        for i in range(mol.natm):
            f1.write(str(mol.atom_charge(i))+'\n')
        f1.close()
        exe_cosmo(self)
        #
        # task-1
        #
        f1 = open(os.path.join(self.tmpdir, self.parfile),'w')
        f1.write('Task = 1\n')
        f1.write('NRadii = '+str(self.nradii)+'\n')
        #for i in range(self.nradii):
        #    f1.write(str(mol.atom_charge(i))+'\n')
        f1.close()
        exe_cosmo(self)
        #
        # task-2
        #
        f1 = open(os.path.join(self.tmpdir, self.parfile),'w')
        f1.write('Task = 2\n')
        f1.write('Natoms = '+str(mol.natm)+'\n')
        for i in range(mol.natm):
            coord = mol.atom_coord(i)
            f1.write('%20.15f %20.15f %20.15f\n' % tuple(coord.tolist()))
        f1.close()
        exe_cosmo(self)
        lib.logger.info(self, '\nCOSMO tmpdir  %s',
                        os.path.abspath(self.tmpdir))
        self._built = True
        return 0

    def loadsegs(self):
        #
        # task-0
        #
        #with open(os.path.join(self.tmpdir, self.segfile), 'r') as f1:
        #    print ''.join(f1.readlines()[:10])
        f1 = open(os.path.join(self.tmpdir, self.segfile), 'r')
        line = f1.readline()
        self.nps = int(line)
        line = f1.readline()
        self.npspher = int(line)
        line = f1.readline()
        self.de = float(line)
        line = f1.readline()
        self.qsum = float(line)
        line = f1.readline()
        self.dq = float(line)
        line = f1.readline()
        self.ediel = float(line)
        line = f1.readline()
        self.edielrs = float(line)
        # Control parameters
        line = f1.readline()
        self.asso_cosurf = int(line)
        line = f1.readline()
        self.asso_qcos = int(line)
        line = f1.readline()
        self.asso_qcosrs = int(line)
        line = f1.readline()
        self.asso_ar = int(line)
        line = f1.readline()
        self.asso_phi = int(line)
        line = f1.readline()
        self.asso_phio = int(line)
        line = f1.readline()
        self.asso_phic = int(line)
        line = f1.readline()
        self.asso_qcosc = int(line)
        line = f1.readline()
        self.asso_iatsp = int(line)
        line = f1.readline()
        self.asso_gcos = int(line)
        # Float
        self.qcos   = numpy.zeros(self.nps)
        self.cosurf = numpy.zeros(3*(self.nps+self.npspher))
        self.phi    = numpy.zeros(self.nps)
        self.phio   = numpy.zeros(self.npspher)
        for i in range(3*(self.nps+self.npspher)):
            line = f1.readline()
            self.cosurf[i] = float(line)
        for i in range(self.nps):
            line = f1.readline()
            self.qcos[i] = float(line)
        if self.asso_ar > 0:
            for i in range(self.nps):
                line = f1.readline()
        if self.asso_phi> 0:
            for i in range(self.nps):
                line = f1.readline()
                self.phi[i] = float(line)
        if self.asso_phio> 0:
            for i in range(self.npspher):
                line = f1.readline()
                self.phio[i] = float(line)
        f1.close()

    def savesegs(self):
        #
        # task-0
        #
        f1 = open(os.path.join(self.tmpdir, self.segfile),'r')
        text = []
        for i in range(17):
            text.append(f1.readline())
        for i in range(3*(self.nps+self.npspher)):
            f1.readline()
            text.append('%24.15f'%(self.cosurf[i])+'\n')
        for i in range(self.nps):
            f1.readline()
            text.append('%24.15f'%(self.qcos[i])+'\n')
        if self.asso_ar > 0:
            for i in range(self.nps):
                text.append(f1.readline())
        if self.asso_phi> 0:
            for i in range(self.nps):
                f1.readline()
                text.append('%24.15f'%(self.phi[i])+'\n')
        if self.asso_phio> 0:
            for i in range(self.npspher):
                f1.readline()
                text.append('%24.15f'%(self.phio[i])+'\n')
        for line in f1.readlines():
            text.append(line)
        f1.close()
        with open(os.path.join(self.tmpdir, self.segfile),'w') as f1:
            f1.writelines(text)
        return 0

    def charges(self):
        f1 = open(os.path.join(self.tmpdir, self.parfile),'w')
        f1.write('Task=3\n')
        f1.close()
        exe_cosmo(self)
        return 0

    def occ0(self):
        f1 = open(os.path.join(self.tmpdir, self.parfile),'w')
        f1.write('Task = 40\n')
        f1.close()
        lib.logger.info(self, '')
        lib.logger.info(self, 'Start outlying charge correction:')
        exe_cosmo(self)
        return 0

    def occ1(self):
        mol = self.mol
        f1 = open(os.path.join(self.tmpdir, self.parfile),'w')
        f1.write('Task = 4\n')
        f1.write('Escf = %24.15f\n'%self.e_tot)
        f1.write('Natoms = '+str(mol.natm)+'\n')
        for i in range(mol.natm):
            coord = mol.atom_coord(i)
            f1.write('%20.15f %20.15f %20.15f\n' % tuple(coord.tolist()))
        f1.close()
        lib.logger.info(self, 'COSMO outlying charge correction output %s/iface.cosmo.oc\n',
                        os.path.abspath(self.tmpdir))
        lib.logger.info(self, 'Final outlying charge correction:')

        # backup COSMO temporary file, avoid Task 40 overide initial settings
        #self.tmpdir, tmpdir_bak = tempfile.mkdtemp(prefix='cosmotmp', dir=settings.COSMOSCRATCHDIR), self.tmpdir
        self.tmpdir, tmpdir_bak = tempfile.mktemp(prefix='cosmotmp', dir=settings.COSMOSCRATCHDIR), self.tmpdir
        shutil.copytree(tmpdir_bak, self.tmpdir)
        exe_cosmo(self)
        with open(os.path.join(self.tmpdir, 'iface.cosmo'), 'r') as fin:
            dat = fin.readline()
            while dat:
                if dat.startswith('$cosmo_energy'):
                    break
                dat = fin.readline()
            dat = []
            for i in range(5):
                dat.append(fin.readline())

        dat.pop(2)
        lib.logger.info(self, ''.join(dat))
        e_tot = float(dat[1].split('=')[1])
        lib.logger.note(self, 'Total energy with COSMO corection %.15g', e_tot)
        shutil.copy(os.path.join(self.tmpdir, 'iface.cosmo'),
                    os.path.join(tmpdir_bak, 'iface.cosmo.oc'))
        shutil.rmtree(self.tmpdir)
        self.tmpdir = tmpdir_bak
        return e_tot

    def cosmo_fock(self, dm):
        return cosmo_fock(self, dm)

    def cosmo_occ(self, dm):
        return cosmo_occ(self, dm)

def exe_cosmo(mycosmo):
    mycosmo.stdout.flush()
    cmd = ' '.join((settings.COSMOEXE, mycosmo.base, mycosmo.suffix))
    try:
        output = subprocess.check_output(cmd, cwd=mycosmo.tmpdir, shell=True)
        if output:
            lib.logger.debug(mycosmo, 'COSMO output\n%s', output)
    except AttributeError:  # python 2.6 and older
        p = subprocess.check_call(cmd, cwd=mycosmo.tmpdir, shell=True)

def cosmo_fock_o0(cosmo, dm):
    mol = cosmo.mol
    debug = False
    # phi
    cosmo.loadsegs()
    for i in range(cosmo.nps):
        phi = 0.0
        coords = numpy.array((cosmo.cosurf[3*i],cosmo.cosurf[3*i+1],cosmo.cosurf[3*i+2]))
        for iatom in range(mol.natm):
            dab = numpy.linalg.norm(mol.atom_coord(iatom)-coords)
            phi += mol.atom_charge(iatom)/dab
        # Potential
        mol.set_rinv_origin_(coords)
        vpot = mol.intor('cint1e_rinv_sph')
        phi -= numpy.einsum('ij,ij',dm,vpot)
        cosmo.phi[i] = phi
        if debug:
            print(i,cosmo.qcos[i],cosmo.cosurf[i],cosmo.phi[i])
    cosmo.savesegs()
    # qk
    cosmo.charges()
    # vpot
    cosmo.loadsegs()
    fock = numpy.zeros(vpot.shape)
    for i in range(cosmo.nps):
        # Potential
        coords = numpy.array((cosmo.cosurf[3*i],cosmo.cosurf[3*i+1],cosmo.cosurf[3*i+2]))
        mol.set_rinv_origin_(coords)
        vpot = mol.intor('cint1e_rinv_sph')
        fock -= vpot*cosmo.qcos[i]
    fepsi = cosmo.fepsi() 
    fock = fepsi*fock
    return fock
def cosmo_fock_o1(cosmo, dm):
    mol = cosmo.mol
    nao = dm.shape[0]
    # phi
    cosmo.loadsegs()
    coords = cosmo.cosurf[:cosmo.nps*3].reshape(-1,3)
    fakemol = _make_fakemol(coords)
    j3c = df.incore.aux_e2(mol, fakemol, intor='cint3c2e_sph', aosym='s2ij')
    tril_dm = lib.pack_tril(dm) * 2
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm[diagidx] *= .5
    cosmo.phi = -numpy.einsum('x,xk->k', tril_dm, j3c)
    for ia in range(mol.natm):
        cosmo.phi += mol.atom_charge(ia)/lib.norm(mol.atom_coord(ia)-coords, axis=1)
    cosmo.savesegs()
    # qk
    cosmo.charges()
    # vpot
    cosmo.loadsegs()
#X    fakemol = _make_fakemol(cosmo.cosurf[:cosmo.nps*3].reshape(-1,3))
#X    j3c = df.incore.aux_e2(mol, fakemol, intor='cint3c2e_sph', aosym='s2ij')
    fock = lib.unpack_tril(numpy.einsum('xk,k->x', j3c, -cosmo.qcos[:cosmo.nps]))
    fepsi = cosmo.fepsi() 
    fock = fepsi*fock
    return fock
def cosmo_fock(cosmo, dm):
    if not isinstance(dm, numpy.ndarray) or dm.ndim != 2:
        dm = dm[0]+dm[1]
    return cosmo_fock_o1(cosmo, dm)

def cosmo_occ_o0(cosmo, dm):
    mol = cosmo.mol
    #cosmo.check()
    cosmo.occ0()
    cosmo.loadsegs()
    #cosmo.check()
    ioff = 3*cosmo.nps
    for i in range(cosmo.npspher):
        phi = 0.0
        coords = numpy.array((cosmo.cosurf[ioff+3*i],cosmo.cosurf[ioff+3*i+1],cosmo.cosurf[ioff+3*i+2]))
        for iatom in range(mol.natm):
            dab = numpy.linalg.norm(mol.atom_coord(iatom)-coords)
            phi += mol.atom_charge(iatom)/dab
        # Potential
        mol.set_rinv_origin_(coords)
        vpot = mol.intor('cint1e_rinv_sph')
        phi -= numpy.einsum('ij,ij',dm,vpot)
        cosmo.phio[i] = phi
    cosmo.savesegs()
    return cosmo.occ1()
def cosmo_occ_o1(cosmo, dm):
    mol = cosmo.mol
    nao = dm.shape[0]
    #cosmo.check()
    cosmo.occ0()
    cosmo.loadsegs()
    #cosmo.check()
    ioff = 3*cosmo.nps
    coords = cosmo.cosurf[ioff:ioff+cosmo.npspher*3].reshape(-1,3)
    fakemol = _make_fakemol(coords)
    j3c = df.incore.aux_e2(mol, fakemol, intor='cint3c2e_sph', aosym='s2ij')
    tril_dm = lib.pack_tril(dm) * 2
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm[diagidx] *= .5
    cosmo.phio = -numpy.einsum('x,xk->k', tril_dm, j3c)
    for ia in range(mol.natm):
        cosmo.phio += mol.atom_charge(ia)/lib.norm(mol.atom_coord(ia)-coords, axis=1)
    cosmo.savesegs()
    return cosmo.occ1()
def cosmo_occ(cosmo, dm):
    if not isinstance(dm, numpy.ndarray) or dm.ndim != 2:
        dm = dm[0]+dm[1]
    return cosmo_occ_o1(cosmo, dm)

def _make_fakemol(coords):
    nbas = coords.shape[0]
    fakeatm = numpy.zeros((nbas,gto.ATM_SLOTS), dtype=numpy.int32)
    fakebas = numpy.zeros((nbas,gto.BAS_SLOTS), dtype=numpy.int32)
    fakeenv = [0] * gto.PTR_ENV_START
    ptr = gto.PTR_ENV_START
    fakeatm[:,gto.PTR_COORD] = numpy.arange(ptr, ptr+nbas*3, 3)
    fakeenv.append(coords.ravel())
    ptr += nbas*3
    fakebas[:,gto.ATOM_OF] = numpy.arange(nbas)
    fakebas[:,gto.NPRIM_OF] = 1
    fakebas[:,gto.NCTR_OF] = 1
# approximate point charge with gaussian distribution exp(-1e9*r^2)
    fakebas[:,gto.PTR_EXP] = ptr
    fakebas[:,gto.PTR_COEFF] = ptr+1
    expnt = 1e9
    fakeenv.append([expnt, 1/(2*numpy.sqrt(numpy.pi)*gto.mole._gaussian_int(2,expnt))])
    ptr += 2
    fakemol = gto.Mole()
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = numpy.hstack(fakeenv)
    fakemol.natm = nbas
    fakemol.nbas = nbas
    fakemol._built = True
    return fakemol


#
# NOTE: cosmo_for_scf and cosmo_for_mcscf/cosmo_for_casci modified different
# functions, so that we can pass a "COSMOlized"-MF object to the CASCI/CASSCF class.
#


# NOTE: be careful with vhf/vhf_last (in scf kernel) when direct_scf is applied
def cosmo_for_scf(mf, cosmo):
    oldMF = mf.__class__
    cosmo.initialization(cosmo.mol)
    if cosmo.dm is not None:
        # static solvation environment.  The potential and dielectric energy
        # are not updated SCFly
        cosmo._v = cosmo.cosmo_fock(cosmo.dm)

    class MF(oldMF):
        def __init__(self):
            self.__dict__.update(mf.__dict__)

        def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
                     diis_start_cycle=None, level_shift_factor=None,
                     damp_factor=None):
            if cosmo.dm is None:
                cosmo._v = cosmo.cosmo_fock(dm)
            return self.get_fock_(h1e+cosmo._v, s1e, vhf, dm, cycle, adiis,
                                  diis_start_cycle, level_shift_factor, damp_factor)

        def get_grad(self, mo_coeff, mo_occ, h1_vhf=None):
            if h1_vhf is None:
                dm1 = self.make_rdm1(mo_coeff, mo_occ)
                h1_vhf = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
            fock = h1_vhf + cosmo._v
            return oldMF.get_grad(self, mo_coeff, mo_occ, fock)

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            e_tot, e_coul = oldMF.energy_elec(self, dm, h1e, vhf)
            e_tot += cosmo.ediel
            lib.logger.debug(self, 'E_diel = %.15g', cosmo.ediel)
            return e_tot, e_coul

        def _finalize_(self):
            cosmo.e_tot = self.e_tot
            self.e_tot = cosmo.cosmo_occ(self.make_rdm1())
            return oldMF._finalize_(self)
    return MF()

def cosmo_for_mcscf(mc, cosmo):
    oldCAS = mc.__class__
    cosmo.initialization(cosmo.mol)
    if cosmo.dm is not None:  # A frozen COSMO potential
        vcosmo = cosmo.cosmo_fock(cosmo.dm)
        cosmo._dm_guess = cosmo.dm

    class CAS(oldCAS):
        def __init__(self):
            self.__dict__.update(mc.__dict__)

        def dump_flags(self):
            oldCAS.dump_flags(self)
            if hasattr(self, 'conv_tol') and self.conv_tol < 1e-6:
                lib.logger.warn(self, 'CASSCF+COSMO might not be able to'
                                'converge to conv_tol %g', self.conv_tol)

        def update_casdm(self, mo, u, fcivec, e_ci, eris):
            casdm1, casdm2, gci, fcivec = \
                    oldCAS.update_casdm(self, mo, u, fcivec, e_ci, eris)
            mocore = mo[:,:self.ncore]
            mocas = mo[:,self.ncore:self.ncore+self.ncas]
# We save the density of micro iteration in cosmo._dm_guess.  It's not the
# same to the CASSCF density of macro iteration, which we used to measure
# convergence.  But when CASSCF converged, cosmo._dm_guess should be almost
# the same to the density of macro iteration .
            cosmo._dm_guess = reduce(numpy.dot, (mocas, casdm1, mocas.T))
            cosmo._dm_guess += numpy.dot(mocore, mocore.T) * 2
            return casdm1, casdm2, gci, fcivec

# We modify hcore to feed the potential of outlying charge into the orbital
# gradients (see CASSCF function gen_h_op).  Note hcore is also used to
# compute the Ecore.  The energy contribution from outlying charge needs to be
# removed.
# Note the approximation _edup = Tr(Vcosmo, DM_CASCI) = Tr(Vcosmo, DM_input)
# When CASSCF converged, we assumed the input DM (computed in update_casdm) is
# identical to the CASCI DM (from the macro iteration).
        def get_hcore(self, mol=None):
            if cosmo.dm is not None:
                v1 = vcosmo
            else:
                if cosmo._dm_guess is None:  # Initial guess
                    na = self.ncore + self.nelecas[0]
                    nb = self.ncore + self.nelecas[1]
                    dm =(numpy.dot(self.mo_coeff[:,:na], self.mo_coeff[:,:na].T)
                       + numpy.dot(self.mo_coeff[:,:nb], self.mo_coeff[:,:nb].T))
                else:
                    dm = cosmo._dm_guess
                v1 = cosmo.cosmo_fock(dm)
            cosmo._v = v1
            return self._scf.get_hcore(mol) + v1

        def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
            e_tot, e_cas, fcivec = oldCAS.casci(self, mo_coeff, ci0, eris,
                                                verbose, envs)

            if self.fcisolver.nroots > 1 and cosmo.casci_state_id is not None:
                c = fcivec[cosmo.casci_state_id]
                casdm1 = self.fcisolver.make_rdm1(c, self.ncas, self.nelecas)
            else:
                casdm1 = self.fcisolver.make_rdm1(fcivec, self.ncas, self.nelecas)
            mocore = mo_coeff[:,:self.ncore]
            mocas = mo_coeff[:,self.ncore:self.ncore+self.ncas]
            cosmo._dm_guess = reduce(numpy.dot, (mocas, casdm1, mocas.T))
            cosmo._dm_guess += numpy.dot(mocore, mocore.T) * 2
            edup = numpy.einsum('ij,ij', cosmo._v, cosmo._dm_guess)
            # Substract <VP> to get E0, then add Ediel
            e_tot = e_tot - edup + cosmo.ediel
            return e_tot, e_cas, fcivec

        def _finalize_(self):
            cosmo.e_tot = self.e_tot
            self.e_tot = cosmo.cosmo_occ(self.make_rdm1())
            return oldCAS._finalize_(self)

    return CAS()

def cosmo_for_casci(mc, cosmo):
    oldCAS = mc.__class__
    cosmo.initialization(cosmo.mol)
    if cosmo.dm is not None:
        cosmo._dm_guess = cosmo.dm
        vcosmo = cosmo.cosmo_fock(cosmo.dm)

    class CAS(oldCAS):
        def __init__(self):
            self.__dict__.update(mc.__dict__)

        def get_hcore(self, mol=None):
            if cosmo.dm is not None:
                v1 = vcosmo
            else:
                if cosmo._dm_guess is None:  # Initial guess
                    na = self.ncore + self.nelecas[0]
                    nb = self.ncore + self.nelecas[1]
                    #log.Initial('Initial DM: na,nb,nelec=',na,nb,na+nb)
                    dm =(numpy.dot(self.mo_coeff[:,:na], self.mo_coeff[:,:na].T)
                       + numpy.dot(self.mo_coeff[:,:nb], self.mo_coeff[:,:nb].T))
                else:
                    dm = cosmo._dm_guess
                v1 = cosmo.cosmo_fock(dm)
            cosmo._v = v1
            return self._scf.get_hcore(mol) + v1

        def kernel(self, mo_coeff=None, ci0=None):
            if mo_coeff is None:
                mo_coeff = self.mo_coeff
            else:
                self.mo_coeff = mo_coeff
            if ci0 is None:
                ci0 = self.ci
            if self.mol.symmetry:
                mcscf.casci_symm.label_symmetry_(self, self.mo_coeff)
            log = lib.logger.Logger(self.stdout, self.verbose)

            def casci_iter(mo_coeff, ci, cycle):
                # casci.kernel call get_hcore, which initialized cosmo._v
                e_tot, e_cas, fcivec = mcscf.casci.kernel(self, mo_coeff,
                                                          ci0=ci0, verbose=log)
                if self.fcisolver.nroots > 1 and cosmo.casci_state_id is not None:
                    c = fcivec[cosmo.casci_state_id]
                    casdm1 = self.fcisolver.make_rdm1(c, self.ncas, self.nelecas)
                else:
                    casdm1 = self.fcisolver.make_rdm1(fcivec, self.ncas, self.nelecas)
                mocore = mo_coeff[:,:self.ncore]
                mocas = mo_coeff[:,self.ncore:self.ncore+self.ncas]
                cosmo._dm_guess = reduce(numpy.dot, (mocas, casdm1, mocas.T))
                cosmo._dm_guess += numpy.dot(mocore, mocore.T) * 2
                edup = numpy.einsum('ij,ij', cosmo._v, cosmo._dm_guess)
                # Substract <VP> to get E0, then add Ediel
                e_tot = e_tot - edup + cosmo.ediel

                log.debug('COSMO E_diel = %.15g', cosmo.ediel)

                if (log.verbose >= lib.logger.INFO and
                    hasattr(self.fcisolver, 'spin_square')):
                    ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
                    if isinstance(e_cas, (float, numpy.number)):
                        log.info('COSMO_cycle %d CASCI E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                                 cycle, e_tot, e_cas, ss[0])
                    else:
                        for i, e in enumerate(e_cas):
                            log.info('COSMO_cycle %d CASCI root %d  E = %.15g  '
                                     'E(CI) = %.15g  S^2 = %.7f',
                                     cycle, i, e_tot[i], e, ss[0][i])
                else:
                    if isinstance(e_cas, (float, numpy.number)):
                        log.note('COSMO_cycle %d CASCI E = %.15g  E(CI) = %.15g',
                                 cycle, e_tot, e_cas)
                    else:
                        for i, e in enumerate(e_cas):
                            log.note('COSMO_cycle %d CASCI root %d  E = %.15g  E(CI) = %.15g',
                                     cycle, i, e_tot[i], e)
                return e_tot, e_cas, fcivec

            if cosmo.dm is not None:
                self.e_tot, self.e_cas, self.ci = casci_iter(mo_coeff, ci0, 0)
            else:
                e_tot = 0
                for icycle in range(cosmo.casci_max_cycle):
                    self.e_tot, self.e_cas, self.ci = casci_iter(mo_coeff, ci0,
                                                                 icycle)
                    if abs(self.e_tot - e_tot) < cosmo.casci_conv_tol:
                        log.debug('    delta E(CAS) = %.15g', self.e_tot - e_tot)
                        break
                    ci0 = self.ci
                    e_tot = self.e_tot

            if not isinstance(self.e_cas, (float, numpy.number)):
                self.mo_coeff, _, self.mo_energy = \
                        self.canonicalize(mo_coeff, self.ci[0], verbose=log)
            else:
                self.mo_coeff, _, self.mo_energy = \
                        self.canonicalize(mo_coeff, self.ci, verbose=log)
            self._finalize_()
            return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

        def _finalize_(self):
            cosmo.e_tot = self.e_tot
            self.e_tot = cosmo.cosmo_occ(self.make_rdm1())
            return oldCAS._finalize_(self)

    return CAS()


if __name__ == '__main__':

    #------
    # FeH2
    #------
    mol = gto.Mole()
    mol.atom = ''' Fe                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = '3-21g' #cc-pvdz'
    mol.verbose = 4
    mol.build()

    sol = COSMO(mol)
    sol.eps = -1
    mf = cosmo_for_scf(scf.RHF(mol), sol)
    mf.init_guess = 'atom' # otherwise it cannot converge to the correct result!
    escf = mf.kernel()  
    assert (abs(escf+1256.8956727624)<1.e-6)
    
    sol.eps = 37
    mf = cosmo_for_scf(scf.RHF(mol), sol)
    mf.init_guess = 'atom' # otherwise it cannot converge to the correct result!
    escf = mf.kernel()  
    assert (abs(escf+1256.8946175945)<1.e-6)

    #------
    # H2O
    #------
    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = '3-21g' #cc-pvdz'
    mol.verbose = 4
    mol.build()

    sol = COSMO(mol)
    sol.eps = -1
    mf = cosmo_for_scf(scf.RHF(mol), sol)
    mf.init_guess = 'atom' # otherwise it cannot converge to the correct result!
    escf = mf.kernel()  
 
    sol.eps = 37
    mc = mcscf.CASSCF(mf, 4, 4)
    mc = cosmo_for_mcscf(mc, sol)
    mo = mc.sort_mo([3,4,6,7])
    emc = mc.kernel(mo)[0]
    assert(abs(emc+75.6377646267)<1.e-6)
    
    mc = mcscf.CASCI(mf, 4, 4)
    mc = cosmo_for_casci(mc, sol)
    eci = mc.kernel()[0]
    assert(abs(eci+75.5789635682)<1.e-6)

    # Single-step CASCI
    sol.dm = sol._dm_guess
    #sol.casci_max_cycle = 1
    mc = mcscf.CASCI(mf, 4, 4)
    mc = cosmo_for_casci(mc, sol)
    eci = mc.kernel()[0]
    assert(abs(eci+75.5789635647)<1.e-6)

    from pyscf.mrpt.nevpt2 import sc_nevpt
    ec = sc_nevpt(mc)
    assert(abs(ec+0.128805510364)<1.e-6)
