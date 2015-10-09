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
from pyscf import gto, scf

'''
COSMO interface
'''

try:
    from pyscf.cosmo import settings
except ImportError:
    msg = '''settings.py not found.  Please create %s
''' % os.path.join(os.path.dirname(__file__), 'settings.py')
    sys.stderr.write(msg)

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

    def initialization(self,mol):
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
        self._built = True
        return 0

    def loadsegs(self):
        #
        # task-0
        #
        f1 = open(os.path.join(self.tmpdir, self.segfile),'rw')
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

    def occ1(self, mol, escf):
        f1 = open(os.path.join(self.tmpdir, self.parfile),'w')
        f1.write('Task = 4\n')
        f1.write('Escf = %24.15f'%escf+'\n')
        f1.write('Natoms = '+str(mol.natm)+'\n')
        for i in range(mol.natm):
            coord = mol.atom_coord(i)
            f1.write('%20.15f %20.15f %20.15f\n' % tuple(coord.tolist()))
        f1.close()
        lib.logger.info(self, 'Final outlying charge correction:')
        lib.logger.info(self, '')
        exe_cosmo(self)
        return 0

    def cosmo_fock(self, mf, dm):
        return cosmo_fock(mf, dm)

    def cosmo_occ(self, mf, dm, escf):
        return cosmo_occ(mf, dm, escf)

def exe_cosmo(mycosmo):
    mycosmo.stdout.flush()
    try:
        cmd = ' '.join((settings.COSMOEXE, mycosmo.base, mycosmo.suffix))
        output = subprocess.check_output(cmd, cwd=mycosmo.tmpdir, shell=True)
        if output:
            lib.logger.debug(mycosmo, 'COSMO output\n%s', output)
    except AttributeError:  # python 2.6 and older
        p = subprocess.check_call(cmd, cwd=mycosmo.tmpdir, shell=True)

def cosmo_fock(mf,dm):
    debug = False
    mol = mf.mol
    cosmo = mf.mol.cosmo
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
    return fock

def cosmo_occ(mf,dm,escf):
    mol = mf.mol
    cosmo = mf.mol.cosmo
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
    cosmo.occ1(mol, escf)
    return 0



# NOTE: be careful with vhf (in scf kernel) when direct_scf is applied
def cosmo_for_rhf(mf):
    oldMF = mf.__class__
    class RHF(oldMF):
        def __init__(self):
            self.__dict__.update(mf.__dict__)
            self.mol.cosmo = COSMO(self.mol)

        def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            if (self._eri is not None or self._is_mem_enough() or not self.direct_scf):
                vhf = oldMF.get_veff(self, mol, dm, hermi=hermi)
            else:
                if (self.direct_scf and isinstance(vhf_last, numpy.ndarray) and
                    hasattr(self, '_dm_last')):
                    vhf = oldMF.get_veff(self, mol, dm, self._dm_last,
                                         self._vhf_last, hermi)
                else:
                    vhf = oldMF.get_veff(self, mol, dm, hermi=hermi)
                if self.direct_scf:
                    self._dm_last = dm
            self._vhf_last = vhf  # save JK for electronic energy
            if not self.mol.cosmo._built:
                self.mol.cosmo.initialization(mol)
            return vhf + self.mol.cosmo.cosmo_fock(self, dm)

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            hf_energy, e_coul = oldMF.energy_elec(self, dm, h1e, self._vhf_last)
            hf_energy += self.mol.cosmo.ediel
            lib.logger.debug(self, 'E_diel = %.15g', self.mol.cosmo.ediel)
            return hf_energy, e_coul

        def _finalize_(self):
            dm = self.make_rdm1()
            self.mol.cosmo.cosmo_occ(self,dm,self.hf_energy)
            oldMF._finalize_(self)
    return RHF()

def cosmo_for_uhf(mf):
    oldMF = mf.__class__
    class UHF(oldMF):
        def __init__(self):
            self.__dict__.update(mf.__dict__)
            self.mol.cosmo = COSMO(self.mol)

        def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            if (self._eri is not None or self._is_mem_enough() or not self.direct_scf):
                vhf = oldMF.get_veff(self, mol, dm, hermi=hermi)
            else:
                if (self.direct_scf and isinstance(vhf_last, numpy.ndarray) and
                    hasattr(self, '_dm_last')):
                    vhf = oldMF.get_veff(self, mol, dm, self._dm_last,
                                         self._vhf_last, hermi)
                else:
                    vhf = oldMF.get_veff(self, mol, dm, hermi=hermi)
                if self.direct_scf:
                    self._dm_last = dm
            self._vhf_last = vhf  # save JK for electronic energy
            if not self.mol.cosmo._built:
                self.mol.cosmo.initialization(mol)
            return vhf + self.mol.cosmo.cosmo_fock(self, dm[0]+dm[1])

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            hf_energy, e_coul = oldMF.energy_elec(self, dm, h1e, self._vhf_last)
            hf_energy += self.mol.cosmo.ediel
            lib.logger.debug(self, 'E_diel = %.15g', self.mol.cosmo.ediel)
            return hf_energy, e_coul

        def _finalize_(self):
            dm = self.make_rdm1()
            self.mol.cosmo.cosmo_occ(self, dm[0]+dm[1], self.hf_energy)
            oldMF._finalize_(self)
    return UHF()

def comso_for_mcscf(mf):
    raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto,scf
    from pyscf import lib

    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = 'cc-pvdz'
    mol.verbose = 5
    mol.build()

    mf = cosmo_for_rhf(scf.RHF(mol))
    mf.kernel()  # -76.0030469182364
