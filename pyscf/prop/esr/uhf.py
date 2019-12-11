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
Non-relativistic unrestricted Hartree-Fock electron spin-rotation coupling
(In testing)

Refs:
    J. Phys. Chem. A. 114, 9246, 2010
    Mole. Phys. 9, 6, 585, 1964
'''

import time
import numpy
import sys
from pyscf import lib
from pyscf.lib import logger
from pyscf.prop.zfs.uhf import koseki_charge
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.magnetizability import rhf as rhf_mag
from pyscf.prop.gtensor import uhf as uhf_gtensor
from pyscf.prop.rotational_gtensor import rhf as rhf_rotg
from pyscf.data import nist


def dia(gobj, dm0, gauge_orig=None):
    '''Note the side effects of set_common_origin'''

    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))
    mol = gobj.mol
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton

    dma, dmb = dm0
    totdm = dma + dmb
    spindm = dma - dmb
    alpha2 = nist.ALPHA ** 2
    #Many choices of qed_fac, see JPC, 101, 3388
    #qed_fac = (nist.G_ELECTRON - 1)
    #qed_fac = nist.G_ELECTRON / 2
    qed_fac = 1

    assert(not mol.has_ecp())
    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)

    e11 = numpy.zeros((3,3))
    im, mass_center = rhf_rotg.inertia_tensor(mol)
    for ia in range(mol.natm):
        Z = koseki_charge(mol.atom_charge(ia))
        R = mol.atom_coord(ia) - mass_center
        with mol.with_rinv_origin(R):
            h11 = mol.intor('int1e_drinv', comp=3)  * Z # * mol.atom_charge(ia)
            t1 = numpy.einsum('xij,ij->x', h11, spindm)
            e11 +=  numpy.dot(R, t1)*numpy.eye(3) -  numpy.kron(R, t1).reshape(3,3)

        #GIAO part of dia-magnetic constribution
        #print('kron', numpy.kron(R, t1).reshape(3,3))
            #h22 =  mol.intor('int1e_a01gp', comp=9)
            #e11 -=  Z * numpy.einsum('xij,ij->x', h22, spindm).reshape(3,3)
    gdia = e11 * alpha2 / effspin /  4.
    return gdia


def para(obj, mo10, mo_coeff, mo_occ, qed_fac=1):
    mol = obj.mol
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    #qed_fac = (nist.G_ELECTRON - 1)
    #qed_fac = nist.G_ELECTRON / 2
    orboa = mo_coeff[0][:,mo_occ[0]>0]
    orbob = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)
    dm10a = [reduce(numpy.dot, (mo_coeff[0], x, orboa.T)) for x in mo10[0]]
    dm10b = [reduce(numpy.dot, (mo_coeff[1], x, orbob.T)) for x in mo10[1]]
    dm10a = numpy.asarray([x-x.T for x in dm10a])
    dm10b = numpy.asarray([x-x.T for x in dm10b])

    hso1e = make_h01_soc1e(obj, mo_coeff, mo_occ, qed_fac)
    para1e =-numpy.einsum('xji,yij->xy', dm10a, hso1e)
    para1e += numpy.einsum('xji,yij->xy', dm10b, hso1e)
    para1e *= 1. / effspin / muB
    if obj.verbose >= logger.INFO:
        _write(obj, para1e, 'SOC(1e)/OZ') # Jia, different than in g tensor, not sure about consequence

    if obj.para_soc2e:
        raise NotImplementedError('para_soc2e = %s' % obj.para_soc2e)

    para = para1e #+ gpara2e # Jia
    return para

def make_h01_soc1e(obj, mo_coeff, mo_occ, qed_fac=1):
    mol = obj.mol
    assert(not mol.has_ecp())
    alpha2 = nist.ALPHA ** 2
    #qed_fac = (nist.G_ELECTRON - 1)
    if obj.so_eff_charge:
        hso1e = 0
        for ia in range(mol.natm):
            Z = koseki_charge(mol.atom_charge(ia))
            mol.set_rinv_origin(mol.atom_coord(ia))
            hso1e += -Z * mol.intor_asymmetric('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor_asymmetric('int1e_pnucxp', 3)
    hso1e *= qed_fac * (alpha2/4)
    return hso1e


def align(tensor):
    '''Transform the orientation of g-tensor.
    The new orientations are the eigenvector of G matrix (G=g.gT)
    '''
    w, v = numpy.linalg.eigh(numpy.dot(tensor, tensor.T))
    idxmax = abs(v).argmax(axis=0)
    v[:,v[idxmax,[0,1,2]]<0] *= -1  # format phase
    sorted_axis = numpy.argsort(idxmax)
    v = v[:,sorted_axis]
    if numpy.linalg.det(v) < 0: # ensure new axes in RHS
        v[:,2] *= -1
    g2 = reduce(numpy.dot, (v.T, tensor, v)) # g2 is gtensor after transform of orientation, Jia
    return g2, v

def _write(obj, tensor, title):
    obj.stdout.write('%s %s\n' % (title, tensor.diagonal()))
    rhf_nmr._write(obj.stdout, tensor, title+' tensor')
    w = numpy.linalg.eigvals(tensor)
    obj.stdout.write('eigenvalues: %s\n' % w)

class ESR(lib.StreamObject):
    ''' dE = S dot ESR_tensor dot J

    Attributes:
        koseki_charge : bool
        Whether to use Koseki effective SOC charge in 1-electron
        diamagnetic term and paramagnetic term.  Default is False.
    '''

    def __init__(self, mf):
        self.mol = mf.mol
        self.verbose = mf.mol.verbose
        self.stdout = mf.mol.stdout
        self.chkfile = mf.chkfile
        self._scf = mf

        self.dia_soc2e = False
        self.para_soc2e = False
        self.so_eff_charge = True # change default from False to True, Jia
        # gauge_orig=None will call GIAO. A coordinate array leads to common gauge
        self.gauge_orig = [0.0, 0.0, 0.0]
        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9

        self.mo10 = None
        self.mo_e10 = None
        self._keys = set(self.__dict__.keys()) # keep this as input for attributes. Jia

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        #self.dump_flags()

        dm0 = self._scf.make_rdm1()
        #print('dm0 from kernel', dm0.shape)
        #dm0 = dm0[0] - dm0[1]
        tt_dia = self.dia( dm0  )
        tt_para = self.para(mo1, self._scf.mo_coeff, self._scf.mo_occ)

        im, mass_center = rhf_rotg.inertia_tensor(self.mol)
        im_inv = numpy.linalg.inv(im)
        #print('im', im)
        #print('im_inv', im_inv)
        To_MHz = 1804742.77343363
        esr_para = -1.0 * numpy.einsum('ij,jk->ik', im_inv, tt_para) * To_MHz * 2
        esr_dia = -1.0 * numpy.einsum('ij,jk->ik', im_inv, tt_dia) * To_MHz * 2
        print('esr_dia', esr_dia)
        esr_tensor = esr_para  + esr_dia

        #print('esr_para', esr_para) for debug
        #print('esr_dia', esr_dia)
        #print('esr_tot', esr_tot)

        esr_tot, v = self.align(esr_tensor)
        esr_dia = reduce(numpy.dot, (v.T, esr_dia, v))
        esr_para = reduce(numpy.dot, (v.T, esr_para, v))
        _write(self, esr_dia, 'esr-tensor diamagnetic terms (in MHz)')
        _write(self, esr_para, 'esr-tensor paramagnetic terms (in MHz)')
        _write(self, esr_tot, 'esr-tensor total (in MHz)')
        return esr_tot

    def dia(self, dm0=None, gauge_orig=None):
        if gauge_orig is None: gauge_orig = self.gauge_orig
        #print('gauge_orig', gauge_orig)  Jia
        #if not (isinstance(dm0, numpy.ndarray) and dm0.ndim == 2):
        #    dm0 = dm0[0] + dm0[1]
        #print('gauge_orig', gauge_orig)
        #print('dm0', dm0)
        #if dm0 == None: dm0 = self._scf.makr_rdm1()
        return dia(self, dm0, gauge_orig)

    def para(self, mo10=None, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self._scf.mo_coeff
        if mo_occ is None:   mo_occ = self._scf.mo_occ
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(self, mo10, mo_coeff, mo_occ)

    def get_ovlp(self, mol=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        return rhf_nmr.get_ovlp(mol, gauge_orig) #Jia, try to understand this later

    solve_mo1 = uhf_nmr.solve_mo1
    get_fock = uhf_nmr.get_fock

    def align(self, tensor):
        return align(tensor)
#====================================================
if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.prop import esr
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    mf = scf.RHF(mol).run()
    esr = esr.uhf.ESR(mf)

    esr.cphf = True
    esr.gauge_orig = (0,0,0)
    print(esr.kernel())
