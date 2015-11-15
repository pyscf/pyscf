import pyscf.scf
import pyscf.mcscf
from pyscf.cosmo import icosmo
from pyscf.cosmo.icosmo import COSMO


def cosmo_(method, acosmo=None):
    ''' Pass in a cosmo object, to hold the COSMO potential. The COSMO object
    can be used for the next call, eg applying cosmo for a serial calling to
    CASSCF and CASCI, one can code

    sol = COSMO(mol)
    mc = cosmo_(CASSCF(mf, 2, 2), sol)
    mc.kernel()
    mc = cosmo_(CASCI(mc, 6, 6), sol)
    mc.kernel()
    '''
    if acosmo is None:
        acosmo = COSMO(method.mol)
        method.cosmo = acosmo
    if not acosmo._built:
        acosmo.initialization(method.mol)

    if isinstance(method, pyscf.scf.SCF):
        return icosmo.cosmo_for_scf(method, acosmo)
    elif isinstance(method, pyscf.mcscf.CASCI):
        return icosmo.cosmo_for_mcscf(method, acosmo)
    else:
        raise RuntimeError('COSMO %s interface is not implemented.')
