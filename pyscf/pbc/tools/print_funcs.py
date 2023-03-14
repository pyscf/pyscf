import numpy
from pyscf.lib import logger

def print_mo_energy_occ_kpts(mf,mo_energy_kpts,mo_occ_kpts,is_uhf):

    if is_uhf:
        nocc = len(mo_energy_kpts[0][0])
        numpy.set_printoptions(precision=6,threshold=nocc,suppress=True)
        logger.debug(mf, '     k-point                  alpha mo_energy/mo_occ')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                         k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[0][k])
            logger.debug(mf, '                              %s', mo_occ_kpts[0][k])
        logger.debug(mf, '     k-point                  beta  mo_energy/mo_occ')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                         k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[1][k])
            logger.debug(mf, '                              %s', mo_occ_kpts[1][k])
        numpy.set_printoptions()
    else:
        nocc = len(mo_energy_kpts[0])
        numpy.set_printoptions(precision=6,threshold=nocc,suppress=True)
        logger.debug(mf, '     k-point                  mo_energy/mo_occ')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                         k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[k])
            logger.debug(mf, '                              %s', mo_occ_kpts[k])
        numpy.set_printoptions()

def print_mo_energy_occ(mf,mo_energy,mo_occ,is_uhf):

    if is_uhf:
        nocc = len(mo_energy[0])
        numpy.set_printoptions(precision=6,threshold=nocc,suppress=True)
        logger.debug(mf, '  alpha mo_energy/mo_occ')
        logger.debug(mf, '  %s', mo_energy[0])
        logger.debug(mf, '  %s', mo_occ[0])
        logger.debug(mf, '  beta  mo_energy/mo_occ')
        logger.debug(mf, '  %s', mo_energy[1])
        logger.debug(mf, '  %s', mo_occ[1])
        numpy.set_printoptions()
    else:
        nocc = len(mo_energy)
        numpy.set_printoptions(precision=6,threshold=nocc,suppress=True)
        logger.debug(mf, '  mo_energy/mo_occ')
        logger.debug(mf, '  %s', mo_energy)
        logger.debug(mf, '  %s', mo_occ)
        numpy.set_printoptions()

