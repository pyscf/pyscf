# Author: Daniel S. King

'''
APC Ranked-Orbital Active Space Selection
If you find this module useful for your work, please consider citing the following:

A Ranked-Orbital Approach to Select Active Spaces for High-Throughput Multireference Computation
https://doi.org/10.1021/acs.jctc.1c00037

Large-Scale Benchmarking of Multireference Vertical-Excitation Calculations via Automated Active-Space Selection
https://doi.org/10.1021/acs.jctc.2c00630
'''

from pyscf.lib import logger
from pyscf import scf, lib
import numpy as np

class Chooser():
    """
    Chooser Class
    Implements the ranked-orbital selection scheme outlined in https://doi.org/10.1021/acs.jctc.1c00037
    Given a set of entropies, will select all orbitals for the active space and then drop the lowest-entropy orbitals
    until the size constraint max_size is met.

    Args:
        orbs: 2D Numpy Array
            Orbitals to choose from, spanning the entire basis (must be square matrix of coefficients)
        occ: 1D Numpy Array
            Orbital occupations for orbs (2,1,0); nactel will be set to the number of electrons in the selected orbitals
        entropies: 1D Numpy Array
            Importance measurement used to rank the orbitals
        max_size: Int or Tuple
            Active space size constraint.
            If tuple, interpreted as (nelecas,ncas)
            If int, interpreted as max # of orbitals

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess, chosen-active-orbital-indices

    Example:

    #Randomly ranked orbitals
    >>> import numpy as np
    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.mcscf import apc
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvtz')
    >>> mf = scf.RHF(mol).run()
    >>> entropies = np.random.choice(np.arange(len(mf.mo_occ)),len(mf.mo_occ),replace=False)
    >>> chooser = apc.Chooser(mf.mo_coeff,mf.mo_occ,entropies,max_size=(2,2))
    >>> ncas, nelecas, casorbs, active_idx = chooser.kernel()
    >>> mc = mcscf.CASSCF(mf, ncas, nelecas).run(casorbs)
    """

    def __init__(self,orbs,occ,entropies,max_size=(8,8),verbose=4):
        #Check that we have a full set of orbitals:
        assert(orbs.shape[0] == orbs.shape[1])
        assert(len(occ) == len(entropies))
        assert(len(entropies) == orbs.shape[1])

        self.log = logger.new_logger(lib.StreamObject,verbose)
        self.orbs = orbs
        self.occ = np.array(occ)
        self.entropies = np.asarray(entropies)
        self.max_size = max_size
        self.verbose = verbose

    def _ncsf(self,nactel,norbs):
        """
        Returns number of CSFs in a (nactel,nactorbs) active space
        Assumes minimum number Sz = alpha - beta (0 for even nactel, 1 for odd nactel)
        """
        from scipy.special import comb
        alpha = int(nactel//2 + nactel%2)
        beta = int(nactel//2)
        term1 = comb(norbs,alpha)*comb(norbs,beta)
        term2 = comb(norbs,alpha+1)*comb(norbs,beta-1)
        ncsf = term1-term2
        return ncsf

    def _calc_ncsf(self,active_idx):
        """
        Returns the number of CSFs given the active index using the info in self.occ
        Passes this info to self._ncsf to calculate the size of the active space
        """
        occ = self.occ
        nactel = np.sum(np.array(occ)[active_idx])
        norbs = len(active_idx)
        return self._ncsf(nactel,norbs)

    def _as_is_reasonable(self,active_idx):
        #Checks active space reasonability
        occ = self.occ

        nactel = np.sum(np.array(occ)[active_idx])
        num_os = len(np.where(occ == 1)[0])
        nactorbs = len(active_idx)

        condition1 = (nactel > 0)
        condition2 = (nactel < 2*len(active_idx))
        condition3 = (nactorbs >= num_os)

        if (condition1 and condition2 and condition3):
            return True
        else:
            self.log.debug("Active space is not reasonable!")
            self.log.debug(f"Nactel: {nactel}, Nactorbs: {nactorbs}, Num OS: {num_os}")
            if not condition1:
                self.log.debug("Condition 1 not met")
            elif not condition2:
                self.log.debug("Condition 2 not met")
            elif not condition3:
                self.log.debug("Condition 3 not met")
            return False

    def kernel(self):
        log = self.log
        entropies = self.entropies.copy()
        occ = self.occ.copy()

        #Change singly occupied orbitals to have larger entropies so they are selected:
        os_idx = np.where(occ == 1)[0]
        entropies[os_idx] = np.max(entropies) + 0.01
        if len(os_idx) > 0:
            log.info("Singly occupied orbitals found, setting them to have entropy of max + 0.01...")

        #Start with all orbitals in active space:
        active_idx = list(range(len(entropies)))
        inactive_idx = []
        secondary_idx = []

        #Size constraint:
        if isinstance(self.max_size, (tuple, list, np.ndarray)):
            nactel,norbs = self.max_size
            max_size = self._ncsf(nactel,norbs)
            as_size = self._calc_ncsf(active_idx)
        else:
            max_size = self.max_size
            as_size = len(active_idx)

        nactel = int(np.sum(np.array(occ)[active_idx]))
        nactorbs = len(active_idx)
        log.debug(f"Initial active space of ({nactel},{nactorbs}) has size {as_size}")
        log.debug(f"Maximum active space size set to {max_size}")

        #Drop orbitals until size constraint is satisfied:
        while as_size > max_size:
            nactel = int(np.sum(np.array(occ)[active_idx]))
            nactorbs = len(active_idx)
            log.debug(f"Active space of ({nactel},{nactorbs}) has size {as_size} larger than {max_size}")
            log.debug("Dropping lowest entropy orbital...")

            #Get active orbital entropies
            active_entropies = entropies[active_idx]
            ranked_active_idx = [active_idx[i] for i in np.argsort(active_entropies)]

            #Drop lowest orbital in succession, checking for reasonability:
            active_space_is_reasonable = False
            tries = 0

            while not active_space_is_reasonable:
                try:
                    dropped_idx = ranked_active_idx[tries] #Move to next possibility
                    dropped_idx_entropy = np.round(entropies[dropped_idx],4)
                    dropped_idx_occ = int(occ[dropped_idx])
                except IndexError:
                    log.error("Not enough orbitals to choose a reasonable active space!")
                    raise RuntimeError("Not enough orbitals to choose a reasonable active space!")

                new_inactive_idx = inactive_idx.copy()
                new_active_idx = active_idx.copy()
                new_secondary_idx = secondary_idx.copy()

                if dropped_idx_occ > 0:
                    new_active_idx.remove(dropped_idx)
                    new_inactive_idx += [dropped_idx]
                else:
                    new_active_idx.remove(dropped_idx)
                    new_secondary_idx += [dropped_idx]

                log.debug(f"Attempting to drop orbital {dropped_idx} \
                (occ={dropped_idx_occ}, S={dropped_idx_entropy})...")
                active_space_is_reasonable = self._as_is_reasonable(new_active_idx)
                if active_space_is_reasonable:
                    log.debug("Orbital has been dropped")
                else:
                    log.debug("Active space becomes unreasonable if this orbital is dropped, trying next option...")

                tries += 1

            inactive_idx = new_inactive_idx
            active_idx = new_active_idx
            secondary_idx = new_secondary_idx

            #Calculate new NCSFs:
            if isinstance(self.max_size,tuple):
                nactel,norbs = self.max_size
                as_size = self._calc_ncsf(active_idx)
            else:
                as_size = len(active_idx)

        #Final checks:
        assert(len(active_idx) <= len(entropies))
        assert(as_size <= max_size)

        orbs = self.orbs.copy()
        inactive_orbs = orbs[:,inactive_idx]
        active_orbs = orbs[:,active_idx]
        secondary_orbs = orbs[:,secondary_idx]
        casorbs = np.hstack([inactive_orbs,active_orbs,secondary_orbs])

        nactorbs = active_orbs.shape[1]
        active_occ = np.array(occ)[active_idx]
        nboth = int(np.sum(active_occ[np.where(active_occ == 2)])/2)
        nalpha = int(np.sum(active_occ[np.where(active_occ == 1)]))
        alpha = nboth + nalpha
        beta = nboth
        nactel = (alpha,beta)

        log.info(f"Final selected active space: ({nactel},{nactorbs})")

        return nactorbs, nactel, casorbs, active_idx

class APC():

    """
    APC Class
    Implements APC orbital entropy estimation from https://doi.org/10.1021/acs.jctc.1c00037
    APC-N implemented from https://doi.org/10.1021/acs.jctc.2c00630

    .kernel() combines this with the ranked-orbital scheme implemented in Chooser() to select
    an active space of size max_size from the orbitals in mf.mo_coeff with occupancy mf.mo_occ

    Args:
        mf: an :class:`SCF` object
            Must expose mf.mo_coeff, mf.mo_occ, mf.get_fock(), and mf.get_k()
        max_size: Int or Tuple
            Active space size constraint.
            If tuple, interpreted as (nelecas,ncas)
            If int interpreted as max # of orbitals
        n: Int
            Number of times to remove highest-entropy virtual orbitals in entropy calculation.
            A higher value will tend to select active spaces with less doubly occupied orbitals.

    Kwargs:
        eps: Float
            Small offset added to singly occupied and removed virtual orbital entropies (can generally be ignored)

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess (following AVAS convention)

    Example:
    >>> import numpy as np
    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.mcscf import apc
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvtz')
    >>> mf = scf.RHF(mol).run()
    >>> myapc = apc.APC(mf,max_size=2)
    >>> ncas,nelecas,casorbs = myapc.kernel()
    >>> mc = mcscf.CASSCF(mf, ncas, nelecas).run(casorbs)
    """

    def __init__(self,mf,max_size=(8,8),n=2,eps=1e-3,verbose=4):
        self.log = logger.new_logger(lib.StreamObject,verbose)
        self.mf = mf
        self.n = n
        self.eps = eps
        assert(eps > 0) #Check that eps > 0
        self.max_size = max_size
        self.verbose = verbose

    def _apc(self,orbs,occ,f_mo,k_mo):
        """
        Calculates APC entropies for given orbitals, occupations, and F and K matrix elements
        Singly occupied orbitals are set to max value of other orbitals + self.eps

        Args:
            orbs: 2D Numpy Array
                A nbasis x nmo array of candidate AS orbitals
            occ: 1D Numpy Array
                Orbital occupations for orbs (2,1,0)
            f_mo: 2D Numpy Array
                Fock operator in the basis of the orbs (nmo x nmo)
            k_mo: 2D Numpy Array
                Exchange operator in the basis of the orbs (nmo x nmo)
        """
        eps = self.eps
        docc_idx = np.where(occ == 2)[0]
        os_idx = np.where(occ == 1)[0]
        virt_idx = np.where(occ == 0)[0]

        #Calculate APCs
        apcs = np.zeros([len(docc_idx),len(virt_idx)])
        for i,d in enumerate(docc_idx):
            for j,v in enumerate(virt_idx):
                k12 = 0.5*k_mo[v,v]
                delta = f_mo[v,v] - f_mo[d,d]
                c = -k12/(delta + np.sqrt(k12**2 + delta**2))
                apcs[i,j] = c

        #Calculate entropies
        apc_entropies = np.zeros(orbs.shape[1])
        for o in range(orbs.shape[1]):

            #Collect APCs for this orbital:
            if o in os_idx:
                continue #Fill in later with max value
            elif o in docc_idx:
                idx = np.where(docc_idx == o)[0][0]
                apcs_o = apcs[idx,:]
            elif o in virt_idx:
                idx = np.where(virt_idx == o)[0][0]
                apcs_o = apcs[:,idx]

            #Normalize APCs:
            cis = apcs_o
            cis2 = cis**2
            sumci2 = np.sum(cis2)
            norm = np.sqrt((sumci2 + 1))
            cisnorm = cis/norm

            #Square Normalized APCs to calculate entropies:
            cisnorm2 = cisnorm**2
            assert((cisnorm2 < 1).any().all())
            sumcisnorm2 = np.sum(cisnorm2)
            assert(np.allclose((sumcisnorm2 + (1/norm)**2),1,atol=1e-6))
            exent = -sumcisnorm2 * np.log(sumcisnorm2)
            gsent = -(1/norm)**2 * np.log((1/norm)**2)
            ent = exent + gsent
            apc_entropies[o] = ent

        #Assign max value to singly occupied orbitals plus some small value:
        apc_entropies[os_idx] = np.max(apc_entropies) + eps

        return apc_entropies

    def _calc_apc_entropies(self,mf):
        """
        Implements the "APC-N" approach in which high-entropy virtual orbitals are repeatedly set to singly occupied
        Then sets the singly occupied orbitals and previously removed orbitals to high values
        Reads the value of n from self.n

        Args:
            mf: an :class:`SCF` object
                Must expose mf.mo_coeff, mf.mo_occ, mf.get_fock(), and mf.get_k()
        """

        log = self.log
        n = self.n
        eps = self.eps
        log.info(f"Calculating APC entropies (N={n})...")

        f_ao = mf.get_fock()
        k_ao = mf.get_k()

        if isinstance(mf, scf.uhf.UHF):
            log.note('UHF object found. APC uses averaged F, summed K, summed occupation, and alpha orbitals.')
            orbs = mf.mo_coeff[0]
            occ = mf.mo_occ.sum(axis=0) #summed occupation
            f_ao = np.sum(f_ao,axis=0)/2 #averaged fock
            k_ao = np.sum(k_ao,axis=0) #summed exchange
        elif isinstance(mf, scf.rohf.ROHF):
            log.note('ROHF object found. APC uses summed K')
            orbs = mf.mo_coeff
            occ = mf.mo_occ.copy()
            k_ao = np.sum(k_ao,axis=0) #summed exchange
        else:
            orbs = mf.mo_coeff
            occ = mf.mo_occ.copy()

        #Calculate f and k in mo basis
        log.info("Transforming F and K to MO basis...")
        f_mo = np.linalg.multi_dot([orbs.T,f_ao,orbs])
        k_mo = np.linalg.multi_dot([orbs.T,k_ao,orbs])

        #Calculate entropies
        removed_idx = []
        original_os = np.where(occ == 1)[0]

        log.info("Calculating initial APC entropies...")
        apc_entropies = self._apc(orbs,occ,f_mo,k_mo)

        for loop_n in range(n):
            if loop_n > 0:
                log.info(f"Calculating APC entropies (Round {loop_n})...")
                apc_entropies = self._apc(orbs,occ,f_mo,k_mo)
            maxS = np.round(np.max(apc_entropies),5)
            log.info(f"Maximum entropy: {maxS}")

            #Remove highest virtual and set occ to 1
            virt_idx = np.where(occ == 0)[0]
            to_remove = virt_idx[np.argmax(apc_entropies[virt_idx])]
            removed_idx += [to_remove]
            log.info(f"Setting maximum virtual orbitals {removed_idx} to occupation 1...")
            occ[removed_idx] = 1

        log.info("Calculating final APC entropies...")
        apc_entropies = self._apc(orbs,occ,f_mo,k_mo)
        maxS = np.round(np.max(apc_entropies),5)
        log.info(f"Maximum entropy: {maxS}")

        #Iterate over os and removed virtuals and set to max in order:
        maxs = np.max(apc_entropies)
        for i,o in enumerate(original_os):
            apc_entropies[o] = maxs + 2*eps - i*eps*1e-2
        for i,o in enumerate(removed_idx):
            apc_entropies[o] = maxs + eps - i*eps*1e-2

        return apc_entropies

    def kernel(self):
        log = self.log
        log.info('\n** APC Active Space Selection **')
        entropies = self._calc_apc_entropies(self.mf)
        self.entropies = entropies

        if isinstance(self.mf, scf.uhf.UHF):
            orbs = self.mf.mo_coeff[0] #alpha orbitals
            occ = self.mf.mo_occ.sum(axis=0) #summed occupation
        else:
            orbs = self.mf.mo_coeff
            occ = self.mf.mo_occ

        max_size = self.max_size
        log.info(f"Choosing active space with ranked orbital approach (max_size = {max_size})...")
        chooser = Chooser(orbs,occ,entropies,max_size,verbose=self.verbose)
        nactorbs, nactel, casorbs, active_idx = chooser.kernel()
        self.active_idx = active_idx
        return nactorbs, nactel, casorbs
