import numpy as np
from pyscf.grad import lpdft as lpdft_grad


class Gradients(lpdft_grad.Gradients):
    """L-PDFT gradients for StateAverageMixFCISolver"""

    def __init__(self, pdft, state=None):
        super().__init__(pdft, state=state)

        if getattr(pdft, "_irrep_slices", None) is not None:
            spin_states = np.empty(len(pdft.weights), dtype=int)
            for iblk, slices in enumerate(pdft._irrep_slices):
                spin_states[slices] = iblk
            self.spin_states = list(spin_states)
        elif hasattr(pdft, "spin_states") and pdft.spin_states is not None:
            self.spin_states = list(pdft.spin_states)

    def project_Aop(self, Aop, ci, state):
        """
        Override original L-PDFT CI projection and project 
        out redundant DOF for the CI components. 
        Only for CI components within each mix-spin solver block
        """
        if ci is None:
            ci = self.base.ci

        def my_Aop(x):
            Ax = Aop(x)
            Ax_orb, Ax_ci = self.unpack_uniq_var(Ax)

            for i, j in np.ndindex(self.nroots, self.nroots):
                if self.spin_states[i] != self.spin_states[j]:
                    continue
                if np.shape(Ax_ci[i]) != np.shape(ci[j]):
                    continue
                Ax_ci[i] -= np.dot(Ax_ci[i].ravel(), ci[j].ravel()) * ci[j]

            return self.pack_uniq_var(Ax_orb, Ax_ci)

        return my_Aop
