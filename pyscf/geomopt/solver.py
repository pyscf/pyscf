from pyscf import grad

def gen_grad_solver(method):
    from pyscf import scf, dft, cc
    if isinstance(method, dft.rks.RKS):
        return grad.RKS(method).as_scanner()
    elif isinstance(method, scf.hf.RHF):
        return grad.RHF(method).as_scanner()
    elif isinstance(method, cc.ccsd.CCSD):
        return grad.ccsd.as_scanner(method)
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)

