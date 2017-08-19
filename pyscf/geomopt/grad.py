from pyscf import grad

def gen_grad_scanner(method):
    from pyscf import scf, cc
    if isinstance(method, scf.hf.SCF) and hasattr(method, 'nuc_grad_method'):
        return method.nuc_grad_method().as_scanner()
    elif isinstance(method, cc.ccsd.CCSD):
        return grad.ccsd.as_scanner(method)
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)

