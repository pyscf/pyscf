def gen_grad_scanner(method):
    if hasattr(method, 'nuc_grad_method'):
        return method.nuc_grad_method().as_scanner()
    else:
        raise NotImplementedError('Nuclear gradients of %s not available' % method)

