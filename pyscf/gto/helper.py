import numpy as np

def canonical_orth(ovlp, threshold=1e-7):
    """Löwdin's canonical orthogonalization."""
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = np.diag(np.power(np.diag(ovlp), -0.5))
    novlp = np.linalg.multi_dot((normlz, ovlp, normlz))
    # Form vectors for normalized overlap matrix
    se, sv = np.linalg.eigh(novlp)
    keep = (se >= threshold)
    x = sv[:,keep] / np.sqrt(se[keep])
    # Plug normalization back in
    x = np.dot(normlz, x)
    return x
