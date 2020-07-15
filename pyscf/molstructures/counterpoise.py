import logging

__all__ = [
        "mod_for_counterpoise",
        ]

log = logging.getLogger(__name__)

GHOST_PREFIX = "GHOST-"

def mod_for_counterpoise(atoms, basis, fragment, remove_basis):
    if isinstance(basis, str):
        basis = {"default" : basis}

    log.debug("Counterpoise for fragment %r", fragment)
    log.debug("Atoms before assigning ghosts:\n%r", atoms)
    log.debug("Basis before assigning ghosts:\n%r", basis)

    atoms_new = []
    if basis is not None:
        basis_new = {}
    else:
        basis_new = None
    # Remove non-fragment atoms and non-fragment basis
    if remove_basis:
        #atoms_new = [atom for atom in atoms if atom[0][0] in fragment]
        atoms_new = [atom for atom in atoms if atom[0] in fragment]
        #basis_new = {key : basis[key] for key in basis if key[0] in fragment}
        if basis is not None:
            basis_new = {key : basis[key] for key in basis if key in fragment}
            basis_new["default"] = basis["default"]
    # Remove non-fragment atoms but keep basis
    else:
        for atom in atoms:
            #if atom[0][0] not in fragment:
            if atom[0] not in fragment:
                atomlabel = GHOST_PREFIX + atom[0]
            else:
                atomlabel = atom[0]
            atoms_new.append((atomlabel, atom[1]))
        if basis is not None:
            for key, val in basis.items():
                #if key[0] not in fragment:
                if key not in fragment and key != "default":
                    key = GHOST_PREFIX + key
                basis_new[key] = val

    log.debug("Atoms after assigning ghosts:\n%r", atoms_new)
    log.debug("Basis after assigning ghosts:\n%r", basis_new)

    if basis is not None and len(basis_new) == 1:
        basis_new = basis_new["default"]

    return atoms_new, basis_new
