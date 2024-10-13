'''
Conversion from the Basis Set Exchange format to PySCF format

17 Nov 2021 Susi Lehtola
'''

try:
    import basis_set_exchange
    from basis_set_exchange import lut, manip, sort
except ImportError:
    basis_set_exchange = None


def _orbital_basis(basis, make_general=False):
    '''Extracts the orbital basis from the BSE format in PySCF format'''

    r = {}

    if make_general:
        basis = manip.make_general(basis, False, use_copy=True)
        basis = sort.sort_basis(basis, use_copy=False)
    else:
        basis = sort.sort_basis(basis, use_copy=True)

    # Elements for which we have electron basis
    electron_elements = [k for k, v in basis['elements'].items() if 'electron_shells' in v]

    # List of references in the used basis
    reference_list = []

    # Electron Basis
    if electron_elements:
        for z in electron_elements:
            data = basis['elements'][z]

            sym = lut.element_sym_from_Z(z, True)

            # List of shells
            atom_shells = []
            for shell in data['electron_shells']:
                exponents = shell['exponents']
                coefficients = shell['coefficients']
                ncontr = len(coefficients)
                nprim = len(exponents)
                am = shell['angular_momentum']
                if len(am) == len(coefficients): # SP basis
                    for l, c in zip(am, coefficients):
                        shell_data = [l]
                        for iprim in range(nprim):
                            row = [float(exponents[iprim]), float(c[iprim])]
                            shell_data.append(row)
                        atom_shells.append(shell_data)
                else:
                    assert len(am) == 1
                    shell_data = [am[0]]
                    for iprim in range(nprim):
                        row = [float(coefficients[ic][iprim]) for ic in range(ncontr)]
                        row.insert(0, float(exponents[iprim]))
                        shell_data.append(row)
                    atom_shells.append(shell_data)
            r[sym] = atom_shells

            # Collect the literature references
            if 'references' in data:
                for ref in data['references']:
                    for key in ref['reference_keys']:
                        if key not in reference_list:
                            reference_list.append(key)

    return r, reference_list


def _ecp_basis(basis):
    '''Extracts the ECP from the BSE format in PySCF format'''

    r = {}

    basis = manip.make_general(basis, False, True)
    basis = sort.sort_basis(basis, False)

    # Elements for which we have ECP
    ecp_elements = [k for k, v in basis['elements'].items() if 'ecp_potentials' in v]

    # Electron Basis
    if ecp_elements:
        for z in ecp_elements:
            data = basis['elements'][z]
            sym = lut.element_sym_from_Z(z, True)

            # Sort lowest->highest
            ecp_list = sorted(data['ecp_potentials'], key=lambda x: x['angular_momentum'])

            # List of ECP
            atom_ecp = [data['ecp_electrons'], []]
            for ir, pot in enumerate(ecp_list):
                rexponents = pot['r_exponents']
                gexponents = pot['gaussian_exponents']
                coefficients = pot['coefficients']
                am = pot['angular_momentum']
                nprim = len(rexponents)

                shell_data = [am[0], []]
                # PySCF wants the data in order of rexp=0, 1, 2, ..
                for rexpval in range(max(rexponents) + 1):
                    rcontr = []
                    for i in range(nprim):
                        if rexponents[i] == rexpval:
                            rcontr.append([float(gexponents[i]), float(coefficients[0][i])])
                    shell_data[1].append(rcontr)
                atom_ecp[1].append(shell_data)
            r[sym] = atom_ecp

    return r

def _print_basis_information(basis):
    name = basis['name']
    version = basis['version']
    revision_description = basis['revision_description']
    revision_date = basis['revision_date']
    print('{} basis set, version {}'.format(name, version))
    print('Last revised on {}'.format(revision_date))
    print('Revision description: {}'.format(revision_description))

def get_basis(name, elements):
    '''
    Obtain a basis set

    Args:
        name : str
            Name of the basis set, case insensitive.
        elements : str, int or list

    Returns:
        A dict of basis set in PySCF internal basis format.
    '''
    basis = basis_set_exchange.api.get_basis(name, elements)
    return _orbital_basis(basis)[0]

def autoaux(name, elements):
    '''
    Create an auxiliary basis set for the given orbital basis set using
    the Auto-Aux algorithm.

    See also: G. L. Stoychev, A. A. Auer, and F. Neese
    Automatic Generation of Auxiliary Basis Sets
    J. Chem. Theory Comput. 13, 554 (2017)
    http://doi.org/10.1021/acs.jctc.6b01041
    '''
    obs = basis_set_exchange.api.get_basis(name, elements)
    auxbs = manip.autoaux_basis(obs)
    return _orbital_basis(auxbs)[0]

def autoabs(name, elements):
    '''
    Create a Coulomb fitting basis set for the given orbital basis set.
    See also:
    R. Yang, A. P. Rendell, and M. J. Frisch
    Automatically generated Coulomb fitting basis sets: Design and accuracy for systems containing H to Kr
    J. Chem. Phys. 127, 074102 (2007)
    http://doi.org/10.1063/1.2752807
    '''
    obs = basis_set_exchange.api.get_basis(name, elements)
    auxbs = manip.autoabs_basis(obs)
    return _orbital_basis(auxbs)[0]

if __name__ == '__main__':
    from basis_set_exchange import api, references

    # Get reference data
    reference_data = api.get_reference_data()
    #print(references)

    o631gbas = api.get_basis('6-31g', elements='O')
    #print('O 6-31G basis, BSE format\n{}'.format(o631gbas))
    _print_basis_information(o631gbas)
    o631gorb, o631gref = _orbital_basis(o631gbas)
    print('O 6-31G orbital basis, PySCF format\n{}'.format(o631gorb))
    print('Literature references')
    for ref in o631gref:
        print(references.reference_text(ref, reference_data[ref]))
    print('')

    nalanl2dzbas = api.get_basis('lanl2dz', elements='Na')
    #print('Na LANL2DZ basis, BSE format\n{}'.format(nalanl2dzbas))
    _print_basis_information(nalanl2dzbas)
    nalanl2dzorb, nalanl2dzref = _orbital_basis(nalanl2dzbas)
    print('Na LANL2DZ orbital basis, PySCF format\n{}'.format(nalanl2dzorb))
    nalanl2dzecp = _ecp_basis(nalanl2dzbas)
    print('Na LANL2DZ ECP basis, PySCF format\n{}'.format(nalanl2dzecp))
    print('Literature references')
    for ref in nalanl2dzref:
        print(references.reference_text(ref, reference_data[ref]))
    print('')
