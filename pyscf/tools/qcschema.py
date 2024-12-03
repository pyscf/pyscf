'''
This lib loads results that are in a QCSchema format json.
Funcs below can recreate the mol and scf objects from the info in the json.
'''
import json
import numpy as np
import pyscf
from pyscf.lib.parameters import BOHR

def load_qcschema_json( file_name ):
    '''
    Does: loads qcschema format json into a dictionary
    Input:
        file_name: qcschema format json file

    Returns: dict in qcschema format
    '''
    # load qcschema output json file
    data = None
    with open(file_name,'r') as f:
        data = json.load(f)
    return data

def load_qcschema_go_final_json( file_name ):
    '''
    Does: loads qcschema format geometry optimization json
          and returns only the optimized 'final' geometry
          qcschema info as a dictionary.

    Input:
        file_name: qcschema format json file

    Returns: dict in qcschema format
    '''
    # load qcschema GO output json file
    # and return last 'trajectory' point's entries
    # (this is the optimized molecule)
    data = None
    temp = None
    with open(file_name,'r') as f:
        temp = json.load(f)
    data = temp["trajectory"][-1]
    return data

def load_qcschema_molecule(qcschema_dict, to_Angstrom=False, xyz=False, mol_select=1, step=0):
    '''
    Does: Loads molecule from qcschema format dict.
          Molecule may be single point molecule or from a geometry optimization/trajectory.

    Input:
        syms: atom symbols (qcschema format)
        coords: x y z coordinates (in qcschema format)
        mol_select: specifies which molecule to load from qcschema format results.
                    Default loads 'molecule' from qcschema.
                    Geometry optimizations or trajectories have mutliple geometries saved in the schema.
                    mol_select = 1 (default) molecule from standard qcschema format
                               = 2 initial_molecule in GO or traj qcschema
                               = 3 final_molecule in GO or traj qcschema
                               = 4 a specific step in the GO or traj qcschema, specify with 'step' arg.
        step: for geometry optimization or trajectory, which have multiple molecules in a qcschema output.
              This specifies which step to load the molecule from.
        to_Angstrom (optional): convert coordinates to Angstrom (default is Bohr)
        xyz (optional): controls output (see below)

    Returns:
        xyz=False (default): 'atom x y z' format string
        xyz=True: output a string in xyz file format
                  i.e. first line is number of atoms.
    '''
    if(mol_select == 1):
        syms = np.array(qcschema_dict["molecule"]["symbols"])
        geo = np.array(qcschema_dict["molecule"]["geometry"])
    elif(mol_select == 2):
        syms = np.array(qcschema_dict["initial_molecule"]["symbols"])
        geo = np.array(qcschema_dict["initial_molecule"]["geometry"])
    elif(mol_select == 3):
        syms = np.array(qcschema_dict["final_molecule"]["symbols"])
        geo = np.array(qcschema_dict["final_molecule"]["geometry"])
    elif(mol_select == 4):
        # for geometry or md, can load a specific geometry
        syms = np.array(qcschema_dict["trajectory"][step]["molecule"]["symbols"])
        geo = np.array(qcschema_dict["trajectory"][step]["molecule"]["geometry"])

    if(to_Angstrom):
        # convert Bohr to Angstrom
        geo = geo*BOHR

    NAtoms = len(syms)
    geo = np.reshape(geo, (NAtoms,3))

    PySCF_atoms = list(zip(syms, geo))

    # Return as string or return as xyz-format string (i.e. top is NAtoms,blankline)
    if(xyz):
        bldstr = f'{NAtoms}\n\n'
        for element, coordinates in PySCF_atoms:
            bldstr += f'{element} {coordinates[0]}, {coordinates[1]}, {coordinates[2]}\n'
            PySCF_atoms = bldstr
    return PySCF_atoms

def load_qcschema_hessian(qcschema_dict):
    '''
    Does: loads hessian from qcschema format dictionary
    Input:
        qcschema_dict

    Returns: hessian with format (N,N,3,3)
    '''
    # qcschema_dict: pass in dict that has the qcschema output json loaded into it

    # load qcschema hessian
    qc_h = []
    qc_h = qcschema_dict["return_result"]

    # Get Number of atoms N
    syms = np.array(qcschema_dict["molecule"]["symbols"])
    NAtom = len(syms)

    # reshape from (3N)**2 array to (N,N,3,3)
    hessian = np.array(qc_h).reshape(NAtom,NAtom,3,3)
    return hessian

def load_qcschema_scf_info(qcschema_dict):
    '''
    Does: loads scf info from qcschema format dictionary
    Input:
        qcschema_dict

    Returns:
        scf_dict: contains the relevent scf info only
    '''

    # Restricted wfn has schema scf_occupations_a occ of 1 or 0.
    # Need to double if rhf/rks/rohf
    method = qcschema_dict["keywords"]["scf"]["method"]
    if(method == 'rks' or method == 'roks' or method == 'rhf' or method == 'rohf'):
        OccFactor = 2.0
        have_beta = False
    elif(method == 'uks' or method == 'uhf'):
        OccFactor = 1.0
        have_beta = True
    elif(method == 'gks' or method == 'ghf'):
        OccFactor = 1.0
        have_beta = False
    else:
        raise RuntimeError('qcschema: cannot determine method..exit')
        return

    # need to reshape MO coefficients for PySCF shape.
    nao = qcschema_dict["properties"]["calcinfo_nbasis"]
    # nmo info often missing
    try:
        nmo = qcschema_dict["properties"]["calcinfo_nmo"]
    except KeyError:
        # key not provided..so we make an assumption
        # note: assumes nmo=nao which isn't the case if linear dependencies etc.
        # ..so may give error when reading coeffs
        nmo = nao
    assert nmo == nao

    # get the 4 things that PySCF wants
    # ...remembering to reshape coeffs and scale occupancies.
    e_tot = float( qcschema_dict["properties"]["return_energy"] )
    mo_coeff = np.reshape(qcschema_dict["wavefunction"]["scf_orbitals_a"],(nao,nmo))
    mo_occ = np.array( qcschema_dict["wavefunction"]["scf_occupations_a"] )*OccFactor
    mo_energy = np.array( qcschema_dict["wavefunction"]["scf_eigenvalues_a"] )
    if(have_beta):
        # for each useful piece of info we need to combine alpha and beta into 2d array, with alpha first
        # MO occupations
        mo_occ_beta = qcschema_dict["wavefunction"]["scf_occupations_b"]
        mo_occ = np.vstack( (mo_occ, mo_occ_beta) )
        # MO coefficients
        mo_coeff_beta = np.reshape(qcschema_dict["wavefunction"]["scf_orbitals_b"],(nao,nmo))
        mo_coeff = np.vstack( (mo_coeff,mo_coeff_beta))
        mo_coeff = np.reshape(mo_coeff,(2,nao,nmo))
        # MO energies
        mo_energy_beta = np.array( qcschema_dict["wavefunction"]["scf_eigenvalues_b"] )
        mo_energy = np.vstack( (mo_energy, mo_energy_beta) )
        # etot obviously doesn't need manipulation

    # convert to dictionary for PySCF
    scf_dic = {'e_tot'    : e_tot,
               'mo_energy': mo_energy,
               'mo_occ'   : mo_occ,
               'mo_coeff' : mo_coeff}

    return scf_dic


def recreate_mol_obj(qcschema_dict,to_Angstrom=False):
    '''
    Does: recreates mol object from qcschema format dictionary
    Input:
        qcschema_dict
        to_Angstrom: optional bool to convert geometry to Angstrom (default is Bohr)

    Returns: mol object
    '''

    ## Mol info: ##
    PySCF_charge = int( qcschema_dict["molecule"]["molecular_charge"] )
    # PySCF 'spin' is number of unpaired electrons, it will be mult-1
    PySCF_spin = int( qcschema_dict["molecule"]["molecular_multiplicity"] ) - 1
    PySCF_basis = str( qcschema_dict["model"]["basis"] )

    # Cartesian/Pure basis
    PySCF_cart = bool( qcschema_dict["keywords"]["basisSet"]["cartesian"] )

    # Get molecular structure.
    PySCF_atoms = load_qcschema_molecule(qcschema_dict, to_Angstrom,False)

    # Unit Bohr or Angstrom. QCSchema default is Bohr but can change here.
    if(to_Angstrom):
        units='A'
    else:
        units='B'

    ## Create mol ##
    mol = pyscf.gto.Mole(atom=PySCF_atoms,basis=PySCF_basis,ecp=PySCF_basis,
                         charge=PySCF_charge,spin=PySCF_spin,cart=PySCF_cart,unit=units)
    mol.build(False,False)

    return mol

def recreate_scf_obj(qcschema_dict,mol):
    '''
    Does: recreates scf object from qcschema format dictionary
    Input:
        qcschema_dict
        mol object

    Returns: scf object
    '''
    # load info from qcschema needed for scf obj
    scf_dict = load_qcschema_scf_info(qcschema_dict)

    # create scf object
    method =  qcschema_dict["keywords"]["scf"]["method"]
    if(method =='rks'):
        ks = mol.RKS()
    elif(method =='uks'):
        ks = mol.UKS()
    elif(method =='rhf'):
        ks = mol.RHF()
    elif(method =='uhf'):
        ks = mol.UHF()
    elif(method =='gks'):
        ks = mol.GKS()
    elif(method =='ghf'):
        ks = mol.GHF()
    else:
        raise RuntimeError('qcschema: cannot determine method..exit')
        return

    # get functional
    if(method == 'rks' or method == 'uks' or method == 'gks'):
        functional = qcschema_dict["keywords"]["xcFunctional"]["name"]
        ks.xc = functional

    # Load 4 key pieces of info we got from json into SCF object
    ks.mo_coeff = scf_dict["mo_coeff"]
    ks.mo_energy = scf_dict["mo_energy"]
    ks.mo_occ = scf_dict["mo_occ"]
    ks.e_tot = scf_dict["e_tot"]
    return ks
