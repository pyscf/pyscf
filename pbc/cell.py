import gc
import pyscf
from pyscf.gto.mole import _symbol, _rm_digit, _std_symbol
import pseudo
import basis

def format_pseudo(pseudo_tab):
    '''Convert the input :attr:`Cell.pseudo` (dict) to the internal data format.

    ``{ atom: ( (nelec_s, nele_p, nelec_d, ...),
                rloc, nexp, (cexp_1, cexp_2, ..., cexp_nexp),
                nproj_types,
                (r1, nproj1, ( (hproj1[1,1], hproj1[1,2], ..., hproj1[1,nproj1]),
                               (hproj1[2,1], hproj1[2,2], ..., hproj1[2,nproj1]),
                               ...
                               (hproj1[nproj1,1], hproj1[nproj1,2], ...        ) )),
                (r2, nproj2, ( (hproj2[1,1], hproj2[1,2], ..., hproj2[1,nproj1]),
                ... ) )
                )
        ... }``

    Args:
        pseudo_tab : dict 
            Similar to :attr:`Cell.pseudo` (a dict), it **cannot** be a str

    Returns:
        Formatted :attr:`~Cell.pseudo`

    Examples:

    >>> pbc.format_pseudo({'H':'gth-blyp', 'He': 'gth-pade'})
    {'H': [[1], 
        0.2, 2, [-4.19596147, 0.73049821], 0], 
     'He': [[2], 
        0.2, 2, [-9.1120234, 1.69836797], 0]}
    '''
    fmt_pseudo = {}
    for atom in pseudo_tab.keys():
        symb = _symbol(atom)
        rawsymb = _rm_digit(symb)
        stdsymb = _std_symbol(rawsymb)
        symb = symb.replace(rawsymb, stdsymb)

        if isinstance(pseudo_tab[atom], str):
            fmt_pseudo[symb] = pseudo.load(pseudo_tab[atom], stdsymb)
        else:
            fmt_pseudo[symb] = pseudo_tab[atom]
    return fmt_pseudo

def format_basis(basis_tab):
    '''Convert the input :attr:`Cell.basis` to the internal data format.

    ``{ atom: (l, kappa, ((-exp, c_1, c_2, ..), nprim, nctr, ptr-exps, ptr-contraction-coeff)), ... }``

    Args:
        basis_tab : list
            Similar to :attr:`Cell.basis`, it **cannot** be a str

    Returns:
        Formated :attr:`~Cell.basis`

    Examples:

    >>> pbc.format_basis({'H':'gth-szv'})
    {'H': [[0,
        (8.3744350009, -0.0283380461),
        (1.8058681460, -0.1333810052),
        (0.4852528328, -0.3995676063),
        (0.1658236932, -0.5531027541)]]}
    '''
    fmt_basis = {}
    for atom in basis_tab.keys():
        symb = _symbol(atom)
        rawsymb = _rm_digit(symb)
        stdsymb = _std_symbol(rawsymb)
        symb = symb.replace(rawsymb, stdsymb)

        if isinstance(basis_tab[atom], str):
            fmt_basis[symb] = basis.load(basis_tab[atom], stdsymb)
        else:
            fmt_basis[symb] = basis_tab[atom]
    return fmt_basis

class Cell(pyscf.gto.Mole):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.h = None
        self.vol = 0.
        self.nimgs = []
        self.pseudo = None

    def build(self, *args, **kwargs):
        return self.build_(*args, **kwargs)

    def build_(self, *args, **kwargs):
        '''Setup Mole molecule and Cell and initialize some control parameters.  
        Whenever you change the value of the attributes of :class:`Cell`, 
        you need call this function to refresh the internal data of Cell.

        Kwargs:
            pseudo : dict or str
                To define pseudopotential.  If given, overwrite :attr:`Cell.pseudo`
        '''
        if 'pseudo' in kwargs.keys():
            self.pseudo = kwargs.pop('pseudo')

        # Set-up pseudopotential if it exists
        if self.pseudo is not None:
            # release circular referred objs
            # Note obj.x = obj.member_function causes circular referrence
            gc.collect()
        
            # Second, do pseudopotential-related things
            if isinstance(self.pseudo, str):
                # specify global pseudo for whole molecule
                uniq_atoms = set([a[0] for a in self.atom])
                self._pseudo = self.format_pseudo(dict([(a, self.pseudo)
                                                      for a in uniq_atoms]))
            else:
                self._pseudo = self.format_pseudo(self.pseudo)
            #TODO(TCB): Change this - PP gives fewer nelectrons!
            self.nelectron = self.tot_electrons()
            if (self.nelectron+self.spin) % 2 != 0:
                raise RuntimeError('Electron number %d and spin %d are not consistent\n' %
                                   (self.nelectron, self.spin))

        # Check if we're using a GTH basis
        if 'basis' in kwargs.keys():
            self.basis = kwargs['basis']
        if isinstance(self.basis, str):
            basis_name = self.basis.lower().replace(' ', '').replace('-', '').replace('_', '')
            if basis_name in ['gthaugdzvp','gthaugqzv2p','gthaugqzv3p','gthaugtzv2p','gthaugtzvp','gthdzv','gthdzvp','gthqzv2p','gthqzv3p','gthszv','gthtzv2p','gthtzvp']:
                # specify global basis for whole molecule
                uniq_atoms = set([a[0] for a in self._atom])
                self.basis = self.format_basis(dict([(a, basis_name)
                                                     for a in uniq_atoms]))
            # This sets self.basis to be internal format, and will
            # be parsed appropriately by Mole.build
                               
        # Finally, call regular Mole.build_
        pyscf.gto.Mole.build_(self,*args,**kwargs)

        self._built = True

    def format_pseudo(self, pseudo_tab):
        return format_pseudo(pseudo_tab)

    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    def atom_charge(self, atm_id):
        '''Return the atom charge, accounting for pseudopotential.'''
        if self.pseudo is None:
            # This is what the original Mole.atom_charge() returns
            CHARGE_OF  = 0
            return self._atm[atm_id,CHARGE_OF]
        else:
            # Remember, _pseudo is a dict
            nelecs = self._pseudo[ self.atom_symbol(atm_id) ][0]
            return sum(nelecs)
            
