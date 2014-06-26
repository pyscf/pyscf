#
# File: chkfile.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import cPickle as pickle
import gto

def load_chkfile_key(chkfile, key):
    ftmp = open(chkfile, "r")
    rec = pickle.load(ftmp)
    ftmp.close()
    if rec.has_key(key):
        return rec[key]
    else:
        raise KeyError("No key %s are found in chkfile %s" \
                       % (key, chkfile))

def dump_chkfile_key(chkfile, key, value):
    ftmp = open(chkfile, "r")
    rec = pickle.load(ftmp)
    ftmp.close()
    ftmp = open(chkfile, "w")
    if not rec.has_key(key):
        rec[key] = {}
    rec[key] = value
    pickle.dump(rec, ftmp, pickle.HIGHEST_PROTOCOL)
    ftmp.close()



###########################################
def read_scf(chkfile):
    ftmp = open(chkfile, 'r')
    rec = pickle.load(ftmp)
    ftmp.close()

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.atom     = rec['mol']['atom']
    mol.basis    = rec['mol']['basis']
    mol.etb      = rec['mol']['etb']
    mol.build(False, False)

    return mol, rec['scf']

def dump_scf(mol, chkfile, hf_energy, mo_energy, mo_occ, mo_coeff):
    '''save temporary results'''
    rec = {}
    rec['mol'] = mol.pack()
    rec['scf'] = {'hf_energy': hf_energy, \
                  'mo_energy': mo_energy, \
                  'mo_occ'   : mo_occ, \
                  'mo_coeff' : mo_coeff, }
    ftmp = open(chkfile, 'w')
    pickle.dump(rec, ftmp, pickle.HIGHEST_PROTOCOL)
    ftmp.close()
