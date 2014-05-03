#
# File: chkfile.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import cPickle as pickle

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
