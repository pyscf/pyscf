import os, sys

pyscf_src_dir = os.path.abspath(os.path.join(__file__, '..', '..'))
msg = ('''

    If you see this error message, it means your  PYTHONPATH  points to

        %s

    Starting from version 1.4, the PySCF package was restrucutred following
    the guidelines listed in
    http://python-guide-pt-br.readthedocs.io/en/latest/writing/structure/

    Please update your  PYTHONPATH  to

        %s

''' % (pyscf_src_dir, os.path.join(pyscf_src_dir, 'pyscf')))

raise ImportError(msg)
