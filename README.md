<div align="left">
  <img src="https://github.com/pyscf/pyscf-doc/blob/master/logo/pyscf-logo.png" height="80px"/>
</div>

Python-based Simulations of Chemistry Framework
===============================================
[![Build Status](https://github.com/pyscf/pyscf/workflows/CI/badge.svg)](https://github.com/pyscf/pyscf/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/pyscf/pyscf/branch/master/graph/badge.svg)](https://codecov.io/gh/pyscf/pyscf)

2022-09-21

* [Stable release 2.1.1](https://github.com/pyscf/pyscf/releases/tag/v2.1.1)
* [Changelog](../master/CHANGELOG)
* [Documentation](http://www.pyscf.org)
* [Installation](#installation)
* [Features](../master/FEATURES)


Installation
------------

* Install stable release

        pip install pyscf

* (Optionally) Extensions projects geomopt, dftd3, dmrgscf, doci, icmpspt,
  properties, semiempirical, shciscf ... (more on https://github.com/pyscf) can
  be installed using pip

        pip install pyscf[all]

  Install an individual extension

        pip install pyscf[dftd3]

* More details of custom install can be found in
  [installation manual](http://pyscf.org/install.html#compiling-from-source-code)


Citing PySCF
------------
The following paper should be cited in publications utilizing the PySCF program package:

[PySCF: the Python‐based simulations of chemistry framework](https://onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1340),
Q. Sun, T. C. Berkelbach, N. S. Blunt, G. H. Booth, S. Guo, Z. Li, J. Liu,
J. McClain, E. R. Sayfutyarova, S. Sharma, S. Wouters, G. K.-L. Chan (2018),
*WIREs Comput. Mol. Sci.*, **8**: e1340. doi:[10.1002/wcms.1340](https://onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1340)

[Recent developments in the PySCF program package](https://aip.scitation.org/doi/10.1063/5.0006074),
Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan,
*J. Chem. Phys.*, **153**, 024109 (2020). doi:[10.1063/5.0006074](https://aip.scitation.org/doi/10.1063/5.0006074)


Bug reports and feature requests
--------------------------------
Please submit tickets on the [issues](https://github.com/pyscf/pyscf/issues) page.

