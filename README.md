<div align="left">
  <img src="https://github.com/pyscf/pyscf-doc/blob/master/logo/pyscf-logo.png" height="80px"/>
</div>

Python-based Simulations of Chemistry Framework
-----------------------------------------------
[![Build Status](https://github.com/pyscf/pyscf/workflows/CI/badge.svg)](https://github.com/pyscf/pyscf/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/pyscf/pyscf/branch/master/graph/badge.svg)](https://codecov.io/gh/pyscf/pyscf)

2026-01-20

* [Stable release 2.12.0](https://github.com/pyscf/pyscf/releases/tag/v2.12.0)
* [Changelog](../master/CHANGELOG)
* [Documentation](http://www.pyscf.org)
* [Installation](#installation)
* [Features](../master/FEATURES)
* [News](https://pyscf.org/news.html): **3rd PySCF Developers Meeting!**


# Installation

* Install stable release:

        pip install pyscf

* New features developed in recent years are available in the pyscf-forge package:

        pip install pyscf-forge

* Certain modules are maintained as extensions of PySCF, such as dispersion,
  dmrgscf, fciqmc, icmpspt, properties, semiempirical, shciscf ... (more on
  https://github.com/pyscf) can be installed using pip:

        pip install pyscf[all]

  An individual extension can be installed:

        pip install pyscf[dispersion]

* More details of custom installation can be found in
  [installation manual](http://pyscf.org/user/install.html#build-from-source)


# Citing PySCF

## Base PySCF
The following paper should be cited in publications utilizing the PySCF program package:

[Recent developments in the PySCF program package](https://doi.org/10.1063/5.0006074),
Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan,
*J. Chem. Phys.*, **153**, 024109 (2020). doi:[10.1063/5.0006074](https://doi.org/10.1063/5.0006074)

## Density functional calculations

As PySCF does not implement density functionals, instead employing external libraries to handle their evaluation, these libraries should also be cited in publications employing PySCF for density functional calculations.

If your calculation employed Libxc, cite

[Recent developments in libxc — A comprehensive library of functionals for density functional theory](https://doi.org/10.1016/j.softx.2017.11.002),
Susi Lehtola, Conrad Steigemann, Micael J.T. Oliveira, and Miguel A.L. Marques,
*SoftwareX* **7**, 1 (2018). doi:[10.1016/j.softx.2017.11.002](https://doi.org/10.1016/j.softx.2017.11.002)

If your calculation employed XCFun, cite

[Arbitrary-order density functional response theory from automatic differentiation](https://doi.org/10.1021/ct100117s),
Ulf Ekström, Lucas Visscher, Radovan Bast, Andreas J. Thorvaldsen, and Kenneth Ruud,
*J. Chem. Theory Comput.* **6**, 1971 (2010). doi:[10.1021/ct100117s](https://doi.org/10.1021/ct100117s)

# Bug reports and feature requests

Please submit tickets on the [issues](https://github.com/pyscf/pyscf/issues) page.

