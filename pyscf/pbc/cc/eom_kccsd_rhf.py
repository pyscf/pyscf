from pyscf.pbc.cc import eom_kccsd_rhf_ip, eom_kccsd_rhf_ea

amplitudes_to_vector_ip = eom_kccsd_rhf_ip.amplitudes_to_vector
vector_to_amplitudes_ip = eom_kccsd_rhf_ip.vector_to_amplitudes
ipccsd_matvec = eom_kccsd_rhf_ip.matvec
ipccsd_diag = eom_kccsd_rhf_ip.diag
ipccsd = eom_kccsd_rhf_ip.kernel

amplitudes_to_vector_ea = eom_kccsd_rhf_ea.amplitudes_to_vector
vector_to_amplitudes_ea = eom_kccsd_rhf_ea.vector_to_amplitudes
eaccsd_matvec = eom_kccsd_rhf_ea.matvec
eaccsd_diag = eom_kccsd_rhf_ea.diag
eaccsd = eom_kccsd_rhf_ea.kernel
