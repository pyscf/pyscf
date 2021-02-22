            # Taylored CC in iterations > 1
            #if True:
            if self.base.tccT1 is not None:
                log.debug("Adding tailorfunc for tailored CC.")

                tcc_mix_factor = 1

                # Transform to cluster basis
                act = cc.get_frozen_mask()
                Co = mo_coeff[:,act][:,mo_occ[act]>0]
                Cv = mo_coeff[:,act][:,mo_occ[act]==0]
                Cmfo = self.base.mo_coeff[:,self.base.mo_occ>0]
                Cmfv = self.base.mo_coeff[:,self.base.mo_occ==0]
                S = self.mf.get_ovlp()
                Ro = np.linalg.multi_dot((Co.T, S, Cmfo))
                Rv = np.linalg.multi_dot((Cv.T, S, Cmfv))

                ttcT1, ttcT2 = self.transform_amplitudes(Ro, Rv, self.base.tccT1, self.base.tccT2)
                #ttcT1 = 0
                #ttcT2 = 0

                # Get occupied bath projector
                #Pbath = self.get_local_projector(Co, inverse=True)
                ##Pbath2 = self.get_local_projector(Co)
                ##log.debug("%r", Pbath)
                ##log.debug("%r", Pbath2)
                ##1/0
                ##CSC = np.linalg.multi_dot((Co.T, S, self.C_env))
                ##Pbath2 = np.dot(CSC, CSC.T)
                ##assert np.allclose(Pbath, Pbath2)

                ##CSC = np.linalg.multi_dot((Co.T, S, np.hstack((self.C_occclst, self.C_occbath))))
                #CSC = np.linalg.multi_dot((Co.T, S, np.hstack((self.C_bath, self.C_occbath))))
                #Pbath2 = np.dot(CSC, CSC.T)
                #assert np.allclose(Pbath, Pbath2)

                #log.debug("DIFF %g", np.linalg.norm(Pbath - Pbath2))
                #log.debug("DIFF %g", np.linalg.norm(Pbath + Pbath2 - np.eye(Pbath.shape[-1])))

                def tailorfunc(T1, T2):
                    # Difference of bath to local amplitudes
                    dT1 = ttcT1 - T1
                    dT2 = ttcT2 - T2

                    # Project difference amplitudes to bath-bath block in occupied indices
                    #pT1, pT2 = self.project_amplitudes(Co, dT1, dT2, indices_T2=[0, 1])
                    ###pT1, pT2 = self.project_amplitudes(Pbath, dT1, dT2, indices_T2=[0, 1])
                    ###_, pT2_0 = self.project_amplitudes(Pbath, None, dT2, indices_T2=[0])
                    ###_, pT2_1 = self.project_amplitudes(Pbath, None, dT2, indices_T2=[1])
                    ###pT2 += (pT2_0 + pT2_1)/2

                    # Inverse=True gives the non-local (bath) part
                    pT1, pT2 = self.get_local_amplitudes(cc, dT1, dT2, inverse=True)

                    #pT12, pT22 = self.get_local_amplitudes(cc, pT1, pT2, inverse=True, symmetrize=False)
                    #assert np.allclose(pT12, pT1)
                    #assert np.allclose(pT22, pT2)
                    #pT1, pT2 = self.get_local_amplitudes(cc, dT1, dT2, variant="democratic", inverse=True)
                    # Subtract to get non-local amplitudes
                    #pT1 = dT1 - pT1
                    #pT2 = dT2 - pT2

                    log.debug("Norm of pT1=%6.2g, dT1=%6.2g, pT2=%6.2g, dT2=%6.2g", np.linalg.norm(dT1), np.linalg.norm(pT1), np.linalg.norm(dT2), np.linalg.norm(pT2))
                    # Add projected difference amplitudes
                    T1 += tcc_mix_factor*pT1
                    T2 += tcc_mix_factor*pT2
                    return T1, T2

                cc.tailorfunc = tailorfunc

            # Use FCI amplitudes
            #if True:
            #if False:
            if self.coupled_bath:
                log.debug("Coupling bath")
                for x in self.loop_clusters(exclude_self=True):
                    if not x.amplitudes:
                        continue
                    log.debug("Coupling bath with fragment %s with solver %s", x.name, x.solver)

                #act = cc.get_frozen_mask()
                #Co = mo_coeff[:,act][:,mo_occ[act]>0]
                #Cv = mo_coeff[:,act][:,mo_occ[act]==0]
                #no = Co.shape[-1]
                #nv = Cv.shape[-1]

                #T1_ext = np.zeros((no, nv))
                #T2_ext = np.zeros((no, no, nv, nv))
                #Pl = []
                #Po = []
                #Pv = []

                #for x in self.loop_clusters(exclude_self=True):
                #    if not x.amplitudes:
                #        continue
                #    log.debug("Amplitudes found in cluster %s with solver %s", x.name, x.solver)
                #    #C_x_occ = x.amplitudes["C_occ"]
                #    #C_x_vir = x.amplitudes["C_vir"]
                #    Cx_occ = x.C_occact
                #    Cx_vir = x.C_viract
                #    C1x = x.amplitudes["C1"]
                #    C2x = x.amplitudes["C2"]

                #    actx = x.cc.get_frozen_mask()
                #    assert np.allclose(Cx_occ, x.cc.mo_coeff[:,actx][x.cc.mo_occ[actx]>0])
                #    assert np.allclose(Cx_vir, x.cc.mo_coeff[:,actx][x.cc.mo_occ[actx]==0])

                #    assert Cx_occ.shape[-1] == C1x.shape[0]
                #    assert Cx_vir.shape[-1] == C1x.shape[1]

                #    T1x, T2x = amplitudes_C2T(C1x, C2x)

                #    # Project to local first index
                #    Plx = x.get_local_projector(Cx_occ)
                #    T1x = einsum("xi,ia->xa", Plx, T1x)
                #    T2x = einsum("xi,ijab->xjab", Plx, T2x)

                #    # Transform to current basis
                #    S = self.mf.get_ovlp()
                #    Rox = np.linalg.multi_dot((Co.T, S, Cx_occ))
                #    Rvx = np.linalg.multi_dot((Cv.T, S, Cx_vir))
                #    T1x, T2x = self.transform_amplitudes(Rox, Rvx, T1x, T2x)

                #    T1_ext += T1x
                #    T2_ext += T2x

                #    Plx = np.linalg.multi_dot((x.C_local.T, S, Co))
                #    Pox = np.linalg.multi_dot((x.C_occclst.T, S, Co))
                #    Pvx = np.linalg.multi_dot((x.C_virclst.T, S, Cv))
                #    Pl.append(Plx)
                #    Po.append(Pox)
                #    Pv.append(Pvx)

                #Pl = np.vstack(Pl)
                #Pl = np.dot(Pl.T, Pl)
                #Po = np.vstack(Po)
                #Po = np.dot(Po.T, Po)
                #Pv = np.vstack(Pv)
                #Pv = np.dot(Pv.T, Pv)

                #def tailorfunc(T1, T2):
                #    # Difference amplitudes
                #    dT1 = (T1_ext - T1)
                #    dT2 = (T2_ext - T2)
                #    # Project to Px
                #    #pT1 = einsum("xi,ia->xa", Pext, dT1)
                #    #pT2 = einsum("xi,ijab->xjab", Pext, dT2)
                #    pT1 = einsum("xi,ya,ia->xy", Pl, Pv, dT1)
                #    pT2 = einsum("xi,yj,za,wb,ijab->xyzw", Pl, Po, Pv, Pv, dT2)
                #    # Symmetrize pT2
                #    pT2 = (pT2 + pT2.transpose(1,0,3,2))/2
                #    T1 += pT1
                #    T2 += pT2
                #    return T1, T2
                S = self.mf.get_ovlp()

                def tailorfunc(T1, T2):
                    T1out = T1.copy()
                    T2out = T2.copy()
                    for x in self.loop_clusters(exclude_self=True):
                        if not x.amplitudes:
                            continue
                        C1x, C2x = x.amplitudes["C1"], x.amplitudes["C2"]
                        T1x, T2x = amplitudes_C2T(C1x, C2x)

                        # Remove double counting
                        # Transform to x basis [note that the sizes of the active orbitals will be changed]
                        Ro = np.linalg.multi_dot((x.C_occact.T, S, self.C_occact))
                        Rv = np.linalg.multi_dot((x.C_viract.T, S, self.C_viract))
                        T1x_dc, T2x_dc = self.transform_amplitudes(Ro, Rv, T1, T2)
                        dT1x = (T1x - T1x_dc)
                        dT2x = (T2x - T2x_dc)

                        Px = x.get_local_projector(x.C_occact)
                        pT1x = einsum("xi,ia->xa", Px, dT1x)
                        pT2x = einsum("xi,ijab->xjab", Px, dT2x)

                        # Transform back and add
                        pT1x, pT2x = self.transform_amplitudes(Ro.T, Rv.T, pT1x, pT2x)
                        T1out += pT1x
                        T2out += pT2x

                    return T1out, T2out

                cc.tailorfunc = tailorfunc


