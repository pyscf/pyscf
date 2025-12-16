(load "gen-code-ft.cl")

(gen-ftao "ft_ao_ppnl_auto.c"
  '("rc"                     ( rc \| ))
  '("rc_r2_origi"            ( rc r dot r \| ))
  '("rc_r4_origi"            ( rc r dot r r dot r \| ))
)
