from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt




def func(t, y, k1, k_1, k2, k_2, k3, k_3, k4, k_4, k5, k_5, k6, k_6, k7, k_7, k8, k_8,
         k9, k_9, k10, k_10, k11, k_11, k12, k_12, k13, k_13, k14, k_14, k15, k_15, k16,
         k_16, k17, k_17, k18, k_18, k19, k_19, k20, k_20, k21, k_21, k22, k_22, k23, k_23,
         k24, k_24, k25, k_25, fg6p, fnadpplus, fhplus, fnadph, fh2o, fco2, fribo5p, kout_g6p,
         kout_nadpplus, kout_hplus, kout_nadph, kout_h2o, kout_co2, kout_ribo5p):
    (g6p, g6pd, g6p_g6pd, nadpplus, g6p_g6pd_nadpplus, g6pd_nadpplus, pg6dl_g6pd_nadph, hplus,
     pg6dl, g6pd_nadph, nadph, pg6dl_g6pd, lac, pg6dl_lac, h2o, pg6_lac, pg6,
     pg6d, pg6_pg6d, pg6_pg6d_nadpplus, pg6d_nadpplus, k3pg_pg6d_nadph, ribu5p_pg6d_nadph, co2,
     ribu5p, pg6d_nadph, ribu5p_pg6d, ppi, ribu5p_ppi, ribo5p_ppi, ribo5p) = y

    dydt = [
        # Cg6p
        -k1*(g6p)*(g6pd) + k_1*(g6p_g6pd) - k4*(g6p) *
        (g6pd_nadpplus) + k_4*(g6p_g6pd_nadpplus) + fg6p - kout_g6p*g6p,

        # Cg6pd
        -k1*(g6p)*(g6pd) + k_1*(g6p_g6pd) - k3*(g6pd)*(nadpplus) + k_3*(g6pd_nadpplus) +
        k7*(g6pd_nadph) - k_7*(g6pd)*(nadph) +
        k9*(pg6dl_g6pd) - k_9*(g6pd)*(pg6dl),

        # Cg6p_g6pd =
        k1*(g6p)*(g6pd) - k_1*(g6p_g6pd) - k2 * \
        (g6p_g6pd)*(nadpplus) + k_2*(g6p_g6pd_nadpplus),

        # Cnadpplus=
        -k2*(g6p_g6pd)*(nadpplus) + k_2*(g6p_g6pd_nadpplus) - k3*(g6pd)*(nadpplus) + k_3*(g6pd_nadpplus) - \
        k14*(pg6_pg6d)*(nadpplus) + k_14*(pg6_pg6d_nadpplus) - \
        k15*(pg6d)*(nadpplus) + k_15*(pg6d_nadpplus) + \
        fnadpplus - kout_nadpplus*nadpplus,

        # Cg6p_g6pd_nadpplus =
        k2*(g6p_g6pd)*(nadpplus) - k_2*(g6p_g6pd_nadpplus) + k4*(g6p)*(g6pd_nadpplus) - \
        k_4*(g6p_g6pd_nadpplus) - k5*(g6p_g6pd_nadpplus) + \
        k_5*(pg6dl_g6pd_nadph)*(hplus),

        # Cg6pd_nadpplus =
        k3*(g6pd)*(nadpplus) - k_3*(g6pd_nadpplus) - k4 * \
        (g6p)*(g6pd_nadpplus) + k_4*(g6p_g6pd_nadpplus),

        # Cpg6dl_g6pd_nadph=
        k5*(g6p_g6pd_nadpplus) - k_5*(pg6dl_g6pd_nadph)*(hplus) - \
        k6*(pg6dl_g6pd_nadph) + k_6*(pg6dl)*(g6pd_nadph) - \
        k8*(pg6dl_g6pd_nadph) + k_8*(pg6dl_g6pd)*(nadph),

        # Chplus =
        k5*(g6p_g6pd_nadpplus) - k_5*(pg6dl_g6pd_nadph)*(hplus) + \
        k17*(pg6_pg6d_nadpplus) - k_17*(k3pg_pg6d_nadph) * \
        (hplus) + fhplus - kout_hplus*hplus,

        # Cpg6dl =
        k6*(pg6dl_g6pd_nadph) - k_6*(pg6dl)*(g6pd_nadph) + k9*(pg6dl_g6pd) - \
        k_9*(pg6dl)*(g6pd) - k10*(pg6dl)*(lac) + k_10*(pg6dl_lac),

        # Cg6pd_nadph =
        k6*(pg6dl_g6pd_nadph) - k_6*(pg6dl)*(g6pd_nadph) - \
        k7*(g6pd_nadph) + k_7*(g6pd)*(nadph),

        # Cnadph =
        k7*(g6pd_nadph) - k_7*(g6pd)*(nadph) + k8*(pg6dl_g6pd_nadph) - k_8*(pg6dl_g6pd)*(nadph) + \
        k20*(pg6d_nadph) - k_20*(pg6d)*(nadph) + k21 * \
        (ribu5p_pg6d_nadph) - k_21*(ribu5p_pg6d) * \
        (nadph) + fnadph - kout_nadph*nadph,

        # Cpg6dl_g6pd =
        k8*(pg6dl_g6pd_nadph) - k_8*(pg6dl_g6pd) * \
        (nadph) - k9*(pg6dl_g6pd) + k_9*(pg6dl)*(g6pd),

        # Clac =
        -k10*(pg6dl)*(lac) + k_10*(pg6dl_lac) + \
        k12*(pg6_lac) - k_12*(pg6)*(lac),

        # Cpg6dl_lac =
        k10*(pg6dl)*(lac) - k_10*(pg6dl_lac) - \
        k11*(pg6dl_lac)*(h2o) + k_11*(pg6_lac),

        # CH2O =
        -k11*(pg6dl_lac)*(h2o) + k_11*(pg6_lac) + fh2o - kout_h2o*h2o,

        # Cpg6_lac =
        k11*(pg6dl_lac)*(h2o) - k_11*(pg6_lac) - \
        k12*(pg6_lac) + k_12*(pg6)*(lac),

        # Cpg6 =
        k12*(pg6_lac) - k_12*(pg6)*(lac) - k13*(pg6)*(pg6d) + k_13 * \
        (pg6_pg6d) - k16*(pg6)*(pg6d_nadpplus) + k_16*(pg6_pg6d_nadpplus),

        # Cpg6d =
        -k13*(pg6)*(pg6d) + k_13*(pg6_pg6d) - k15*(pg6d)*(nadpplus) + k_15*(pg6d_nadpplus) + \
        k20*(pg6d_nadph) - k_20*(pg6d)*(nadph) + \
        k22*(ribu5p_pg6d) - k_22*(ribu5p)*(pg6d),

        # Cpg6_pg6d =
        k13*(pg6)*(pg6d) - k_13*(pg6_pg6d) - k14 * \
        (pg6_pg6d)*(nadpplus) + k_14*(pg6_pg6d_nadpplus),

        # Cpg6_pg6d_nadpplus =
        k14*(pg6_pg6d)*(nadpplus) - k_14*(pg6_pg6d_nadpplus) + k16*(pg6)*(pg6d_nadpplus) - \
        k_16*(pg6_pg6d_nadpplus) - k17*(pg6_pg6d_nadpplus) + \
        k_17*(k3pg_pg6d_nadph)*(hplus),

        # Cpg6d_nadpplus =
        k15*(pg6d)*(nadpplus) - k_15*(pg6d_nadpplus) - k16 * \
        (pg6)*(pg6d_nadpplus) + k_16*(pg6_pg6d_nadpplus),

        # Ck3pg_pg6d_nadph =
        k17*(pg6_pg6d_nadpplus) - k_17*(k3pg_pg6d_nadph)*(hplus) - \
        k18*(k3pg_pg6d_nadph) + k_18*(ribu5p_pg6d_nadph)*(co2),

        # Cribu5p_pg6d_nadph =
        k18*(k3pg_pg6d_nadph) - k_18*(ribu5p_pg6d_nadph)*(co2) - k19*(ribu5p_pg6d_nadph) + \
        k_19*(ribu5p)*(pg6d_nadph) - k21 * \
        (ribu5p_pg6d_nadph) + k_21*(ribu5p_pg6d)*(nadph),

        # CCO2 =
        k18*(k3pg_pg6d_nadph) - k_18*(ribu5p_pg6d_nadph) * \
        (co2) + fco2 - kout_co2*co2,

        # Cribu5p =
        k19*(ribu5p_pg6d_nadph) - k_19*(ribu5p)*(pg6d_nadph) + k22*(ribu5p_pg6d) - \
        k_22*(ribu5p)*(pg6d) - k23*(ribu5p)*(ppi) + k_23*(ribu5p_ppi),

        # Cpg6d_nadph =
        k19*(ribu5p_pg6d_nadph) - k_19*(ribu5p)*(pg6d_nadph) - \
        k20*(pg6d_nadph) + k_20*(pg6d)*(nadph),

        # Cribu5p_pg6d =
        k21*(ribu5p_pg6d_nadph) - k_21*(ribu5p_pg6d) * \
        (nadph) - k22*(ribu5p_pg6d) + k_22*(ribu5p)*(pg6d),

        # Cppi =
        -k23*(ribu5p)*(ppi) + k_23*(ribu5p_ppi) + \
        k25*(ribo5p_ppi) - k_25*(ribo5p)*(ppi),

        # Cribu5p_ppi =
        k23*(ribu5p)*(ppi) - k_23*(ribu5p_ppi) - \
        k24*(ribu5p_ppi) + k_24*(ribo5p_ppi),

        # Cribo5p_ppi =
        k24*(ribu5p_ppi) - k_24*(ribo5p_ppi) - \
        k25*(ribo5p_ppi) + k_25*(ribo5p)*(ppi),

        # Cribo5p =
        k25*(ribo5p_ppi) - k_25*(ribo5p)*(ppi) + fribo5p - kout_ribo5p*ribo5p,

    ]
    return dydt


def pentose_phosphate():
    # step 1
    k1 = 1.0
    k_1 = 0.01
    k2 = 1.0
    k_2 = 0.01
    k3 = 1.0
    k_3 = 0.01
    k4 = 1.0
    k_4 = 0.01
    k5 = 0.01  # ** CATALYTIC STEP **
    k_5 = 0.00001
    k6 = 1.0
    k_6 = 0.01
    k7 = 1.0
    k_7 = 0.01
    k8 = 1.0
    k_8 = 0.01
    k9 = 1.0
    k_9 = 0.01

    # step 2
    k10 = 1.0
    k_10 = 0.01
    k11 = 0.01        # ** CATALYTIC STEP **
    k_11 = 0.00001
    k12 = 1.0
    k_12 = 0.01

    # step 3
    k13 = 1.0
    k_13 = 0.01
    k14 = 1.0
    k_14 = 0.01
    k15 = 1.0
    k_15 = 0.01
    k16 = 1.0
    k_16 = 0.01
    k17 = 0.01     # ** CATALYTIC STEP **
    k_17 = 0.00001
    k18 = 0.01      # ** CATALYTIC STEP **
    k_18 = 0.00001
    k19 = 1.0
    k_19 = 0.01
    k20 = 1.0
    k_20 = 0.01
    k21 = 1.0
    k_21 = 0.01
    k22 = 1.0
    k_22 = 0.01

    # step 4
    k23 = 1.0
    k_23 = 0.01
    k24 = 0.01     # ** CATALYTIC STEP **
    k_24 = 0.00001
    k25 = 1.0
    k_25 = 0.01

    # external substrate/product/co-factor/metabolite concs
    ext_g6p = 10.0
    ext_nadpplus = 50.00
    ext_hplus = 10.0
    ext_nadph = 10.0
    ext_h2o = 10.0
    ext_co2 = 10.0
    ext_ribo5p = 10.0

    # input flow constants

    kin_g6p = 0.05
    kin_nadpplus = 0.0005
    kin_hplus = 0.0005
    kin_nadph = 0.0005
    kin_h2o = 0.0005
    kin_co2 = 0.0005
    kin_ribo5p = 0.0005

    # output flow rate constants
    kout_g6p = 0.0005
    kout_nadpplus = 0.0005
    kout_hplus = 0.0005
    kout_nadph = 0.0005
    kout_h2o = 0.0005
    kout_co2 = 0.0005
    kout_ribo5p = 0.0005

    # input fluxes
    fg6p = ext_g6p * kin_g6p
    fnadpplus = ext_nadpplus * kin_nadpplus
    fhplus = ext_hplus * kin_hplus
    fnadph = ext_nadph * kin_nadph
    fh2o = ext_h2o * kin_h2o
    fco2 = ext_co2 * kin_co2
    fribo5p = ext_ribo5p * kin_ribo5p

    yy0 = [
        0.0,  # 0  - g6p
        1.0,   # 1  - g6pd
        0.0,   # 2  - g6p_g6pd
        0.0,   # 3  - nadpplus
        0.0,   # 4  - g6p_g6pd_nadpplus
        0.0,   # 5  - g6pd_nadpplus
        0.0,   # 6  - pg6dl_g6pd_nadph
        0.0,   # 7  - hplus
        0.0,   # 8  - pg6dl
        0.0,   # 9  - g6pd_nadph
        0.0,   # 10 - nadph
        0.0,   # 11 - pg6dl_g6pd
        1.0,   # 12 - lac
        0.0,   # 13 - pg6dl_lac
        0.0,   # 14 - h2o
        0.0,   # 15 - pg6_lac
        0.0,   # 16 - pg6
        1.0,   # 17 - pg6d
        0.0,   # 18 - pg6_pg6d
        0.0,   # 19 - pg6_pg6d_nadpplus
        0.0,   # 20 - pg6d_nadpplus
        0.0,   # 21 - k3pg_pg6d_nadph
        0.0,   # 22 - ribu5p_pg6d_nadph
        0.0,   # 23 - co2
        0.0,   # 24 - ribu5p
        0.0,   # 25 - pg6_nadph
        0.0,   # 26 - ribu5p_pg6d
        1.0,   # 27 - ppi
        0.0,   # 28 - ribu5p_ppi
        0.0,   # 29 - ribo5p_ppi
        0.0    # 30 - ribo5p
    ]

    interval = (0, 10000)
    p = (k1, k_1, k2, k_2, k3, k_3, k4, k_4, k5, k_5, k6, k_6, k7, k_7, k8, k_8,
         k9, k_9, k10, k_10, k11, k_11, k12, k_12, k13, k_13, k14, k_14, k15, k_15, k16,
         k_16, k17, k_17, k18, k_18, k19, k_19, k20, k_20, k21, k_21, k22, k_22, k23, k_23,
         k24, k_24, k25, k_25, fg6p, fnadpplus, fnadph, fhplus, fh2o, fco2, fribo5p, kout_g6p,
         kout_nadpplus, kout_nadph, kout_hplus, kout_h2o, kout_co2, kout_ribo5p)

    sol = solve_ivp(func, y0=yy0, t_span=interval, args=p, method='LSODA')

    print('\nIntegration Complete\n')
    print('At Steady State:')
    print('\tConsumption Rates:')
    print(
        '\t\tg6p  : {0:3.2e} units/s'.format(fg6p - kout_g6p * sol.y[0, -1]))
    print(
        '\t\tnadpplus    : {0:3.2e} units/s'.format(fnadpplus - kout_nadpplus * sol.y[3, -1]))
    print(
        '\t\th2o   : {0:3.2e} units/s'.format(fh2o - kout_h2o * sol.y[14, -1]))
    print('\tProduction Rates:')
    print('\t\tribo5p : {0:3.2e} units/s'.format(kout_ribo5p *
          sol.y[30, -1] - fribo5p))
    print('\t\tnadph : {0:3.2e} units/s'.format(kout_nadph *
          sol.y[10, -1] - fnadph))
    print(
        '\t\thplus : {0:3.2e} units/s'.format(kout_hplus * sol.y[7, -1] - fhplus))
    print(
        '\t\tco2 : {0:3.2e} units/s'.format(kout_co2 * sol.y[23, -1] - fco2))

    plt.figure(1)
    plt.plot(sol.t, sol.y[0, :], sol.t, sol.y[30, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration')
    plt.legend(['g6p', 'ribo5p'])
    plt.title('Substrate and Product')

    plt.figure(2)
    plt.plot(sol.t, sol.y[23, :], sol.t, sol.y[7, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration')
    plt.legend(['co2', 'H+'])
    plt.title("Metabolites")

    plt.figure(3)
    plt.plot(sol.t, sol.y[1, :], sol.t, sol.y[12, :], sol.t, sol.y[17, :],
             sol.t, sol.y[27, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration')
    plt.legend(['g6pd', 'lac', 'pg6d',
               'ppi'])
    plt.title("Enzymes")

    plt.figure(4)
    plt.plot(sol.t, sol.y[4, :], sol.t, sol.y[6, :],
             sol.t, sol.y[19, :], sol.t, sol.y[21, :], sol.t, sol.y[22, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration')
    plt.legend(['g6p_g6pd_nadpplus', 'pg6dl_g6pd_nadph',
               'pg6_pg6d_nadpplus', 'k3pg_pg6d_nadph', 'ribu5p_pg6d_nadph'])
    plt.title('Ternary Complexes')

    plt.figure(5)
    plt.plot(sol.t, sol.y[8, :], sol.t, sol.y[16, :], sol.t, sol.y[24, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration')
    plt.legend(['pg6dl', 'pg6', 'ribu5p'])
    plt.title('Products of First Three Steps')

    plt.figure(6)
    plt.plot(sol.t, sol.y[3, :], sol.t, sol.y[10, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration')
    plt.legend(['nadp+', 'nadph'])
    plt.title('Energy Carriers')

    plt.show()


pentose_phosphate()
