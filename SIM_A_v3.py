"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PARADIGMA ZERO — SIM-A v3 + v3b                                           ║
║                                                                              ║
║  v3:  P1-03 — circularização COMPLETA (e → 0.055)                          ║
║  v3b: P1-06 — varredura k₂/Q expandida para fechar L3                      ║
║                                                                              ║
║  CORREÇÃO vs v2 [C1]:                                                       ║
║    de/dt usava sqrt(G/M_T)*M_T/M_L — dimensionalmente incorreto.           ║
║    Equação correta: de/dt = -(57/8) * k2/Q * (M_T/M_L) * (R_L/a)^5        ║
║                             * n * e * P(e)                                  ║
║    onde n = sqrt(G*M_T/a^3) — movimento médio geocêntrico.                 ║
║                                                                              ║
║  CRITÉRIO DE PARADA:                                                        ║
║    v3:  e ≤ 0.055 (circularização completa) — registra t_{e*}              ║
║    v3b: mesmo critério + varredura k₂/Q_Terra ∈ [0.025–0.150]             ║
║                                                                              ║
║  Referências:                                                                ║
║    Goldreich (1966) Rev. Geophys. 4, 411                                    ║
║    Hut (1981) A&A 99, 126                                                   ║
║    Williams et al. (2014) — calibração LLR                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv, time

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
G      = 6.674e-11
M_T    = 5.972e24
M_L    = 7.342e22
R_T    = 6.371e6
R_L    = 1.7374e6
YR_S   = 3.156e7
A_ATU  = 3.844e8      # m = 60.27 R⊕
E_ATU  = 0.0549       # excentricidade atual
E0_BP  = 12000.0      # anos — âncora temporal

# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE EXCENTRICIDADE — Hut (1981)
# ══════════════════════════════════════════════════════════════════════════════
def N(e):
    """Hut f5(e) — torque tidal para da/dt"""
    e2=e*e
    return ((1+(15/2)*e2+(45/8)*e2**2+(5/16)*e2**3)/(1-e2)**6.5)

def P(e):
    """Hut f3(e) — dissipação de excentricidade para de/dt"""
    e2=e*e
    return ((1+(15/4)*e2+(15/8)*e2**2+(5/64)*e2**3)/(1-e2)**6.5)

# ══════════════════════════════════════════════════════════════════════════════
# ODE CORRIGIDA [C1]
# ══════════════════════════════════════════════════════════════════════════════
def ode_corrected(t_yr, state, k2q_T, k2q_L):
    """
    EDO corrigida para evolução tidal da órbita geocêntrica da Lua.

    da/dt = (3 k2/Q_T) * sqrt(G M_T) * (M_L/M_T) * R_T^5 / a^5.5 * N(e)
            (Goldreich 1966, regime omega_T >> n)

    de/dt = -(57/8) * k2/Q_L * (M_T/M_L) * (R_L/a)^5 * n * e * P(e)
            (Goldreich 1966 — [C1] CORRIGIDO: usa n=sqrt(G M_T/a^3))

    Unidades: a em m, retorna [m/ano, 1/ano]
    """
    a, e = state
    if a <= R_T*1.5: return [0.0, 0.0]
    if e < 1e-8:     e = 1e-8
    if e >= 0.9999:  e = 0.9999

    n_rad = np.sqrt(G * M_T / a**3)   # rad/s — movimento médio

    # da/dt — maré na Terra eleva a Lua (regime omega_T >> n)
    da_si = (3.0 * k2q_T * np.sqrt(G * M_T)
             * M_L * R_T**5 / (M_T * a**5.5) * N(e))
    da_yr = da_si * YR_S   # m/ano

    # de/dt — maré na Lua circulariza a órbita [C1 CORRIGIDO]
    de_si = -(57.0/8.0) * k2q_L * (M_T/M_L) * (R_L/a)**5 * n_rad * e * P(e)
    de_yr = de_si * YR_S   # /ano

    # Contribuição da Terra para de/dt (menor)
    de_T_si = -(57.0/8.0) * k2q_T * (M_L/M_T) * (R_T/a)**5 * n_rad * e * P(e)
    de_yr  += de_T_si * YR_S

    return [da_yr, de_yr]

# ══════════════════════════════════════════════════════════════════════════════
# VALIDAÇÃO LLR
# ══════════════════════════════════════════════════════════════════════════════
def validate_llr(k2q_T):
    dy = ode_corrected(0, [A_ATU, E_ATU], k2q_T, 0.003)
    return dy[0] * 1000.0   # mm/ano

# ══════════════════════════════════════════════════════════════════════════════
# INTEGRAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def integrate(a0_m, e0, k2q_T, k2q_L, t_max_yr=5e5, label=""):
    """
    Integra até e ≤ E_ATU (circularização completa) OU a ≥ A_ATU.
    Registra t_{a*} (semi-eixo atual) e t_{e*} (excentricidade atual).
    """
    t_star_a = None   # quando a chega ao valor atual
    t_star_e = None   # quando e chega ao valor atual

    def ev_a(t, y, *args):
        return y[0] - A_ATU
    ev_a.terminal = False   # não para — só registra
    ev_a.direction = 1.0

    def ev_e(t, y, *args):
        return y[1] - E_ATU
    ev_e.terminal = True    # para quando e chega ao valor atual
    ev_e.direction = -1.0

    def roche(t, y, *args):
        return y[0] - 1.5*R_T
    roche.terminal = True
    roche.direction = -1.0

    t_eval = np.concatenate([
        np.linspace(0, 5000, 200),
        np.linspace(5000, 50000, 400),
        np.linspace(50000, t_max_yr, 400),
    ])
    t_eval = np.unique(t_eval)

    try:
        sol = solve_ivp(
            ode_corrected, (0, t_max_yr), [a0_m, e0],
            method='RK45', t_eval=t_eval,
            args=(k2q_T, k2q_L),
            events=[ev_a, ev_e, roche],
            rtol=1e-9, atol=[100, 1e-12],
            max_step=1000.0
        )
    except Exception as ex:
        return {"error": str(ex), "label": label}

    t_arr = sol.t
    a_arr = sol.y[0]
    e_arr = sol.y[1]

    # t_{a*}: quando a ≥ A_ATU
    if len(sol.t_events[0]) > 0:
        t_star_a = sol.t_events[0][0]
    else:
        idx_a = np.where(a_arr >= A_ATU)[0]
        t_star_a = t_arr[idx_a[0]] if len(idx_a) > 0 else None

    # t_{e*}: quando e ≤ E_ATU (evento de parada)
    if sol.status == 1 and len(sol.t_events[1]) > 0:
        t_star_e = sol.t_events[1][0]

    t_final = t_arr[-1]
    a_final = a_arr[-1]
    e_final = e_arr[-1]

    # Status para t_{a*}
    def status(t, target=E0_BP):
        if t is None: return "N/A"
        r = t / target
        if r <= 1.20: return "FECHADO ✓"
        if r <= 2.00: return "CONVERGENTE"
        if r <= 10.0: return "CONSISTENTE"
        if r <= 50.0: return "PRÓXIMO"
        return "DIVERGENTE"

    return {
        "label"       : label,
        "a0_RE"       : a0_m / R_T,
        "e0"          : e0,
        "a_final_RE"  : a_final / R_T,
        "e_final"     : e_final,
        "t_star_a_yr" : t_star_a,
        "t_star_a_kyr": t_star_a/1000 if t_star_a else None,
        "t_star_e_yr" : t_star_e,
        "t_star_e_kyr": t_star_e/1000 if t_star_e else None,
        "ratio_a"     : t_star_a/E0_BP if t_star_a else None,
        "ratio_e"     : t_star_e/E0_BP if t_star_e else None,
        "status_a"    : status(t_star_a) if t_star_a else "N/A",
        "status_e"    : status(t_star_e) if t_star_e else "NOT REACHED",
        "t_arr"       : t_arr,
        "a_arr"       : a_arr,
        "e_arr"       : e_arr,
        "error"       : None,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PARÂMETROS
# ══════════════════════════════════════════════════════════════════════════════
CANDIDATES = [
    {"id": "P1-03", "r_min_RE": 9.84,  "delta_T": 122.650},
    {"id": "P1-06", "r_min_RE": 15.51, "delta_T":  17.791},
]

def get_orbits(r_min_RE):
    r_peri = r_min_RE * R_T
    orbs = {}
    for lbl, f in [("f3",3),("f5",5),("f8",8),("f12",12),("f20",20)]:
        r_apo = min(f * r_peri, 450 * R_T)
        a = (r_peri + r_apo) / 2
        e = (r_apo - r_peri) / (r_apo + r_peri)
        orbs[lbl] = {"a_m": a, "e": e}
    return orbs

# k2/Q Terra — v3 usa grade v2; v3b adiciona valores altos para P1-06
K2Q_T_V3 = {
    "atual_Q12"  : 0.0249,
    "jovem_Q9"   : 0.0332,
    "jovem_Q7"   : 0.0427,
    "jovem_Q5"   : 0.0598,
}
K2Q_T_V3B = {   # grade expandida para P1-06
    **K2Q_T_V3,
    "turb_Q4"    : 0.0747,
    "turb_Q3"    : 0.0996,
    "turb_Q2"    : 0.1495,
}

# k2/Q Lua — inclui valores maiores para circularização
K2Q_L_V3 = {
    "quase_sol"  : 0.003,
    "semi_sol"   : 0.010,
    "transicao"  : 0.030,
    "semi_fundid": 0.080,
    "fundida"    : 0.200,
}

# ══════════════════════════════════════════════════════════════════════════════
# VARREDURA V3 — P1-03 circularização completa
# ══════════════════════════════════════════════════════════════════════════════
def run_v3():
    print(f"\n{'█'*72}")
    print(f"  SIM-A v3 — P1-03 CIRCULARIZAÇÃO COMPLETA")
    print(f"  Critério: e ≤ {E_ATU} (excentricidade atual)")
    print(f"  k2/Q_Lua expandido para [0.003, 0.200]")
    print(f"{'█'*72}")

    cand   = CANDIDATES[0]   # P1-03
    orbits = get_orbits(cand["r_min_RE"])
    results = []
    run_id = 0

    for orb_n, orb in orbits.items():
        for kt_n, k2q_t in K2Q_T_V3.items():
            for kl_n, k2q_l in K2Q_L_V3.items():
                run_id += 1
                lbl = f"P1-03|{orb_n}|T={kt_n}|L={kl_n}"
                r = integrate(orb["a_m"], orb["e"], k2q_t, k2q_l,
                               t_max_yr=1e6, label=lbl)
                results.append(r)
                if r.get("error"): continue

                t_a = f"{r['t_star_a_kyr']:.1f}" if r['t_star_a_kyr'] else "—"
                t_e = f"{r['t_star_e_kyr']:.1f}" if r['t_star_e_kyr'] else ">1Myr"
                f_a = "★" if r['status_a']=="FECHADO ✓" else " "
                f_e = "★" if r['status_e']=="FECHADO ✓" else " "

                print(f"  [{run_id:03d}] e0={orb['e']:.3f}"
                      f"  t_a*={t_a:>8s}kyr{f_a}"
                      f"  t_e*={t_e:>8s}kyr{f_e}"
                      f"  T={kt_n[:6]} L={kl_n[:8]}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# VARREDURA V3B — P1-06 k2/Q expandido
# ══════════════════════════════════════════════════════════════════════════════
def run_v3b():
    print(f"\n{'█'*72}")
    print(f"  SIM-A v3b — P1-06 VARREDURA k2/Q EXPANDIDA")
    print(f"  k2/Q_Terra até 0.150 (Terra pós-captura turbulenta)")
    print(f"  Objetivo: fechar L3 para P1-06")
    print(f"{'█'*72}")

    cand   = CANDIDATES[1]   # P1-06
    orbits = get_orbits(cand["r_min_RE"])
    results = []
    run_id = 0

    for orb_n, orb in orbits.items():
        for kt_n, k2q_t in K2Q_T_V3B.items():
            for kl_n, k2q_l in K2Q_L_V3.items():
                run_id += 1
                lbl = f"P1-06|{orb_n}|T={kt_n}|L={kl_n}"
                r = integrate(orb["a_m"], orb["e"], k2q_t, k2q_l,
                               t_max_yr=5e5, label=lbl)
                results.append(r)
                if r.get("error"): continue

                t_a = f"{r['t_star_a_kyr']:.1f}" if r['t_star_a_kyr'] else "—"
                t_e = f"{r['t_star_e_kyr']:.1f}" if r['t_star_e_kyr'] else ">500kyr"
                f_a = "★" if r['status_a']=="FECHADO ✓" else " "
                f_e = "★" if r['status_e']=="FECHADO ✓" else " "

                print(f"  [{run_id:03d}] e0={orb['e']:.3f}"
                      f"  t_a*={t_a:>8s}kyr{f_a}"
                      f"  t_e*={t_e:>8s}kyr{f_e}"
                      f"  T={kt_n[:8]} L={kl_n[:8]}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISE E RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════
def analyze_and_report(results_v3, results_v3b):
    valid_v3  = [r for r in results_v3  if not r.get("error")]
    valid_v3b = [r for r in results_v3b if not r.get("error")]

    print(f"\n{'═'*72}")
    print(f"  ANÁLISE FINAL — SIM-A v3 + v3b")
    print(f"{'═'*72}")

    for label, valid in [("P1-03 (v3 — circ. completa)", valid_v3),
                         ("P1-06 (v3b — k2/Q expandido)", valid_v3b)]:
        print(f"\n  ── {label} ──")

        # Contagem t_{a*}
        n_a_closed = sum(1 for r in valid if r.get("status_a")=="FECHADO ✓")
        n_a_conv   = sum(1 for r in valid if r.get("status_a")=="CONVERGENTE")

        # Contagem t_{e*}
        n_e_closed = sum(1 for r in valid if r.get("status_e")=="FECHADO ✓")
        n_e_conv   = sum(1 for r in valid if r.get("status_e")=="CONVERGENTE")
        n_e_reach  = sum(1 for r in valid if r.get("t_star_e_yr") is not None)

        print(f"    Runs válidos:          {len(valid)}")
        print(f"    t_{{a*}} FECHADO:      {n_a_closed}")
        print(f"    t_{{a*}} CONVERGENTE:  {n_a_conv}")
        print(f"    t_{{e*}} FECHADO:      {n_e_closed}  (circularização ≈ E0)")
        print(f"    t_{{e*}} CONVERGENTE:  {n_e_conv}")
        print(f"    e chegou a 0.055:      {n_e_reach} runs")

        # Melhores resultados
        best_a = sorted([r for r in valid if r.get("t_star_a_yr")],
                        key=lambda x: abs(x["ratio_a"]-1.0))[:3]
        best_e = sorted([r for r in valid if r.get("t_star_e_yr")],
                        key=lambda x: x["t_star_e_yr"])[:5]

        if best_a:
            print(f"\n    Melhores t_{{a*}} (ratio próx 1.0):")
            for r in best_a:
                print(f"      t_a*={r['t_star_a_kyr']:.1f}kyr"
                      f"  ratio={r['ratio_a']:.3f}  {r['status_a']}"
                      f"  [{r['label'][6:50]}]")

        if best_e:
            print(f"\n    Menores t_{{e*}} (circularização mais rápida):")
            for r in best_e:
                print(f"      t_e*={r['t_star_e_kyr']:.1f}kyr"
                      f"  ratio={r['ratio_e']:.3f}  {r['status_e']}"
                      f"  [{r['label'][6:50]}]")

    # Texto para o paper
    print(f"\n{'═'*72}")
    print(f"  TEXTO PARA SEÇÃO 5.4 DO PAPER")
    print(f"{'═'*72}")

    v3_a_closed = [r for r in valid_v3 if r.get("status_a")=="FECHADO ✓"]
    v3_e_closed = [r for r in valid_v3 if r.get("status_e")=="FECHADO ✓"]
    v3b_closed  = [r for r in valid_v3b if r.get("status_a")=="FECHADO ✓"
                   or r.get("status_e")=="FECHADO ✓"]

    best_e_p103 = sorted([r for r in valid_v3 if r.get("t_star_e_yr")],
                          key=lambda x: x["t_star_e_yr"])
    best_e_p106 = sorted([r for r in valid_v3b if r.get("t_star_e_yr")],
                          key=lambda x: x["t_star_e_yr"])

    print(f"\n  P1-03:")
    print(f"    t_{{a*}} FECHADO: {len(v3_a_closed)} configs (mesmo resultado de v2)")
    print(f"    t_{{e*}} FECHADO: {len(v3_e_closed)} configs (NOVO)")
    if best_e_p103:
        print(f"    Melhor t_{{e*}}: {best_e_p103[0]['t_star_e_kyr']:.1f} kyr")

    print(f"\n  P1-06:")
    print(f"    L3 fechada via t_{{a*}} ou t_{{e*}}: {len(v3b_closed)} configs")
    if best_e_p106:
        print(f"    Melhor t_{{e*}}: {best_e_p106[0]['t_star_e_kyr']:.1f} kyr")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def make_plots(results_v3, results_v3b, out_dir="sima_v3_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    valid_v3  = [r for r in results_v3  if not r.get("error")]
    valid_v3b = [r for r in results_v3b if not r.get("error")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0a0a0a')
    for ax in axes.flat:
        ax.set_facecolor('#111')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.tick_params(colors='#ccc', labelsize=8)

    ax_a03, ax_e03, ax_t03, ax_a06, ax_e06, ax_summ = axes.flat

    # Selecionar melhor run de cada candidato para plot
    def best_run(valid, key='t_star_e_yr'):
        runs = [r for r in valid if r.get(key)]
        return sorted(runs, key=lambda x: x[key])[0] if runs else (
               sorted([r for r in valid if r.get('t_star_a_yr')],
                      key=lambda x: abs(x['ratio_a']-1.0))[0] if any(
               r.get('t_star_a_yr') for r in valid) else None)

    best03 = best_run(valid_v3)
    best06 = best_run(valid_v3b)

    for ax, best, cid, col in [
        (ax_a03, best03, 'P1-03', '#4ECDC4'),
        (ax_a06, best06, 'P1-06', '#FF6B6B')
    ]:
        ax.set_title(f'{cid} — Semi-eixo a(t)', color='white', fontsize=10)
        if best:
            t_kyr = best['t_arr'] / 1000
            ax.semilogy(t_kyr, best['a_arr']/R_T, color=col, lw=2)
        ax.axhline(A_ATU/R_T, color='white', ls='--', lw=1, alpha=0.6,
                   label='a atual (60.3 R⊕)')
        ax.axvline(12.0, color='lime', ls=':', lw=2, alpha=0.9, label='E0=12ka')
        ax.axvspan(9.6, 14.4, alpha=0.08, color='lime')
        ax.set_xlabel('t pós-captura (kyr)', color='#ccc')
        ax.set_ylabel('a (R⊕)', color='#ccc')
        ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
        ax.grid(True, alpha=0.12)

    for ax, best, cid, col in [
        (ax_e03, best03, 'P1-03', '#4ECDC4'),
        (ax_e06, best06, 'P1-06', '#FF6B6B')
    ]:
        ax.set_title(f'{cid} — Excentricidade e(t)', color='white', fontsize=10)
        if best:
            t_kyr = best['t_arr'] / 1000
            ax.semilogy(t_kyr, np.maximum(best['e_arr'], 1e-6), color=col, lw=2)
        ax.axhline(E_ATU, color='white', ls='--', lw=1, alpha=0.6,
                   label='e atual (0.055)')
        ax.axvline(12.0, color='lime', ls=':', lw=2, alpha=0.9, label='E0=12ka')
        ax.axvspan(9.6, 14.4, alpha=0.08, color='lime')
        ax.set_xlabel('t pós-captura (kyr)', color='#ccc')
        ax.set_ylabel('e', color='#ccc')
        ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
        ax.grid(True, alpha=0.12)

    # Distribuição t_{e*}
    ax_t03.set_title('Distribuição t_{e*} — circularização completa', color='white', fontsize=10)
    te_03 = [r['t_star_e_kyr'] for r in valid_v3
             if r.get('t_star_e_kyr') and r['t_star_e_kyr'] < 490]
    te_06 = [r['t_star_e_kyr'] for r in valid_v3b
             if r.get('t_star_e_kyr') and r['t_star_e_kyr'] < 490]
    if te_03:
        ax_t03.hist(te_03, bins=20, color='#4ECDC4', alpha=0.7,
                    label='P1-03', density=True)
    if te_06:
        ax_t03.hist(te_06, bins=20, color='#FF6B6B', alpha=0.7,
                    label='P1-06', density=True)
    ax_t03.axvline(12.0, color='lime', ls='-', lw=2.5, label='E0=12ka')
    ax_t03.axvspan(9.6, 14.4, alpha=0.12, color='lime')
    ax_t03.set_xlabel('t_{e*} (kyr)', color='#ccc')
    ax_t03.set_ylabel('Densidade', color='#ccc')
    ax_t03.legend(fontsize=8, facecolor='#222', labelcolor='white')
    ax_t03.grid(True, alpha=0.12)

    # Sumário
    ax_summ.axis('off')
    v3_ne = sum(1 for r in valid_v3 if r.get('t_star_e_yr'))
    v3b_closed = sum(1 for r in valid_v3b
                     if r.get('status_a')=='FECHADO ✓' or r.get('status_e')=='FECHADO ✓')
    best_te_03 = min((r['t_star_e_kyr'] for r in valid_v3
                      if r.get('t_star_e_kyr')), default=None)
    best_te_06 = min((r['t_star_e_kyr'] for r in valid_v3b
                      if r.get('t_star_e_kyr')), default=None)

    llr = validate_llr(0.0332)
    txt = (f"SIM-A v3 + v3b — RESUMO\n{'─'*32}\n"
           f"Calibração LLR: {llr:.1f} mm/a (ref: 38.08)\n\n"
           f"P1-03 (v3):\n"
           f"  e chegou 0.055: {v3_ne} runs\n"
           f"  Menor t_e*: {f'{best_te_03:.1f} kyr' if best_te_03 else '—'}\n\n"
           f"P1-06 (v3b):\n"
           f"  L3 fechada: {v3b_closed} configs\n"
           f"  Menor t_e*: {f'{best_te_06:.1f} kyr' if best_te_06 else '—'}\n\n"
           f"Eq. de/dt: Goldreich(1966) CORRIGIDA\n"
           f"  (57/8)*(M_T/M_L)*(R_L/a)^5*n*e*P(e)")
    ax_summ.text(0.05, 0.95, txt, transform=ax_summ.transAxes,
                 fontsize=9, color='white', fontfamily='monospace',
                 va='top', bbox=dict(boxstyle='round', fc='#1a1a2e', alpha=0.8))

    plt.suptitle('SIM-A v3 + v3b — Circularização Completa + P1-06',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(out_dir, 'SIMA_v3_resultados.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(all_results, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["label","a0_RE","e0","a_final_RE","e_final",
                    "t_star_a_kyr","t_star_e_kyr",
                    "ratio_a","ratio_e","status_a","status_e"])
        for r in all_results:
            if r.get("error"): continue
            w.writerow([
                r["label"], f"{r['a0_RE']:.2f}", f"{r['e0']:.4f}",
                f"{r['a_final_RE']:.2f}", f"{r['e_final']:.6f}",
                f"{r['t_star_a_kyr']:.2f}" if r['t_star_a_kyr'] else "",
                f"{r['t_star_e_kyr']:.2f}" if r['t_star_e_kyr'] else "",
                f"{r['ratio_a']:.4f}"  if r['ratio_a'] else "",
                f"{r['ratio_e']:.4f}"  if r['ratio_e'] else "",
                r["status_a"], r["status_e"],
            ])

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    out_dir = "sima_v3_outputs"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'█'*72}")
    print(f"  SIM-A v3 + v3b — PARADIGMA ZERO")
    print(f"  Equação de/dt CORRIGIDA: (57/8)*(M_T/M_L)*(R_L/a)^5*n*e*P(e)")
    print(f"{'█'*72}\n")

    # Validação
    llr = validate_llr(0.0249)
    print(f"  Calibração LLR (k2/Q_T=0.0249): {llr:.2f} mm/ano (ref: 38.08)")
    print(f"  Erro: {abs(llr-38.08)/38.08*100:.1f}%")
    print(f"  {'✅ VÁLIDO' if abs(llr-38.08)/38.08 < 0.05 else '⚠️  VERIFICAR'}\n")

    n_v3  = len([1 for _ in get_orbits(9.84)])  * len(K2Q_T_V3)  * len(K2Q_L_V3)
    n_v3b = len([1 for _ in get_orbits(15.51)]) * len(K2Q_T_V3B) * len(K2Q_L_V3)
    print(f"  Grade v3  (P1-03): {n_v3} runs")
    print(f"  Grade v3b (P1-06): {n_v3b} runs")
    print(f"  Total: {n_v3+n_v3b} runs\n")

    t0 = time.time()
    results_v3  = run_v3()
    results_v3b = run_v3b()
    dt = time.time() - t0

    print(f"\n  Tempo total: {dt:.1f}s ({dt/60:.1f} min)")

    analyze_and_report(results_v3, results_v3b)

    fname = make_plots(results_v3, results_v3b, out_dir)
    print(f"\n  Plot: {fname}")

    all_results = results_v3 + results_v3b
    csv_path = os.path.join(out_dir, 'SIMA_v3_resumo.csv')
    save_csv(all_results, csv_path)
    print(f"  CSV:  {csv_path}")
    print("\n  SIM-A v3 + v3b concluídas.\n")

if __name__ == "__main__":
    main()
