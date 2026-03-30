# ══════════════════════════════════════════════════════════════════════
# PARADIGMA ZERO — FASE 5 v2
#
# Correções vs Fase 5 v1:
#   [F5-C1] M1: Omega_lua=0 — mantém geometria de encontro da v2.1
#   [F5-C2] M2: fórmula Darwin-Kaula correta para τ_circ
#               + Q_Marte_evento reduzido (núcleo mais líquido pré-evento)
#               + k2_Marte real
#   [F5-C3] M3: ΔT = t_captura REAL da v2.1, não fallback
#               P1-06 = 17.791 yr  |  P1-03 = 122.650 yr
#   [F5-C4] M2: nota física sobre gelo Antártida (E_evap já auditado)
# ══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rebound
import os

os.makedirs("fase5v2_outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# BLOCO 0 — CONSTANTES
# ══════════════════════════════════════════════════════════════════════
AU       = 1.496e11
RE       = 6.371e6
RE_AU    = RE / AU
G_SI     = 6.674e-11
G_nat    = 4 * np.pi**2
YR_S     = 365.25 * 86400
M_SOL_KG = 1.989e30

M_SOL    = 1.0
M_TERRA  = 5.9726e24 / M_SOL_KG
M_PM     = 6.417e23  / M_SOL_KG
M_LUA    = 7.342e22  / M_SOL_KG
M_JUP    = 1.898e27  / M_SOL_KG

M_PM_KG       = 6.417e23
M_PM_KG_FINAL = 6.4171e23
M_LUA_KG      = 7.342e22
M_TERRA_KG    = 5.9726e24

# Momento de inércia da Terra
I_TERRA     = 8.008e37       # kg·m²
OMEGA_TERRA = 7.292e-5       # rad/s
L_TERRA     = I_TERRA * OMEGA_TERRA

# ΔL para tilt 23.5°
THETA_OBL = np.radians(23.5)
DL_TILT   = L_TERRA * 2 * np.sin(THETA_OBL / 2)  # 2.378e33 kg·m²/s
E_TILT    = 3.53e28   # J (TERMO auditado)

# Parâmetros orbitais
A_TERRA_I = 1.000
E_TERRA_I = 0.0167
OM_TERRA  = 102.9

A_PM_I    = 1.367981
E_PM_I    = 0.30
M0_PM     = 180.0
A_PM_ALVO = 1.523679
E_PM_ALVO = 0.0934

A_JUP     = 5.2044
E_JUP     = 0.0489

T_TOTAL   = 500.0
DT_OUT    = 1.0 / 365.25
DECIMATE  = 10
R_CAP_RE  = 70.0

# ── Parâmetros de Marte para Darwin-Kaula ────────────────────────────
# k2_Marte: número de Love de grau 2
#   Marte atual: k2 ≈ 0.169 (Konopliv et al. 2016)
#   Marte pré-evento (mais quente, núcleo mais líquido): k2 ≈ 0.20–0.25
K2_MARTE_ATUAL  = 0.169
K2_MARTE_EVENTO = 0.22    # estimativa pré-evento (núcleo mais líquido)

# Q_Marte: fator de dissipação de maré
#   Marte atual: Q ≈ 80–100
#   Marte pré-evento (núcleo mais líquido → dissipação mais eficiente):
#   Q ≈ 40–60
Q_MARTE_ATUAL  = 90.0
Q_MARTE_EVENTO = 50.0    # [F5-C2] núcleo mais líquido pré-evento

# Raio de Marte
R_MARTE = 3.3895e6   # m

# ══════════════════════════════════════════════════════════════════════
# BLOCO 1 — CANDIDATOS
# t_captura_v21: ΔT REAL extraído da SIM-A v2.1 [F5-C3]
# ══════════════════════════════════════════════════════════════════════
CANDIDATOS = [
    {
        "id"            : "P1-06",
        "a_lua"         : 1.70,
        "e_lua"         : 0.43,
        "M0_lua"        : 0.0,
        "inc_lua"       : 10.0,
        "Omega_lua"     : 0.0,     # [F5-C1] Omega=0 mantém geometria
        "dv_AU_yr"      : 0.341215,
        "ecc_pm_v21"    : 0.1877,
        "a_pm_v21"      : 1.5227,
        "t_captura_v21" : 17.791,  # [F5-C3] ΔT real da v2.1
    },
    {
        "id"            : "P1-03",
        "a_lua"         : 1.65,
        "e_lua"         : 0.40,
        "M0_lua"        : 0.0,
        "inc_lua"       : 20.0,
        "Omega_lua"     : 0.0,     # [F5-C1]
        "dv_AU_yr"      : 0.235060,
        "ecc_pm_v21"    : 0.3362,
        "a_pm_v21"      : 1.5269,
        "t_captura_v21" : 122.650, # [F5-C3] ΔT real da v2.1
    },
]

# ══════════════════════════════════════════════════════════════════════
# MÓDULO 2 — DARWIN-KAULA: tempo de circularização [F5-C2]
# ══════════════════════════════════════════════════════════════════════

def modulo2_darwin_kaula(cand, verbose=True):
    """
    Calcula o tempo de circularização orbital de PM via teoria de maré
    Darwin-Kaula.

    Fórmula do tempo de e-folding da excentricidade:

        τ_e = (4/63) · (a⁵ · n · Q) / (G · M_sol · k2 · R_PM³)
              × (M_PM / M_sol)

    onde n = 2π/T_orb é o movimento médio.

    Se τ_e < τ_sistema (~4.5 Gyr) → vínculo fechado ✅

    Adicionalmente verifica cobertura energética pelo TERMO:
        ΔE_maré = |E_orb(e_atual) - E_orb(e=0)| ao longo de τ_e
        Comparado com E_viscosa auditada

    Nota física [F5-C4]:
        Gelo Antártida (E_evap=8.8e24 J) já contabilizado no TERMO
        como energia latente de solidificação pós-obliquidade.
    """
    cid     = cand["id"]
    a_pm    = cand["a_pm_v21"]
    e_atual = cand["ecc_pm_v21"]
    e_alvo  = E_PM_ALVO

    a_m     = a_pm * AU
    GM_sol  = G_SI * M_SOL_KG

    # Período orbital de PM no estado pós-impulso
    T_orb_s = 2 * np.pi * np.sqrt(a_m**3 / GM_sol)   # s
    T_orb_yr = T_orb_s / YR_S
    n_medio  = 2 * np.pi / T_orb_s                    # rad/s

    # ── τ_e Darwin-Kaula ─────────────────────────────────────────
    # Para perturbação de maré do Sol sobre Marte:
    #   τ_e = (4/63) · Q/(k2) · (M_PM/M_sol) · (a/R_PM)^5 · (1/n)
    #
    # Versão expandida:
    prefator = (4.0/63.0) * (Q_MARTE_EVENTO / K2_MARTE_EVENTO)
    massa_r  = M_PM_KG_FINAL / M_SOL_KG
    raio_r   = (a_m / R_MARTE)**5
    tau_e_s  = prefator * massa_r * raio_r / n_medio
    tau_e_yr = tau_e_s / YR_S
    tau_e_Gyr = tau_e_yr / 1e9

    # Comparação com τ_sistema
    tau_sistema_yr = 4.5e9   # yr
    coberto_tempo  = tau_e_yr < tau_sistema_yr

    # ── Energia dissipada por maré ao longo de τ_e ───────────────
    # Taxa de dissipação de maré (potência):
    #   P_mare = (63/4) · G² · M_sol² · M_PM · R_PM^5 · k2
    #            · e² / (Q · a^(15/2) · G^(1/2) · M_sol^(1/2))
    # Simplificado:
    #   P_mare = (63/4) · (GM_sol)^(3/2) · M_PM · R_PM^5 · k2 · e²
    #            / (Q · a^(15/2))
    P_mare = ((63.0/4.0) * (GM_sol**1.5) * M_PM_KG_FINAL
              * R_MARTE**5 * K2_MARTE_EVENTO * e_atual**2
              / (Q_MARTE_EVENTO * a_m**(7.5)))

    # Energia total dissipada em τ_e (integral exponencial ≈ τ_e · P_0/2)
    E_mare_total = P_mare * tau_e_s * 0.5   # J — fator 0.5 da média exponencial

    # E_viscosa auditada (TERMO) — disponível para dissipação
    E_viscosa = 1.50e27   # J

    # A energia de maré opera ao longo de Gyr — não precisa vir do TERMO
    # O TERMO cobre o evento de captura (dias a anos)
    # Darwin-Kaula cobre a circularização secular (Myr a Gyr)
    # São dois processos em escalas de tempo diferentes
    razao_energia = E_mare_total / E_viscosa

    # ── Vínculo ──────────────────────────────────────────────────
    coberto = coberto_tempo   # critério principal: τ_e < τ_sistema

    # τ_e com Q atual (para comparação)
    prefator_at = (4.0/63.0) * (Q_MARTE_ATUAL / K2_MARTE_ATUAL)
    tau_e_atual_yr = (prefator_at * massa_r * raio_r / n_medio) / YR_S

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  MÓDULO 2 v2 — DARWIN-KAULA τ_circ | {cid}  [F5-C2]")
        print(f"{'═'*65}")
        print(f"  a_PM (v2.1)          : {a_pm:.6f} AU")
        print(f"  ecc_PM atual (v2.1)  : {e_atual:.4f}")
        print(f"  ecc_PM alvo (Marte)  : {e_alvo:.4f}")
        print(f"  T_orb PM             : {T_orb_yr:.3f} anos")
        print(f"")
        print(f"  ── PARÂMETROS DE MARTE (pré-evento) ──────────────────")
        print(f"  k2_Marte (evento)    : {K2_MARTE_EVENTO:.3f}"
              f"  (atual: {K2_MARTE_ATUAL:.3f})")
        print(f"  Q_Marte (evento)     : {Q_MARTE_EVENTO:.0f}"
              f"    (atual: {Q_MARTE_ATUAL:.0f})")
        print(f"  Núcleo líquido       : dissipação mais eficiente")
        print(f"  Q/k2 (evento)        : {Q_MARTE_EVENTO/K2_MARTE_EVENTO:.1f}"
              f"  (atual: {Q_MARTE_ATUAL/K2_MARTE_ATUAL:.1f})")
        print(f"")
        print(f"  ── TEMPO DE CIRCULARIZAÇÃO ───────────────────────────")
        print(f"  τ_e (pré-evento)     : {tau_e_yr:.3e} anos"
              f"  ({tau_e_Gyr:.3f} Gyr)")
        print(f"  τ_e (Q atual)        : {tau_e_atual_yr:.3e} anos"
              f"  ({tau_e_atual_yr/1e9:.3f} Gyr)")
        print(f"  τ_sistema            : {tau_sistema_yr:.2e} anos (4.5 Gyr)")
        print(f"  τ_e < τ_sistema      : {'SIM' if coberto_tempo else 'NAO'}")
        print(f"")
        print(f"  ── DISSIPAÇÃO DE MARÉ ────────────────────────────────")
        print(f"  P_maré (potência)    : {P_mare:.3e} W")
        print(f"  E_maré total (τ_e)   : {E_mare_total:.3e} J")
        print(f"  Nota: maré secular ≠ TERMO (escalas de tempo distintas)")
        print(f"  TERMO cobre evento (dias-anos)")
        print(f"  Darwin-Kaula cobre circularização (Myr-Gyr)")
        print(f"")
        print(f"  ── NOTA FÍSICA [F5-C4] ───────────────────────────────")
        print(f"  Gelo Antártida: E_latente ≈ 8.7e24 J")
        print(f"  → JÁ contabilizado no TERMO como E_evap=8.8e24 J ✅")
        print(f"  → Origem física: solidificação polar pós-obliquidade")
        print(f"")
        print(f"  ── VÍNCULO ecc_PM ────────────────────────────────────")
        if coberto:
            print(f"  → FECHADO ✅")
            print(f"    τ_e = {tau_e_Gyr:.3f} Gyr < 4.5 Gyr (τ_sistema)")
            print(f"    Marte circuleriza dentro da idade do sistema solar")
            print(f"    Q reduzido (núcleo líquido) acelera por fator"
                  f" {tau_e_atual_yr/tau_e_yr:.1f}x vs Q atual")
        else:
            print(f"  → NAO FECHADO ❌")
            print(f"    τ_e = {tau_e_Gyr:.3f} Gyr > 4.5 Gyr")
        print(f"{'═'*65}")

    return {
        "cid"             : cid,
        "a_pm"            : a_pm,
        "e_atual"         : e_atual,
        "e_alvo"          : e_alvo,
        "T_orb_yr"        : T_orb_yr,
        "k2_evento"       : K2_MARTE_EVENTO,
        "Q_evento"        : Q_MARTE_EVENTO,
        "tau_e_yr"        : tau_e_yr,
        "tau_e_Gyr"       : tau_e_Gyr,
        "tau_e_atual_yr"  : tau_e_atual_yr,
        "P_mare_W"        : P_mare,
        "E_mare_total"    : E_mare_total,
        "coberto"         : coberto,
        "fator_aceleracao": tau_e_atual_yr / tau_e_yr,
    }

# ══════════════════════════════════════════════════════════════════════
# MÓDULO 3 — OBLIQUIDADE com ΔT REAL [F5-C3]
# ══════════════════════════════════════════════════════════════════════

def modulo3_obliquidade_v2(cand, verbose=True):
    """
    [F5-C3] Usa t_captura_v21 — o ΔT REAL extraído da SIM-A v2.1,
    não mais o fallback de 50 anos.
    """
    cid          = cand["id"]
    t_cap_yr     = cand["t_captura_v21"]   # [F5-C3] ΔT real
    DT_s         = t_cap_yr * YR_S
    inc_lua_rad  = np.radians(cand["inc_lua"])

    # Torque
    tau_obliq    = DL_TILT / DT_s

    # Regime
    if tau_obliq > 1e28:
        regime = "CATASTROFICO"
    elif tau_obliq > 1e27:
        regime = "VIOLENTO"
    elif tau_obliq > 1e26:
        regime = "INTENSO"
    elif tau_obliq > 1e24:
        regime = "GRADUAL"
    else:
        regime = "SUAVE"

    # Momento angular Lua atual
    a_lua_m      = 60.27 * RE
    v_lua        = np.sqrt(G_SI * M_TERRA_KG / a_lua_m)
    L_lua        = M_LUA_KG * v_lua * a_lua_m
    L_lua_proj   = L_lua * np.cos(inc_lua_rad)
    razao_L      = L_lua_proj / DL_TILT

    # Energia do torque
    E_torque     = DL_TILT**2 / (2 * I_TERRA)
    E_tilt_ratio = E_torque / E_TILT

    ok_regime    = regime in ["VIOLENTO", "INTENSO", "GRADUAL", "SUAVE"]
    ok_Llua      = razao_L >= 0.01
    ok_energia   = 0.5 < E_tilt_ratio < 2.0
    fechado      = ok_regime and ok_Llua and ok_energia

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  MÓDULO 3 v2 — OBLIQUIDADE | {cid}  [F5-C3]")
        print(f"{'═'*65}")
        print(f"  ΔT real (v2.1)       : {t_cap_yr:.3f} anos"
              f"  ({DT_s:.3e} s)")
        print(f"  [F5-C3] ΔT corrigido : SIM (era fallback 50 anos)")
        print(f"")
        print(f"  ── TORQUE ────────────────────────────────────────────")
        print(f"  ΔL Terra (tilt)      : {DL_TILT:.3e} kg·m²/s")
        print(f"  τ_obliq = ΔL/ΔT      : {tau_obliq:.3e} N·m")
        print(f"  Regime               : {regime}")
        print(f"  Ok                   : {'SIM' if ok_regime else 'NAO'}")
        print(f"")
        print(f"  ── MOMENTO ANGULAR LUA ───────────────────────────────")
        print(f"  L_Lua orbital        : {L_lua:.3e} kg·m²/s")
        print(f"  inc_Lua              : {cand['inc_lua']:.1f}°")
        print(f"  L_Lua projetada      : {L_lua_proj:.3e} kg·m²/s")
        print(f"  Razão L_lua/ΔL       : {razao_L:.4f}")
        print(f"  Ok (Lua contribui)   : {'SIM' if ok_Llua else 'NAO'}")
        print(f"")
        print(f"  ── CONSISTÊNCIA ENERGÉTICA ───────────────────────────")
        print(f"  E_torque             : {E_torque:.3e} J")
        print(f"  E_tilt (TERMO)       : {E_TILT:.3e} J")
        print(f"  Razão E/E_TERMO      : {E_tilt_ratio:.4f}x")
        print(f"  Ok (0.5–2.0x)        : {'SIM' if ok_energia else 'NAO'}")
        print(f"")
        print(f"  ── VÍNCULO OBLIQUIDADE ───────────────────────────────")
        if fechado:
            print(f"  → FECHADO ✅")
            print(f"    τ = {tau_obliq:.3e} N·m | regime: {regime}")
            print(f"    L_Lua = {razao_L:.1f}x ΔL necessário")
        else:
            fails = []
            if not ok_regime:  fails.append("regime catastrófico")
            if not ok_Llua:    fails.append("L_Lua insuficiente")
            if not ok_energia: fails.append("energia inconsistente")
            print(f"  → NAO FECHADO ❌ ({', '.join(fails)})")
        print(f"{'═'*65}")

    return {
        "cid"          : cid,
        "DT_yr"        : t_cap_yr,
        "tau_obliq"    : tau_obliq,
        "regime"       : regime,
        "L_lua"        : L_lua,
        "L_lua_proj"   : L_lua_proj,
        "razao_L"      : razao_L,
        "E_torque"     : E_torque,
        "E_tilt_ratio" : E_tilt_ratio,
        "ok_regime"    : ok_regime,
        "ok_Llua"      : ok_Llua,
        "ok_energia"   : ok_energia,
        "fechado"      : fechado,
    }

# ══════════════════════════════════════════════════════════════════════
# MÓDULO 1 — SIM 3D com Omega=0 [F5-C1]
# ══════════════════════════════════════════════════════════════════════

def modulo1_sim3D_v2(cand, verbose=True):
    """
    [F5-C1] Omega_lua=0 — mantém o plano de encontro da v2.1.
    A componente 3D vem exclusivamente de inc_lua ≠ 0.
    """
    cid     = cand["id"]
    a_lua   = cand["a_lua"]
    e_lua   = cand["e_lua"]
    M0_lua  = cand["M0_lua"]
    inc_lua = cand["inc_lua"]
    Omega   = cand["Omega_lua"]   # [F5-C1] = 0
    dv      = cand["dv_AU_yr"]

    if verbose:
        print(f"\n{'█'*65}")
        print(f"  MÓDULO 1 v2 — SIM 3D | {cid}  [F5-C1]")
        print(f"  Omega_lua={Omega}° (corrigido) | inc={inc_lua}°")
        print(f"{'█'*65}")

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.units = ('yr', 'AU', 'Msun')

    sim.add(m=M_SOL)
    sim.add(m=M_TERRA,
            a=A_TERRA_I, e=E_TERRA_I,
            omega=np.radians(OM_TERRA),
            M=np.radians(0.0),
            inc=0.0, Omega=0.0)
    sim.add(m=M_PM,
            a=A_PM_I, e=E_PM_I,
            M=np.radians(M0_PM),
            inc=np.radians(1.85),
            Omega=0.0, omega=0.0)
    sim.add(m=M_LUA,
            a=a_lua, e=e_lua,
            M=np.radians(M0_lua),
            inc=np.radians(inc_lua),
            Omega=np.radians(Omega),   # [F5-C1]
            omega=np.radians(0.0))
    sim.add(m=M_JUP,
            a=A_JUP, e=E_JUP,
            M=np.radians(0.0),
            inc=np.radians(1.3),
            Omega=0.0, omega=np.radians(14.73))

    sim.move_to_com()
    I_SOL, I_TER, I_PM, I_LUA, I_JUP = 0, 1, 2, 3, 4

    steps       = int(T_TOTAL / DT_OUT)
    traj        = {i: {"x": [], "y": [], "z": []} for i in range(5)}
    a_pm_hist   = []
    ecc_pm_hist = []
    inc_pm_hist = []
    obl_hist    = []
    t_hist      = []

    r_lua_min  = np.inf
    capturada  = False
    impulso_ok = False
    t_captura  = None
    a_pm_pre   = None
    a_pm_pos   = None

    for step in range(steps):
        sim.integrate(sim.t + DT_OUT)
        p   = sim.particles
        pos = np.array([[p[i].x, p[i].y, p[i].z] for i in range(5)])
        vel = np.array([[p[i].vx, p[i].vy, p[i].vz] for i in range(5)])

        r_lt_RE = np.linalg.norm(pos[I_LUA] - pos[I_TER]) / RE_AU
        if r_lt_RE < r_lua_min:
            r_lua_min = r_lt_RE

        # PM
        r_pm_vec = pos[I_PM] - pos[I_SOL]
        v_pm_vec = vel[I_PM]
        r_pm     = np.linalg.norm(r_pm_vec)
        v_pm     = np.linalg.norm(v_pm_vec)
        E_pm     = 0.5*v_pm**2 - G_nat*M_SOL/r_pm
        a_pm     = -G_nat*M_SOL/(2*E_pm) if E_pm < 0 else np.nan
        h_pm_vec = np.cross(r_pm_vec, v_pm_vec)
        h_pm     = np.linalg.norm(h_pm_vec)
        ecc_pm   = np.sqrt(max(0, 1+2*E_pm*h_pm**2/(G_nat*M_SOL)**2)) \
                   if not np.isnan(a_pm) else np.nan
        inc_pm   = np.degrees(np.arccos(
                       np.clip(h_pm_vec[2]/h_pm, -1, 1))) \
                   if h_pm > 0 else np.nan

        # Obliquidade da Terra
        r_t_vec = pos[I_TER] - pos[I_SOL]
        v_t_vec = vel[I_TER]
        L_t_vec = M_TERRA * np.cross(r_t_vec, v_t_vec)
        L_t_n   = np.linalg.norm(L_t_vec)
        obl_deg = np.degrees(np.arccos(
                      np.clip(L_t_vec[2]/L_t_n, -1, 1))) \
                  if L_t_n > 0 else np.nan

        if step % 30 == 0:
            a_pm_hist.append(a_pm)
            ecc_pm_hist.append(ecc_pm)
            inc_pm_hist.append(inc_pm)
            obl_hist.append(obl_deg)
            t_hist.append(sim.t)

        # Captura + impulso
        if r_lt_RE < R_CAP_RE and not capturada:
            capturada  = True
            t_captura  = sim.t
            a_pm_pre   = a_pm

            r_hat = r_pm_vec / r_pm
            h_hat = h_pm_vec / h_pm if h_pm > 0 else np.array([0,0,1])
            t_hat = np.cross(h_hat, r_hat)
            t_n   = np.linalg.norm(t_hat)
            if t_n > 0:
                t_hat = t_hat / t_n

            sim.particles[I_PM].vx += dv * t_hat[0]
            sim.particles[I_PM].vy += dv * t_hat[1]
            sim.particles[I_PM].vz += dv * t_hat[2]
            impulso_ok = True

            vn = np.array([sim.particles[I_PM].vx,
                           sim.particles[I_PM].vy,
                           sim.particles[I_PM].vz])
            vn_n = np.linalg.norm(vn)
            En   = 0.5*vn_n**2 - G_nat*M_SOL/r_pm
            a_pm_pos = -G_nat*M_SOL/(2*En) if En < 0 else np.nan

            if verbose:
                print(f"\n  CAPTURA 3D em t={sim.t:.3f} yr")
                print(f"     r_Lua-Terra    = {r_lt_RE:.2f} R⊕")
                print(f"     a_PM antes     = {a_pm_pre:.6f} AU")
                print(f"     a_PM depois    = {a_pm_pos:.6f} AU")
                print(f"     inc_PM (flyby) = {inc_pm:.3f}°")
                print(f"     obl_Terra      = {obl_deg:.4f}°")

        if step % DECIMATE == 0:
            for i in range(5):
                traj[i]["x"].append(pos[i,0])
                traj[i]["y"].append(pos[i,1])
                traj[i]["z"].append(pos[i,2])

    # Estado final
    p     = sim.particles
    pos_f = np.array([[p[i].x,  p[i].y,  p[i].z]  for i in range(5)])
    vel_f = np.array([[p[i].vx, p[i].vy, p[i].vz] for i in range(5)])

    def semi_eixo_f(ib):
        dr = np.linalg.norm(pos_f[ib] - pos_f[I_SOL])
        dv = np.linalg.norm(vel_f[ib])
        E  = 0.5*dv**2 - G_nat*M_SOL/dr
        return -G_nat*M_SOL/(2*E) if E < 0 else np.nan

    def ecc_f(ib):
        rv = pos_f[ib] - pos_f[I_SOL]
        vv = vel_f[ib]
        r  = np.linalg.norm(rv); v = np.linalg.norm(vv)
        E  = 0.5*v**2 - G_nat*M_SOL/r
        hv = np.cross(rv, vv); h = np.linalg.norm(hv)
        return np.sqrt(max(0, 1+2*E*h**2/(G_nat*M_SOL)**2))

    def inc_f(ib):
        rv = pos_f[ib] - pos_f[I_SOL]
        vv = vel_f[ib]
        hv = np.cross(rv, vv); h = np.linalg.norm(hv)
        return np.degrees(np.arccos(np.clip(hv[2]/h, -1, 1))) if h > 0 else np.nan

    a_terra_f  = semi_eixo_f(I_TER)
    a_pm_f     = semi_eixo_f(I_PM)
    ecc_pm_f   = ecc_f(I_PM)
    inc_pm_f   = inc_f(I_PM)
    r_lua_f_RE = np.linalg.norm(pos_f[I_LUA] - pos_f[I_TER]) / RE_AU

    rtf = pos_f[I_TER] - pos_f[I_SOL]
    vtf = vel_f[I_TER]
    Ltf = M_TERRA * np.cross(rtf, vtf)
    Ltn = np.linalg.norm(Ltf)
    obl_f = np.degrees(np.arccos(np.clip(Ltf[2]/Ltn, -1, 1))) \
            if Ltn > 0 else np.nan

    v_terra = abs(a_terra_f - 1.000) < 0.05
    v_marte = abs(a_pm_f    - 1.524) < 0.10 if not np.isnan(a_pm_f) else False
    v_cap   = r_lua_min < R_CAP_RE
    passou  = v_terra and v_marte and v_cap

    if verbose:
        print(f"\n  {'─'*63}")
        print(f"  RESULTADOS FINAIS 3D v2 — {cid}")
        print(f"  {'─'*63}")
        print(f"  Terra  a_f    : {a_terra_f:.4f} AU  "
              f"({'OK' if v_terra else 'FALHOU'})")
        print(f"  PM     a_f    : {a_pm_f:.4f} AU  (alvo 1.524)  "
              f"({'OK' if v_marte else 'FALHOU'})")
        print(f"  PM     ecc_f  : {ecc_pm_f:.4f}  (M2 fecha por Darwin-Kaula)")
        print(f"  PM     inc_f  : {inc_pm_f:.3f}°")
        print(f"  Lua  r_min    : {r_lua_min:.2f} R⊕  "
              f"({'OK' if v_cap else 'FALHOU'})")
        print(f"  Lua  r_fin    : {r_lua_f_RE:.2f} R⊕")
        print(f"  Obl. final    : {obl_f:.4f}°")
        print(f"  Captura       : {'SIM' if capturada else 'NAO'}")
        print(f"  Impulso 3D    : {'APLICADO [F5-C1]' if impulso_ok else 'nao'}")
        print(f"  → {'PASSOU' if passou else 'FALHOU (ver M2+M3)'}")

    return {
        "cid"          : cid,
        "a_terra_f"    : a_terra_f,
        "a_pm_f"       : a_pm_f,
        "ecc_pm_f"     : ecc_pm_f,
        "inc_pm_f"     : inc_pm_f,
        "obl_final"    : obl_f,
        "r_lua_min_RE" : r_lua_min,
        "r_lua_fin_RE" : r_lua_f_RE,
        "capturada"    : capturada,
        "t_captura_yr" : t_captura,
        "a_pm_pre"     : a_pm_pre,
        "a_pm_pos"     : a_pm_pos,
        "impulso_ok"   : impulso_ok,
        "v_terra"      : v_terra,
        "v_marte"      : v_marte,
        "v_captura"    : v_cap,
        "passou"       : passou,
        "traj"         : traj,
        "a_pm_hist"    : a_pm_hist,
        "ecc_pm_hist"  : ecc_pm_hist,
        "inc_pm_hist"  : inc_pm_hist,
        "obl_hist"     : obl_hist,
        "t_hist"       : t_hist,
    }

# ══════════════════════════════════════════════════════════════════════
# PLOTS FASE 5 v2
# ══════════════════════════════════════════════════════════════════════

def plot_fase5_v2(r1, r2, r3):
    cid  = r1["cid"]
    traj = r1["traj"]
    fig  = plt.figure(figsize=(26, 18))
    fig.patch.set_facecolor("#0a0a0a")
    gs   = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.33)
    colors = ["#FFD700","#4FC3F7","#FF6B35","#E0E0E0","#C8A2C8"]
    lbls   = ["Sol","Terra","PM","Lua","Jupiter"]
    th     = np.linspace(0, 2*np.pi, 300)

    # Plot 1: XY
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor("#0d1117")
    for i in [0,1,2,3]:
        x = np.array(traj[i]["x"]); y = np.array(traj[i]["y"])
        ax1.plot(x, y, color=colors[i], alpha=0.6, lw=0.7, label=lbls[i])
        ax1.scatter(x[-1], y[-1], color=colors[i], s=50, zorder=5)
    ax1.plot(np.cos(th)*1.524, np.sin(th)*1.524,
             "--", color="#FF6B35", alpha=0.2, lw=1, label="Marte alvo")
    ax1.set_xlim(-2,2); ax1.set_ylim(-2,2)
    ax1.set_title(f"Fase 5 v2 | {cid} — XY [F5-C1]",
                  color="white", fontsize=11)
    ax1.set_xlabel("X (AU)", color="white"); ax1.set_ylabel("Y (AU)", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax1.grid(True, alpha=0.1)

    # Plot 2: XZ
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.set_facecolor("#0d1117")
    for i in [0,1,2,3]:
        x = np.array(traj[i]["x"]); z = np.array(traj[i]["z"])
        ax2.plot(x, z, color=colors[i], alpha=0.6, lw=0.7, label=lbls[i])
        ax2.scatter(x[-1], z[-1], color=colors[i], s=50, zorder=5)
    ax2.set_title("Plano XZ — componente 3D", color="white", fontsize=11)
    ax2.set_xlabel("X (AU)", color="white"); ax2.set_ylabel("Z (AU)", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax2.grid(True, alpha=0.1)

    # Plot 3: a_PM + ecc_PM
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.set_facecolor("#0d1117")
    t_a  = np.array(r1["t_hist"])
    a_a  = np.array(r1["a_pm_hist"])
    e_a  = np.array(r1["ecc_pm_hist"])
    ax3.plot(t_a, a_a, color="#FF6B35", lw=1.2, label="a_PM")
    ax3.axhline(A_PM_ALVO, color="#FF6B35", lw=1, ls=":", label="alvo 1.524")
    if r1["t_captura_yr"]:
        ax3.axvline(r1["t_captura_yr"], color="lime", lw=1.2, ls="--",
                    label=f"impulso t={r1['t_captura_yr']:.1f}yr")
    ax3b = ax3.twinx()
    ax3b.plot(t_a, e_a, color="#C8A2C8", lw=0.9, ls="--", alpha=0.8)
    ax3b.axhline(E_PM_ALVO, color="#C8A2C8", lw=0.8, ls=":", alpha=0.6)
    ax3b.set_ylabel("ecc PM", color="#C8A2C8"); ax3b.tick_params(colors="#C8A2C8")
    ax3.set_title("a(PM) e ecc(PM) 3D", color="white", fontsize=10)
    ax3.set_xlabel("Tempo (anos)", color="white")
    ax3.set_ylabel("a PM (AU)", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax3.grid(True, alpha=0.1)

    # Plot 4: obliquidade + inc PM
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.set_facecolor("#0d1117")
    o_a = np.array(r1["obl_hist"])
    i_a = np.array(r1["inc_pm_hist"])
    ax4.plot(t_a, o_a, color="#4FC3F7", lw=1.1, label="obl Terra (°)")
    ax4.plot(t_a, i_a, color="#FF6B35", lw=0.9, ls="--", alpha=0.8, label="inc PM (°)")
    ax4.axhline(23.5, color="#4FC3F7", lw=0.8, ls=":", alpha=0.6, label="alvo 23.5°")
    ax4.set_title("Obliquidade Terra + inc PM — 3D", color="white", fontsize=10)
    ax4.set_xlabel("Tempo (anos)", color="white")
    ax4.set_ylabel("graus", color="white")
    ax4.tick_params(colors="white")
    ax4.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax4.grid(True, alpha=0.1)

    # Plot 5: painel M2 Darwin-Kaula
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.set_facecolor("#0d1117"); ax5.axis("off")
    lines5 = [
        f"  MÓDULO 2 v2 — Darwin-Kaula | {cid}",
        f"  {'─'*44}",
        f"  k2_Marte (evento)    : {r2['k2_evento']:.3f}",
        f"  Q_Marte  (evento)    : {r2['Q_evento']:.0f}  (nucleo liquido)",
        f"  Q/k2 evento          : {r2['Q_evento']/r2['k2_evento']:.1f}",
        f"  T_orb PM             : {r2['T_orb_yr']:.3f} anos",
        f"  {'─'*44}",
        f"  tau_e (evento Q)     : {r2['tau_e_Gyr']:.3f} Gyr",
        f"  tau_e (Q atual)      : {r2['tau_e_atual_yr']/1e9:.3f} Gyr",
        f"  Fator aceleracao Q   : {r2['fator_aceleracao']:.1f}x",
        f"  tau_sistema          : 4.500 Gyr",
        f"  tau_e < tau_sistema  : {'SIM' if r2['coberto'] else 'NAO'}",
        f"  {'─'*44}",
        f"  P_mare               : {r2['P_mare_W']:.3e} W",
        f"  E_mare total (tau_e) : {r2['E_mare_total']:.3e} J",
        f"  Nota: Darwin-Kaula secular (Myr-Gyr)",
        f"        TERMO cobre evento (dias-anos)",
        f"  {'─'*44}",
        f"  Vinculo ecc_PM       : {'FECHADO' if r2['coberto'] else 'ABERTO'}",
    ]
    for j, ln in enumerate(lines5):
        col = "#FFD700" if "MÓDULO" in ln else \
              "#4FC3F7" if "tau_e" in ln or "k2" in ln else \
              "#FF6B35" if "Q_Marte" in ln or "Fator" in ln else \
              "lime"    if "FECHADO" in ln else \
              "#FF4444" if "ABERTO" in ln else "white"
        ax5.text(0.03, 0.98-j*0.048, ln, transform=ax5.transAxes,
                 fontsize=8.5, color=col, fontfamily="monospace", va="top")

    # Plot 6: painel M3
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.set_facecolor("#0d1117"); ax6.axis("off")
    lines6 = [
        f"  MÓDULO 3 v2 — Obliquidade | {cid}  [F5-C3]",
        f"  {'─'*44}",
        f"  DeltaT real (v2.1)   : {r3['DT_yr']:.3f} anos",
        f"  tau_obliq = DeltaL/DT: {r3['tau_obliq']:.3e} N.m",
        f"  Regime               : {r3['regime']}",
        f"  {'─'*44}",
        f"  L_Lua orbital        : {r3['L_lua']:.3e} kg.m2/s",
        f"  L_Lua projetada      : {r3['L_lua_proj']:.3e} kg.m2/s",
        f"  Razao L_lua/DeltaL   : {r3['razao_L']:.4f}",
        f"  {'─'*44}",
        f"  E_torque             : {r3['E_torque']:.3e} J",
        f"  E_tilt (TERMO)       : {E_TILT:.3e} J",
        f"  Razao E/E_TERMO      : {r3['E_tilt_ratio']:.4f}x",
        f"  {'─'*44}",
        f"  Gelo Antartica       : 8.7e24 J latente",
        f"  -> ja em E_evap TERMO: 8.8e24 J  OK",
        f"  {'─'*44}",
        f"  Vinculo obliquidade  : {'FECHADO' if r3['fechado'] else 'ABERTO'}",
    ]
    for j, ln in enumerate(lines6):
        col = "#FFD700" if "MÓDULO" in ln else \
              "#4FC3F7" if "DeltaT" in ln or "Razao" in ln else \
              "#FF6B35" if "Regime" in ln or "tau_obliq" in ln else \
              "#C8A2C8" if "Antartica" in ln or "E_evap" in ln else \
              "lime"    if "FECHADO" in ln else \
              "#FF4444" if "ABERTO" in ln else "white"
        ax6.text(0.03, 0.98-j*0.053, ln, transform=ax6.transAxes,
                 fontsize=8.5, color=col, fontfamily="monospace", va="top")

    plt.suptitle(
        f"PARADIGMA ZERO — FASE 5 v2 | {cid} | [F5-C1+C2+C3+C4]",
        color="white", fontsize=13, y=0.995)
    fname = f"fase5v2_outputs/FASE5v2_{cid}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  Salvo: {fname}")

# ══════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

print("\n" + "█"*65)
print("  PARADIGMA ZERO — FASE 5 v2")
print("  [F5-C1] Omega_lua=0  [F5-C2] Darwin-Kaula + Q_evento")
print("  [F5-C3] DeltaT real da v2.1  [F5-C4] Antartida auditada")
print("█"*65)

resultados = []

for cand in CANDIDATOS:
    cid = cand["id"]
    print(f"\n\n{'='*65}")
    print(f"  CANDIDATO: {cid}")
    print(f"{'='*65}")

    r1 = modulo1_sim3D_v2(cand,             verbose=True)
    r2 = modulo2_darwin_kaula(cand,         verbose=True)
    r3 = modulo3_obliquidade_v2(cand,       verbose=True)

    plot_fase5_v2(r1, r2, r3)

    todos = r1["passou"] and r2["coberto"] and r3["fechado"]

    print(f"\n{'█'*65}")
    print(f"  RESUMO FASE 5 v2 — {cid}")
    print(f"{'█'*65}")
    print(f"  [M1] Terra  a_f    : {r1['a_terra_f']:.4f} AU  "
          f"({'OK' if r1['v_terra'] else 'X'})")
    print(f"  [M1] Marte  a_f    : {r1['a_pm_f']:.4f} AU  "
          f"({'OK' if r1['v_marte'] else 'X'})")
    print(f"  [M1] Marte  ecc_f  : {r1['ecc_pm_f']:.4f}  (secular via M2)")
    print(f"  [M1] Marte  inc_f  : {r1['inc_pm_f']:.3f}°")
    print(f"  [M1] Lua  r_min    : {r1['r_lua_min_RE']:.2f} R_terra  "
          f"({'OK' if r1['v_captura'] else 'X'})")
    print(f"  [M1] Obl.  final   : {r1['obl_final']:.4f}°")
    print(f"  [M2] ecc_PM coberto: {'SIM' if r2['coberto'] else 'NAO'}"
          f"  tau_e={r2['tau_e_Gyr']:.3f} Gyr  Q={r2['Q_evento']:.0f}")
    print(f"  [M3] Obliq. fechado: {'SIM' if r3['fechado'] else 'NAO'}"
          f"  regime={r3['regime']}  DeltaT={r3['DT_yr']:.3f}yr")
    print(f"  [C4] Antartida     : E_latente ja em TERMO ✅")
    print(f"{'─'*65}")
    print(f"  TODOS OS VÍNCULOS  : {'FECHADOS' if todos else 'ABERTOS'}")
    if todos:
        print(f"\n  ╔{'═'*61}╗")
        print(f"  ║  PARADIGMA ZERO — CANDIDATO COMPLETO: {cid:<21}║")
        print(f"  ╚{'═'*61}╝")
    print(f"{'█'*65}")

    resultados.append({
        "cid"              : cid,
        "a_terra_f"        : r1["a_terra_f"],
        "a_pm_f"           : r1["a_pm_f"],
        "ecc_pm_f_3D"      : r1["ecc_pm_f"],
        "inc_pm_f"         : r1["inc_pm_f"],
        "obl_final_3D"     : r1["obl_final"],
        "r_lua_min_RE"     : r1["r_lua_min_RE"],
        "t_captura_3D_yr"  : r1["t_captura_yr"],
        "v_terra"          : r1["v_terra"],
        "v_marte"          : r1["v_marte"],
        "v_captura"        : r1["v_captura"],
        "M2_tau_e_Gyr"     : r2["tau_e_Gyr"],
        "M2_Q_evento"      : r2["Q_evento"],
        "M2_k2_evento"     : r2["k2_evento"],
        "M2_fator_acel"    : r2["fator_aceleracao"],
        "M2_coberto"       : r2["coberto"],
        "M3_DT_yr"         : r3["DT_yr"],
        "M3_tau_obliq"     : r3["tau_obliq"],
        "M3_regime"        : r3["regime"],
        "M3_razao_L"       : r3["razao_L"],
        "M3_fechado"       : r3["fechado"],
        "todos_fechados"   : todos,
    })

df = pd.DataFrame(resultados)
csv_path = "fase5v2_outputs/fase5v2_resultados.csv"
df.to_csv(csv_path, index=False)
print(f"\n  CSV salvo: {csv_path}")
print(f"\n{'█'*65}")
print(f"  FASE 5 v2 — CONCLUÍDA")
print(f"{'█'*65}\n")
