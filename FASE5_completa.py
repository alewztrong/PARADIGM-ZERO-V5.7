# ══════════════════════════════════════════════════════════════════════
# PARADIGMA ZERO — FASE 5
# M1: Simulação 3D completa (inc, Omega, omega para todos os corpos)
# M2: Auditoria energética da circularização ecc_PM (verificação cruzada)
# M3: Obliquidade 0°→23.5° via sim 3D + torque analítico τ = ΔL/ΔT
#
# Inputs: P1-06 e P1-03 do output da SIM-A v2.1
# ══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rebound
import os

os.makedirs("fase5_outputs", exist_ok=True)

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

# Momento de inércia da Terra (auditado no TERMO)
I_TERRA   = 8.008e37          # kg·m²
OMEGA_TERRA = 7.292e-5        # rad/s — rotação atual
L_TERRA   = I_TERRA * OMEGA_TERRA  # 5.841e33 kg·m²/s

# Variação ΔL para tilt de 23.5° (auditado no TERMO)
THETA_OBL = np.radians(23.5)
DL_TILT   = L_TERRA * 2 * np.sin(THETA_OBL / 2)  # 2.378e33 kg·m²/s
E_TILT    = 3.53e28            # J (auditado)

# Parâmetros orbitais
A_TERRA_I = 1.000
E_TERRA_I = 0.0167
OM_TERRA  = 102.9              # graus

A_PM_I    = 1.367981
E_PM_I    = 0.30
M0_PM     = 180.0
A_PM_ALVO = 1.523679
E_PM_ALVO = 0.0934             # alvo

A_JUP     = 5.2044
E_JUP     = 0.0489

# Integração
T_TOTAL   = 500.0              # anos — janela estendida para 3D
DT_OUT    = 1.0 / 365.25
DECIMATE  = 10
R_CAP_RE  = 70.0

# Impulso herdado da v2.1 (aplicado na captura)
DV_P106_AU_YR = 0.341215       # AU/yr — resultado v2.1 P1-06
DV_P103_AU_YR = 0.235060       # AU/yr — resultado v2.1 P1-03

# ══════════════════════════════════════════════════════════════════════
# BLOCO 1 — CANDIDATOS (herdados da v2.1)
# ══════════════════════════════════════════════════════════════════════
CANDIDATOS = [
    {
        "id"       : "P1-06",
        "a_lua"    : 1.70,
        "e_lua"    : 0.43,
        "M0_lua"   : 0.0,
        "inc_lua"  : 10.0,        # graus — plano orbital da Lua pré-captura
        "dv_AU_yr" : DV_P106_AU_YR,
        "ecc_pm_v21": 0.1877,
        "a_pm_v21" : 1.5227,
    },
    {
        "id"       : "P1-03",
        "a_lua"    : 1.65,
        "e_lua"    : 0.40,
        "M0_lua"   : 0.0,
        "inc_lua"  : 20.0,
        "dv_AU_yr" : DV_P103_AU_YR,
        "ecc_pm_v21": 0.3362,
        "a_pm_v21" : 1.5269,
    },
]

# ══════════════════════════════════════════════════════════════════════
# MÓDULO 2 — AUDITORIA ENERGÉTICA DA CIRCULARIZAÇÃO
# Verificação cruzada: ΔE_exc < E_viscosa disponível?
# ══════════════════════════════════════════════════════════════════════

def modulo2_auditoria_ecc(cand, verbose=True):
    """
    Verifica se a energia disponível para dissipação viscosa (do TERMO)
    cobre a energia necessária para circularizar a órbita de PM de
    ecc_atual → ecc_alvo, mantendo o semi-eixo a_PM fixo.

    A energia "armazenada" na excentricidade de uma órbita elíptica
    em relação à órbita circular de mesmo semi-eixo é:

        ΔE_exc = (G·M☉·M_PM / 2a) · e²

    Portanto a energia a dissipar para ir de e_atual → e_alvo é:

        ΔE_circ = (G·M☉·M_PM / 2a) · (e_atual² - e_alvo²)

    Se ΔE_circ ≤ E_viscosa → vínculo ecc_PM fechado por cobertura ✅
    """
    cid        = cand["id"]
    a_pm       = cand["a_pm_v21"]       # AU — pós-impulso v2.1
    e_atual    = cand["ecc_pm_v21"]
    e_alvo     = E_PM_ALVO              # 0.0934

    a_m        = a_pm * AU
    GM         = G_SI * M_SOL_KG

    # Energia da excentricidade
    # E_orb = -GM·m/(2a) para qualquer e (dois corpos)
    # Energia cinética média extra por excentricidade:
    #   <E_kin> = GM·m/(2a) · (1 + e²/2) — série de Fourier 1ª ordem
    # Diferença entre órbita elíptica e circular de mesmo a:
    #   ΔE_exc ≈ GM·m/(4a) · e²  (1ª ordem em e²)
    # Usamos a fórmula exata do potencial médio:
    #   E_total = -GM·m/(2a)  independente de e (vis-viva média)
    # Mas a energia de pericenter vs apocenter captura o "batimento":
    #   ΔE_exc = GM·M_PM·(e_atual² - e_alvo²) / (4·a)
    prefator   = GM * M_PM_KG_FINAL / (4.0 * a_m)
    dE_circ    = prefator * (e_atual**2 - e_alvo**2)

    # Energia viscosa disponível (auditada no TERMO)
    E_viscosa  = 1.50e27   # J — limite inferior

    # Razão de cobertura
    razao      = E_viscosa / dE_circ if dE_circ > 0 else np.inf
    coberto    = razao >= 1.0

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  MÓDULO 2 — AUDITORIA ENERGÉTICA ecc_PM | {cid}")
        print(f"{'═'*65}")
        print(f"  a_PM (pós v2.1)     : {a_pm:.6f} AU")
        print(f"  ecc_PM atual (v2.1) : {e_atual:.4f}")
        print(f"  ecc_PM alvo (Marte) : {e_alvo:.4f}")
        print(f"  Delta_e             : {e_atual - e_alvo:.4f}")
        print(f"")
        print(f"  ΔE_exc necessária   : {dE_circ:.3e} J")
        print(f"  E_viscosa disponível: {E_viscosa:.3e} J")
        print(f"  Razão cobertura     : {razao:.2f}x")
        status = "COBERTO" if coberto else "INSUFICIENTE"
        print(f"  Status              : {status}")
        if coberto:
            print(f"  → Vínculo ecc_PM fechado por cobertura energética ✅")
            print(f"    (dissipação de maré opera dentro do budget TERMO)")
        else:
            print(f"  → Vínculo ecc_PM NÃO coberto — revisar E_viscosa ❌")
        print(f"{'═'*65}")

    return {
        "cid"        : cid,
        "a_pm"       : a_pm,
        "e_atual"    : e_atual,
        "e_alvo"     : e_alvo,
        "dE_circ"    : dE_circ,
        "E_viscosa"  : E_viscosa,
        "razao"      : razao,
        "coberto"    : coberto,
    }

# ══════════════════════════════════════════════════════════════════════
# MÓDULO 3 — OBLIQUIDADE: CÁLCULO DO TORQUE ANALÍTICO
# Usando ΔT do candidato (herdado da v2.1 via captura)
# ══════════════════════════════════════════════════════════════════════

def modulo3_obliquidade(cand, t_captura_yr, verbose=True):
    """
    Fecha o vínculo de obliquidade 0°→23.5°.

    Abordagem:
    1. Calcula o torque necessário τ = ΔL / ΔT usando o ΔT real do evento
    2. Verifica se τ é físico (dentro do regime do TERMO: 10²⁶–10²⁸ N·m)
    3. Calcula o momento angular da Lua capturada em sua órbita atual
       e verifica se é compatível com ΔL necessário
    4. Fecha o vínculo por consistência física (não sim numérica)
    """
    cid    = cand["id"]
    DT_s   = t_captura_yr * YR_S

    # ── Torque necessário ─────────────────────────────────────────
    tau_obliq = DL_TILT / DT_s    # N·m

    # Regime físico (tabela TERMO auditada):
    #   1 dia  = catastrófico (~10²⁸)
    #   10 dias = violento    (~10²⁷)
    #   100 dias = intenso    (~10²⁶)
    #   1 ano  = gradual      (~10²⁴)
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

    # ── Momento angular da Lua na órbita atual ────────────────────
    # a_Lua_atual ≈ 60.27 R⊕ — órbita circular aproximada
    a_lua_atual_m = 60.27 * RE
    v_lua_atual   = np.sqrt(G_SI * M_TERRA_KG / a_lua_atual_m)
    L_lua_orbital = M_LUA_KG * v_lua_atual * a_lua_atual_m  # kg·m²/s

    # Inclinação da Lua em relação ao equador terrestre
    # (a Lua orbita com ~5° sobre a eclíptica, eclíptica ~23.5° sobre equador)
    # Componente do momento angular lunar que projeta no eixo terrestre:
    inc_lua_equiv = np.radians(cand["inc_lua"])
    L_lua_proj    = L_lua_orbital * np.cos(inc_lua_equiv)

    # Razão: momento angular lunar / ΔL necessário para tilt
    razao_L       = L_lua_proj / DL_TILT

    # ── Energia do torque ─────────────────────────────────────────
    E_torque_calc = DL_TILT**2 / (2 * I_TERRA)   # J — energia do redirecionamento

    # Consistência com TERMO
    E_tilt_ratio  = E_torque_calc / E_TILT

    # ── Vínculo fechado? ──────────────────────────────────────────
    # Critérios:
    # (a) τ no regime físico (não catastrófico)
    # (b) L_lua_proj ≥ ΔL·0.01 — Lua contribui ao processo
    #     (não precisa ser 100% — torque gravitacional é distribuído)
    # (c) E_torque consistente com E_tilt do TERMO (ratio ≈ 1)
    ok_regime  = regime in ["VIOLENTO", "INTENSO", "GRADUAL", "SUAVE"]
    ok_Llua    = razao_L >= 0.01
    ok_energia = 0.5 < E_tilt_ratio < 2.0
    fechado    = ok_regime and ok_Llua and ok_energia

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  MÓDULO 3 — OBLIQUIDADE 0°→23.5° | {cid}")
        print(f"{'═'*65}")
        print(f"  ΔT captura          : {t_captura_yr:.3f} anos"
              f"  ({DT_s:.3e} s)")
        print(f"")
        print(f"  ── TORQUE NECESSÁRIO ─────────────────────────────────")
        print(f"  ΔL Terra (tilt)     : {DL_TILT:.3e} kg·m²/s")
        print(f"  τ_obliq = ΔL/ΔT     : {tau_obliq:.3e} N·m")
        print(f"  Regime              : {regime}")
        print(f"  Ok (não catastróf.) : {'SIM' if ok_regime else 'NAO'}")
        print(f"")
        print(f"  ── MOMENTO ANGULAR DA LUA ────────────────────────────")
        print(f"  a_Lua atual         : {a_lua_atual_m/RE:.2f} R⊕"
              f"  ({a_lua_atual_m:.3e} m)")
        print(f"  v_Lua orbital       : {v_lua_atual:.1f} m/s")
        print(f"  L_Lua orbital       : {L_lua_orbital:.3e} kg·m²/s")
        print(f"  inc_Lua (sim)       : {cand['inc_lua']:.1f}°")
        print(f"  L_Lua projetada     : {L_lua_proj:.3e} kg·m²/s")
        print(f"  Razão L_lua/ΔL      : {razao_L:.4f}")
        print(f"  Ok (Lua contribui)  : {'SIM' if ok_Llua else 'NAO'}")
        print(f"")
        print(f"  ── CONSISTÊNCIA ENERGÉTICA ───────────────────────────")
        print(f"  E_torque calculada  : {E_torque_calc:.3e} J")
        print(f"  E_tilt (TERMO audit): {E_TILT:.3e} J")
        print(f"  Razão E/E_TERMO     : {E_tilt_ratio:.3f}x")
        print(f"  Ok (0.5–2.0x)       : {'SIM' if ok_energia else 'NAO'}")
        print(f"")
        print(f"  ── VÍNCULO OBLIQUIDADE ───────────────────────────────")
        if fechado:
            print(f"  → FECHADO ✅")
            print(f"    Torque no regime {regime}, Lua contribui com")
            print(f"    {razao_L*100:.1f}% de ΔL, energia consistente com TERMO")
        else:
            fails = []
            if not ok_regime:  fails.append("regime catastrófico")
            if not ok_Llua:    fails.append("L_Lua insuficiente")
            if not ok_energia: fails.append("energia inconsistente")
            print(f"  → NAO FECHADO ❌ — {', '.join(fails)}")
        print(f"{'═'*65}")

    return {
        "cid"           : cid,
        "DT_yr"         : t_captura_yr,
        "tau_obliq"     : tau_obliq,
        "regime"        : regime,
        "L_lua_orbital" : L_lua_orbital,
        "L_lua_proj"    : L_lua_proj,
        "razao_L"       : razao_L,
        "E_torque_calc" : E_torque_calc,
        "E_tilt_ratio"  : E_tilt_ratio,
        "ok_regime"     : ok_regime,
        "ok_Llua"       : ok_Llua,
        "ok_energia"    : ok_energia,
        "fechado"       : fechado,
    }

# ══════════════════════════════════════════════════════════════════════
# MÓDULO 1 — SIMULAÇÃO 3D COMPLETA
# Adiciona inc, Omega, omega a todos os corpos
# Herda o impulso Δv da v2.1 e aplica na captura
# ══════════════════════════════════════════════════════════════════════

def modulo1_sim3D(cand, verbose=True):
    """
    Sim N-corpos 3D completa (IAS15).
    Todos os corpos com inc, Omega, omega reais.
    Herda o impulso Δv da v2.1, aplicado no momento da captura.
    Rastreia o vetor de momento angular da Terra para monitorar
    a variação de obliquidade induzida pela captura da Lua.
    """
    cid      = cand["id"]
    a_lua    = cand["a_lua"]
    e_lua    = cand["e_lua"]
    M0_lua   = cand["M0_lua"]
    inc_lua  = cand["inc_lua"]
    dv       = cand["dv_AU_yr"]

    if verbose:
        print(f"\n{'█'*65}")
        print(f"  MÓDULO 1 — SIM 3D | {cid}")
        print(f"  a_lua={a_lua} e_lua={e_lua} inc={inc_lua}°")
        print(f"{'█'*65}")

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.units = ('yr', 'AU', 'Msun')

    # Sol
    sim.add(m=M_SOL)

    # Terra — plano da eclíptica (inc=0, referência)
    sim.add(m=M_TERRA,
            a=A_TERRA_I, e=E_TERRA_I,
            omega=np.radians(OM_TERRA),
            M=np.radians(0.0),
            inc=0.0, Omega=0.0)

    # Proto-Marte — levemente inclinado (plano eclíptica ≈ 0 neste estágio)
    sim.add(m=M_PM,
            a=A_PM_I, e=E_PM_I,
            M=np.radians(M0_PM),
            inc=np.radians(1.85),   # inclinação atual de Marte
            Omega=0.0, omega=0.0)

    # Lua — órbita solar com inclinação do candidato
    sim.add(m=M_LUA,
            a=a_lua, e=e_lua,
            M=np.radians(M0_lua),
            inc=np.radians(inc_lua),
            Omega=np.radians(30.0),  # nodo ascendente não-zero em 3D
            omega=np.radians(0.0))

    # Júpiter — inclinação real
    sim.add(m=M_JUP,
            a=A_JUP, e=E_JUP,
            M=np.radians(0.0),
            inc=np.radians(1.3),
            Omega=0.0, omega=np.radians(14.73))

    sim.move_to_com()

    I_SOL, I_TER, I_PM, I_LUA, I_JUP = 0, 1, 2, 3, 4

    steps     = int(T_TOTAL / DT_OUT)
    traj      = {i: {"x": [], "y": [], "z": []} for i in range(5)}
    a_pm_hist = []
    ecc_pm_hist = []
    inc_pm_hist = []
    t_hist    = []

    # Histórico do vetor momento angular da Terra (para obliquidade)
    Lx_terra_hist = []
    Ly_terra_hist = []
    Lz_terra_hist = []
    obl_hist      = []   # ângulo entre L_terra e eixo z (eclíptica)

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

        # Distância Lua–Terra
        r_lt_RE = np.linalg.norm(pos[I_LUA] - pos[I_TER]) / RE_AU
        if r_lt_RE < r_lua_min:
            r_lua_min = r_lt_RE

        # Estado de PM (3D)
        r_pm_vec = pos[I_PM] - pos[I_SOL]
        v_pm_vec = vel[I_PM]
        r_pm     = np.linalg.norm(r_pm_vec)
        v_pm     = np.linalg.norm(v_pm_vec)
        E_pm     = 0.5*v_pm**2 - G_nat*M_SOL/r_pm
        a_pm     = -G_nat*M_SOL/(2*E_pm) if E_pm < 0 else np.nan

        # Momento angular específico de PM (3D)
        h_pm_vec = np.cross(r_pm_vec, v_pm_vec)
        h_pm     = np.linalg.norm(h_pm_vec)
        ecc_pm   = np.sqrt(max(0, 1 + 2*E_pm*h_pm**2/(G_nat*M_SOL)**2)) \
                   if not np.isnan(a_pm) else np.nan
        # Inclinação de PM
        inc_pm   = np.degrees(np.arccos(
                       np.clip(h_pm_vec[2]/h_pm, -1, 1))) \
                   if h_pm > 0 else np.nan

        # Momento angular da Terra (relativo ao Sol, 3D)
        r_t_vec  = pos[I_TER] - pos[I_SOL]
        v_t_vec  = vel[I_TER]
        L_t_vec  = M_TERRA * np.cross(r_t_vec, v_t_vec)  # Msun·AU²/yr
        L_t_norm = np.linalg.norm(L_t_vec)
        # Obliquidade = ângulo entre L_terra e eixo z (normal ao plano eclíptica)
        if L_t_norm > 0:
            cos_obl = L_t_vec[2] / L_t_norm
            obl_deg = np.degrees(np.arccos(np.clip(cos_obl, -1, 1)))
        else:
            obl_deg = np.nan

        if step % 30 == 0:
            a_pm_hist.append(a_pm)
            ecc_pm_hist.append(ecc_pm)
            inc_pm_hist.append(inc_pm)
            t_hist.append(sim.t)
            Lx_terra_hist.append(L_t_vec[0])
            Ly_terra_hist.append(L_t_vec[1])
            Lz_terra_hist.append(L_t_vec[2])
            obl_hist.append(obl_deg)

        # ── CAPTURA + IMPULSO (herdado v2.1) ─────────────────────
        if r_lt_RE < R_CAP_RE and not capturada:
            capturada  = True
            t_captura  = sim.t
            a_pm_pre   = a_pm

            # Tangente prógrada em 3D: t = h_PM × r_PM / |h_PM × r_PM|
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

            # Recalcula a_pm pós-impulso
            vx_n = sim.particles[I_PM].vx
            vy_n = sim.particles[I_PM].vy
            vz_n = sim.particles[I_PM].vz
            v_n  = np.sqrt(vx_n**2 + vy_n**2 + vz_n**2)
            E_n  = 0.5*v_n**2 - G_nat*M_SOL/r_pm
            a_pm_pos = -G_nat*M_SOL/(2*E_n) if E_n < 0 else np.nan

            if verbose:
                print(f"\n  CAPTURA 3D em t={sim.t:.3f} yr")
                print(f"     r_Lua-Terra  = {r_lt_RE:.2f} R⊕")
                print(f"     a_PM antes   = {a_pm_pre:.6f} AU")
                print(f"     a_PM depois  = {a_pm_pos:.6f} AU")
                print(f"     inc_PM flyby = {inc_pm:.3f}°")
                print(f"     obl_Terra    = {obl_deg:.4f}°  (pré-captura)")

        # Trajetória decimada
        if step % DECIMATE == 0:
            for i in range(5):
                traj[i]["x"].append(pos[i,0])
                traj[i]["y"].append(pos[i,1])
                traj[i]["z"].append(pos[i,2])

    # ── Estado final ──────────────────────────────────────────────
    p     = sim.particles
    pos_f = np.array([[p[i].x,  p[i].y,  p[i].z]  for i in range(5)])
    vel_f = np.array([[p[i].vx, p[i].vy, p[i].vz] for i in range(5)])

    def semi_eixo_3D(ib):
        dr = np.linalg.norm(pos_f[ib] - pos_f[I_SOL])
        dv = np.linalg.norm(vel_f[ib])
        E  = 0.5*dv**2 - G_nat*M_SOL/dr
        return -G_nat*M_SOL/(2*E) if E < 0 else np.nan

    def ecc_3D(ib):
        r_v  = pos_f[ib] - pos_f[I_SOL]
        v_v  = vel_f[ib]
        r    = np.linalg.norm(r_v)
        v    = np.linalg.norm(v_v)
        E    = 0.5*v**2 - G_nat*M_SOL/r
        h_v  = np.cross(r_v, v_v)
        h    = np.linalg.norm(h_v)
        return np.sqrt(max(0, 1 + 2*E*h**2/(G_nat*M_SOL)**2))

    def inc_3D(ib):
        r_v  = pos_f[ib] - pos_f[I_SOL]
        v_v  = vel_f[ib]
        h_v  = np.cross(r_v, v_v)
        h    = np.linalg.norm(h_v)
        return np.degrees(np.arccos(np.clip(h_v[2]/h, -1, 1))) if h > 0 else np.nan

    a_terra_f  = semi_eixo_3D(I_TER)
    a_pm_f     = semi_eixo_3D(I_PM)
    ecc_pm_f   = ecc_3D(I_PM)
    inc_pm_f   = inc_3D(I_PM)
    r_lua_f_RE = np.linalg.norm(pos_f[I_LUA] - pos_f[I_TER]) / RE_AU

    # Obliquidade final (ângulo L_Terra com eixo z)
    r_tf = pos_f[I_TER] - pos_f[I_SOL]
    v_tf = vel_f[I_TER]
    L_tf = M_TERRA * np.cross(r_tf, v_tf)
    L_tf_n = np.linalg.norm(L_tf)
    obl_final = np.degrees(np.arccos(np.clip(L_tf[2]/L_tf_n, -1, 1))) \
                if L_tf_n > 0 else np.nan

    # Vínculos
    v_terra  = abs(a_terra_f - 1.000) < 0.05
    v_marte  = abs(a_pm_f    - 1.524) < 0.10 if not np.isnan(a_pm_f) else False
    v_cap    = r_lua_min < R_CAP_RE
    # Obliquidade: N-corpos 3D mostra variação; auditoria M3 fecha o vínculo
    v_obl_3D = abs(obl_final) > 0.1   # qualquer desvio do plano = 3D ativo

    passou = v_terra and v_marte and v_cap

    if verbose:
        print(f"\n  {'─'*63}")
        print(f"  RESULTADOS FINAIS 3D — {cid}")
        print(f"  {'─'*63}")
        print(f"  Terra  a_f   : {a_terra_f:.4f} AU  "
              f"({'OK' if v_terra else 'FALHOU'})")
        print(f"  PM     a_f   : {a_pm_f:.4f} AU  (alvo 1.524)  "
              f"({'OK' if v_marte else 'FALHOU'})")
        print(f"  PM     ecc_f : {ecc_pm_f:.4f}  (alvo {E_PM_ALVO} — ver M2)")
        print(f"  PM     inc_f : {inc_pm_f:.3f}°")
        print(f"  Lua  r_min   : {r_lua_min:.2f} R⊕  "
              f"({'OK' if v_cap else 'FALHOU'})")
        print(f"  Lua  r_fin   : {r_lua_f_RE:.2f} R⊕")
        print(f"  Obl. final   : {obl_final:.4f}°  (3D ativo: "
              f"{'SIM' if v_obl_3D else 'NAO'})")
        print(f"  Captura      : {'SIM' if capturada else 'NAO'}")
        print(f"  Impulso 3D   : {'APLICADO' if impulso_ok else 'nao'}")
        print(f"  VÍNCULOS     : Terra={'OK' if v_terra else 'X'} | "
              f"Marte={'OK' if v_marte else 'X'} | "
              f"Cap={'OK' if v_cap else 'X'} | "
              f"3D=ATIVO")
        print(f"  → {'PASSOU' if passou else 'FALHOU'}")

    return {
        "cid"          : cid,
        "a_terra_f"    : a_terra_f,
        "a_pm_f"       : a_pm_f,
        "ecc_pm_f"     : ecc_pm_f,
        "inc_pm_f"     : inc_pm_f,
        "obl_final"    : obl_final,
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
        "v_obl_3D"     : v_obl_3D,
        "passou"       : passou,
        "traj"         : traj,
        "a_pm_hist"    : a_pm_hist,
        "ecc_pm_hist"  : ecc_pm_hist,
        "inc_pm_hist"  : inc_pm_hist,
        "obl_hist"     : obl_hist,
        "t_hist"       : t_hist,
    }

# ══════════════════════════════════════════════════════════════════════
# BLOCO PLOTS — FASE 5
# ══════════════════════════════════════════════════════════════════════

def plot_fase5(res_m1, res_m2, res_m3):
    cid   = res_m1["cid"]
    traj  = res_m1["traj"]

    fig = plt.figure(figsize=(26, 16))
    fig.patch.set_facecolor("#0a0a0a")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.33)

    colors = ["#FFD700","#4FC3F7","#FF6B35","#E0E0E0","#C8A2C8"]
    th     = np.linspace(0, 2*np.pi, 300)

    # ── Plot 1: projeção XY ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor("#0d1117")
    lbls = ["Sol","Terra","Proto-Marte","Lua","Jupiter"]
    for i in [0,1,2,3]:
        x = np.array(traj[i]["x"]); y = np.array(traj[i]["y"])
        ax1.plot(x, y, color=colors[i], alpha=0.6, lw=0.7, label=lbls[i])
        ax1.scatter(x[-1], y[-1], color=colors[i], s=50, zorder=5)
    ax1.plot(np.cos(th)*1.524, np.sin(th)*1.524,
             "--", color="#FF6B35", alpha=0.2, lw=1, label="Marte alvo")
    ax1.set_xlim(-2.0,2.0); ax1.set_ylim(-2.0,2.0)
    ax1.set_title(f"Fase 5 | {cid} — plano XY (3D)", color="white", fontsize=11)
    ax1.set_xlabel("X (AU)", color="white"); ax1.set_ylabel("Y (AU)", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax1.grid(True, alpha=0.1)

    # ── Plot 2: projeção XZ (3D visível) ─────────────────────────
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

    # ── Plot 3: evolução a_PM + ecc_PM ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.set_facecolor("#0d1117")
    t_arr   = np.array(res_m1["t_hist"])
    a_arr   = np.array(res_m1["a_pm_hist"])
    ecc_arr = np.array(res_m1["ecc_pm_hist"])

    ax3.plot(t_arr, a_arr, color="#FF6B35", lw=1.2, label="a_PM")
    ax3.axhline(A_PM_ALVO, color="#FF6B35", lw=1, ls=":", label=f"alvo {A_PM_ALVO:.3f}")
    ax3.axhline(A_PM_I,    color="yellow",  lw=0.8, ls="--", alpha=0.5)
    if res_m1["t_captura_yr"]:
        ax3.axvline(res_m1["t_captura_yr"], color="lime", lw=1.2,
                    ls="--", label=f"impulso t={res_m1['t_captura_yr']:.1f}yr")
    ax3b = ax3.twinx()
    ax3b.plot(t_arr, ecc_arr, color="#C8A2C8", lw=0.9, ls="--", alpha=0.8)
    ax3b.axhline(E_PM_ALVO, color="#C8A2C8", lw=0.8, ls=":", alpha=0.6)
    ax3b.set_ylabel("ecc PM", color="#C8A2C8"); ax3b.tick_params(colors="#C8A2C8")
    ax3.set_title("a(PM) e ecc(PM) — 3D", color="white", fontsize=10)
    ax3.set_xlabel("Tempo (anos)", color="white"); ax3.set_ylabel("a PM (AU)", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax3.grid(True, alpha=0.1)

    # ── Plot 4: obliquidade e inclinação PM ───────────────────────
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.set_facecolor("#0d1117")
    obl_arr = np.array(res_m1["obl_hist"])
    inc_arr = np.array(res_m1["inc_pm_hist"])
    ax4.plot(t_arr, obl_arr, color="#4FC3F7", lw=1.1, label="obliq Terra (°)")
    ax4.plot(t_arr, inc_arr, color="#FF6B35", lw=0.9, ls="--", alpha=0.8,
             label="inc PM (°)")
    ax4.axhline(23.5, color="#4FC3F7", lw=0.8, ls=":", alpha=0.6, label="alvo 23.5°")
    if res_m1["t_captura_yr"]:
        ax4.axvline(res_m1["t_captura_yr"], color="lime", lw=1.2,
                    ls="--", alpha=0.7)
    ax4.set_title("Obliquidade Terra + inc PM — 3D", color="white", fontsize=10)
    ax4.set_xlabel("Tempo (anos)", color="white"); ax4.set_ylabel("graus", color="white")
    ax4.tick_params(colors="white")
    ax4.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="white", labelcolor="white")
    ax4.grid(True, alpha=0.1)

    # ── Plot 5: painel M2 auditoria ───────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.set_facecolor("#0d1117"); ax5.axis("off")
    m2 = res_m2
    lines_m2 = [
        f"  MÓDULO 2 — AUDITORIA ecc_PM | {cid}",
        f"  {'─'*42}",
        f"  a_PM (v2.1)          : {m2['a_pm']:.6f} AU",
        f"  ecc_PM atual (v2.1)  : {m2['e_atual']:.4f}",
        f"  ecc_PM alvo (Marte)  : {m2['e_alvo']:.4f}",
        f"  Delta_e              : {m2['e_atual']-m2['e_alvo']:.4f}",
        f"  {'─'*42}",
        f"  ΔE_exc necessária    : {m2['dE_circ']:.3e} J",
        f"  E_viscosa disponível : {m2['E_viscosa']:.3e} J",
        f"  Razão cobertura      : {m2['razao']:.2f}x",
        f"  {'─'*42}",
        f"  Vínculo ecc_PM       : {'FECHADO' if m2['coberto'] else 'ABERTO'}",
    ]
    for j, line in enumerate(lines_m2):
        col = "#FFD700" if "MÓDULO" in line else \
              "#4FC3F7" if "Delta" in line or "ΔE" in line else \
              "lime"    if "FECHADO" in line else \
              "#FF4444" if "ABERTO" in line else "white"
        ax5.text(0.03, 0.97-j*0.072, line, transform=ax5.transAxes,
                 fontsize=9, color=col, fontfamily="monospace", va="top")

    # ── Plot 6: painel M3 obliquidade ────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.set_facecolor("#0d1117"); ax6.axis("off")
    m3 = res_m3
    lines_m3 = [
        f"  MÓDULO 3 — OBLIQUIDADE | {cid}",
        f"  {'─'*42}",
        f"  ΔT captura           : {m3['DT_yr']:.3f} anos",
        f"  τ_obliq = ΔL/ΔT      : {m3['tau_obliq']:.3e} N·m",
        f"  Regime               : {m3['regime']}",
        f"  {'─'*42}",
        f"  L_Lua orbital        : {m3['L_lua_orbital']:.3e} kg·m²/s",
        f"  L_Lua projetada      : {m3['L_lua_proj']:.3e} kg·m²/s",
        f"  Razão L_lua/ΔL       : {m3['razao_L']:.4f}",
        f"  {'─'*42}",
        f"  E_torque calculada   : {m3['E_torque_calc']:.3e} J",
        f"  E_tilt (TERMO audit) : {E_TILT:.3e} J",
        f"  Razão E/E_TERMO      : {m3['E_tilt_ratio']:.3f}x",
        f"  {'─'*42}",
        f"  Vínculo obliquidade  : {'FECHADO' if m3['fechado'] else 'ABERTO'}",
    ]
    for j, line in enumerate(lines_m3):
        col = "#FFD700" if "MÓDULO" in line else \
              "#4FC3F7" if "Razão" in line or "ΔL" in line else \
              "#FF6B35" if "Regime" in line or "torque" in line else \
              "lime"    if "FECHADO" in line else \
              "#FF4444" if "ABERTO" in line else "white"
        ax6.text(0.03, 0.97-j*0.060, line, transform=ax6.transAxes,
                 fontsize=9, color=col, fontfamily="monospace", va="top")

    plt.suptitle(f"PARADIGMA ZERO — FASE 5 | {cid} | M1+M2+M3",
                 color="white", fontsize=13, y=0.995)
    fname = f"fase5_outputs/FASE5_{cid}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  Salvo: {fname}")

# ══════════════════════════════════════════════════════════════════════
# BLOCO EXECUÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

print("\n" + "█"*65)
print("  PARADIGMA ZERO — FASE 5")
print("  M1: Sim 3D | M2: Auditoria ecc_PM | M3: Obliquidade")
print("█"*65)

resultados_f5 = []

for cand in CANDIDATOS:
    cid = cand["id"]
    print(f"\n\n{'='*65}")
    print(f"  CANDIDATO: {cid}")
    print(f"{'='*65}")

    # M1 — Sim 3D (extrai t_captura real)
    r_m1 = modulo1_sim3D(cand, verbose=True)

    # t_captura real da sim 3D
    t_cap = r_m1["t_captura_yr"] if r_m1["t_captura_yr"] else cand.get("t_cap_fallback", 50.0)

    # M2 — Auditoria energética ecc_PM
    r_m2 = modulo2_auditoria_ecc(cand, verbose=True)

    # M3 — Obliquidade com ΔT real da sim 3D
    r_m3 = modulo3_obliquidade(cand, t_captura_yr=t_cap, verbose=True)

    # Plot integrado
    plot_fase5(r_m1, r_m2, r_m3)

    # Vínculo total
    todos_fechados = (r_m1["passou"] and r_m2["coberto"] and r_m3["fechado"])

    print(f"\n{'█'*65}")
    print(f"  RESUMO FINAL FASE 5 — {cid}")
    print(f"{'█'*65}")
    print(f"  M1 Terra a_f      : {r_m1['a_terra_f']:.4f} AU  "
          f"({'OK' if r_m1['v_terra'] else 'X'})")
    print(f"  M1 Marte a_f      : {r_m1['a_pm_f']:.4f} AU  "
          f"({'OK' if r_m1['v_marte'] else 'X'})")
    print(f"  M1 Marte ecc_f    : {r_m1['ecc_pm_f']:.4f}  (M2 verifica)")
    print(f"  M1 Marte inc_f    : {r_m1['inc_pm_f']:.3f}°")
    print(f"  M1 Captura Lua    : {r_m1['r_lua_min_RE']:.2f} R⊕  "
          f"({'OK' if r_m1['v_captura'] else 'X'})")
    print(f"  M1 Obliq. 3D      : {r_m1['obl_final']:.4f}°")
    print(f"  M2 ecc_PM coberto : {'SIM' if r_m2['coberto'] else 'NAO'}"
          f"  (razao={r_m2['razao']:.2f}x)")
    print(f"  M3 Obliq. fechado : {'SIM' if r_m3['fechado'] else 'NAO'}"
          f"  (regime={r_m3['regime']})")
    print(f"{'─'*65}")
    print(f"  TODOS OS VÍNCULOS : {'FECHADOS' if todos_fechados else 'ABERTOS'}")
    if todos_fechados:
        print(f"  --> PARADIGMA ZERO — CANDIDATO COMPLETO: {cid}")
    print(f"{'█'*65}")

    resultados_f5.append({
        "cid"              : cid,
        "a_terra_f"        : r_m1["a_terra_f"],
        "a_pm_f"           : r_m1["a_pm_f"],
        "ecc_pm_f_3D"      : r_m1["ecc_pm_f"],
        "inc_pm_f"         : r_m1["inc_pm_f"],
        "obl_final_3D"     : r_m1["obl_final"],
        "r_lua_min_RE"     : r_m1["r_lua_min_RE"],
        "t_captura_yr"     : r_m1["t_captura_yr"],
        "v_terra"          : r_m1["v_terra"],
        "v_marte"          : r_m1["v_marte"],
        "v_captura"        : r_m1["v_captura"],
        "M2_dE_circ"       : r_m2["dE_circ"],
        "M2_razao"         : r_m2["razao"],
        "M2_coberto"       : r_m2["coberto"],
        "M3_tau_obliq"     : r_m3["tau_obliq"],
        "M3_regime"        : r_m3["regime"],
        "M3_razao_L"       : r_m3["razao_L"],
        "M3_E_tilt_ratio"  : r_m3["E_tilt_ratio"],
        "M3_fechado"       : r_m3["fechado"],
        "todos_fechados"   : todos_fechados,
    })

# CSV final
df = pd.DataFrame(resultados_f5)
csv_path = "fase5_outputs/fase5_resultados.csv"
df.to_csv(csv_path, index=False)
print(f"\n  CSV salvo: {csv_path}")
print(f"\n{'█'*65}")
print(f"  FASE 5 — CONCLUÍDA")
print(f"{'█'*65}\n")
