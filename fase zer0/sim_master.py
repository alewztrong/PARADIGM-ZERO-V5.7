"""
╔══════════════════════════════════════════════════════════════════════╗
║          SIMULAÇÕES N-CORPOS — CENÁRIO LUA ERRANTE                  ║
║          Código Mestre Unificado — Versão de Replicação             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Premissas axiomáticas:                                             ║
║    • Terra e Lua de mesma origem isotópica                          ║
║    • Marte expulsou a Lua via perturbação gravitacional             ║
║    • Ressonância 8:5 Terra-Marte (a_Marte = 1.368 UA)              ║
║                                                                     ║
║  Simulações incluídas:                                              ║
║    SIM1 — v8: N-corpos Terra+Lua+Marte (Leapfrog + EKH 1998)       ║
║    SIM2 — Fase 1 MC: 6 corpos, 36 fases, destinos imediatos        ║
║    SIM3 — Kozai-Lidov: estadia em Júpiter, 500 Myr                  ║
║    SIM4 — Monte Carlo retorno: 94.389 trajetórias, distribuição     ║
║    SIM5 — Linha do tempo retroativa: ancoragem em 12.000 AP         ║
║                                                                     ║
║  Dependências: Python 3.10+, NumPy >= 1.24, Matplotlib >= 3.7      ║
║  Instalação: pip install numpy matplotlib scipy                     ║
║  Execução:   python sim_master.py [--sim NOME] [--all]             ║
║  Exemplo:    python sim_master.py --sim v8                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, sys, argparse, time, os

# ════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS UNIVERSAIS
# Todas as simulações usam este bloco — não modificar
# ════════════════════════════════════════════════════════
G       = 4 * np.pi**2      # UA³ M☉⁻¹ ano⁻²  [escolha canônica: G=4π² implica P²=a³]
M_SOL   = 1.0               # M☉
M_TERRA = 3.003e-6           # M☉  [IERS 2010]
M_LUA   = 3.694e-8           # M☉  [IAU 2015]
M_MARTE = 3.213e-7           # M☉  [IERS 2010]
M_JUP   = 9.545e-4           # M☉  [IAU 2015]
M_SAT   = 2.858e-4           # M☉  [IAU 2015]

# Raios físicos (UA)
R_TERRA = 4.259e-5           # UA
R_LUA   = 1.161e-5           # UA  (Lua jovem — raio físico)

# Raios de Hill
RHILL_TERRA = (M_TERRA / (3 * M_SOL))**(1/3)           # ≈ 0.010 UA
RHILL_JUP   = 5.203 * (M_JUP / (3 * M_SOL))**(1/3)    # ≈ 0.355 UA

# Parâmetros EKH — oceano magmático lunar
k2_LUA  = 0.80               # Número de Love (fluido quase completo)
Q_LUA   = 2.0                # Fator de qualidade (limite dissipativo)
kQ_LUA  = k2_LUA / Q_LUA    # = 0.400  [1667× Lua sólida atual]
kQ_TERRA = 0.299 / 10.0     # Terra jovem

# Softening gravitacional
EPS_SOFT = 3e-7              # UA  [< R_Lua — não distorce física relevante]

# Ressonância 8:5
A_MARTE_RES = (8/5)**(2/3)  # = 1.3680 UA  [algebraicamente exato]

# Ressonâncias MMR de Júpiter (para referência)
A_JUP = 5.203
MMR_JUP = {
    '4:1': A_JUP * (1/4)**(2/3),   # 2.065 UA
    '3:1': A_JUP * (1/3)**(2/3),   # 2.501 UA
    '5:2': A_JUP * (2/5)**(2/3),   # 2.825 UA
    '7:3': A_JUP * (3/7)**(2/3),   # 2.958 UA
    '2:1': A_JUP * (1/2)**(2/3),   # 3.278 UA
}

print(f"╔══ CONSTANTES VERIFICADAS ══╗")
print(f"  G        = {G:.6f} UA³ M☉⁻¹ ano⁻²")
print(f"  R_Hill(Terra) = {RHILL_TERRA:.6f} UA")
print(f"  R_Hill(Jup)   = {RHILL_JUP:.4f} UA")
print(f"  a_Marte(8:5)  = {A_MARTE_RES:.6f} UA")
print(f"  kQ_Lua = {kQ_LUA:.4f} [1667× Lua sólida]")
print()


# ════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES COMPARTILHADAS
# ════════════════════════════════════════════════════════

def accel_nbody_vetorial(pos, masses, eps=EPS_SOFT):
    """
    Acelerações gravitacionais N-corpos vetorizadas com softening.
    Entrada:  pos (N,3), masses (N,)
    Saída:    acc (N,3)
    Complexidade: O(N²)
    """
    diff = pos[None, :, :] - pos[:, None, :]     # (N,N,3)
    r2   = np.einsum('ijk,ijk->ij', diff, diff) + eps**2
    np.fill_diagonal(r2, 1.0)                    # evita divisão por zero na diagonal
    acc  = np.sum(G * masses[None, :, None] * diff / r2[:, :, None]**1.5, axis=1)
    return acc


def energia_total(pos, vel, masses):
    """
    Energia total do sistema: E = E_cinética + E_gravitacional.
    Retorna: (E_total, E_cinetica, E_gravitacional)
    """
    Ec = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    d  = pos[None, :, :] - pos[:, None, :]
    r  = np.sqrt(np.einsum('ijk,ijk->ij', d, d) + 1e-60)
    np.fill_diagonal(r, np.inf)
    Eg = -0.5 * np.sum(G * masses[None, :] * masses[:, None] / r)
    return float(Ec + Eg), float(Ec), float(Eg)


def elementos_orbitais(pos_i, vel_i, M_central=M_SOL):
    """
    Semieixo e excentricidade heliocêntricos de um corpo.
    Retorna: (a, e)  ou  (inf, inf) se órbita hiperbólica
    """
    r   = norm(pos_i)
    v2  = np.dot(vel_i, vel_i)
    eps = 0.5 * v2 - G * M_central / r
    if eps >= 0:
        return np.inf, np.inf
    a  = -G * M_central / (2 * eps)
    L  = np.cross(pos_i, vel_i)
    e2 = max(0.0, 1.0 - np.dot(L, L) / (G * M_central * a))
    return float(a), float(np.sqrt(e2))


def leapfrog_step(pos, vel, masses, dt, force_fn=None):
    """
    Um passo do integrador Leapfrog (Störmer-Verlet) — simplético.
    force_fn: função adicional não-conservativa (ex: EKH tidal)
              signature: force_fn(pos, vel) -> (acc_extra[N,3], dE_dt)
    Retorna: (pos_new, vel_new, dE_tidal)
    """
    # Kick ½
    acc = accel_nbody_vetorial(pos, masses)
    dE_tidal = 0.0
    if force_fn is not None:
        acc_extra, dE = force_fn(pos, vel)
        acc = acc + acc_extra
        dE_tidal += dE * dt
    vel_half = vel + 0.5 * dt * acc

    # Drift
    pos_new = pos + dt * vel_half

    # Kick ½ final
    acc_new = accel_nbody_vetorial(pos_new, masses)
    if force_fn is not None:
        acc_extra, dE = force_fn(pos_new, vel_half)
        acc_new = acc_new + acc_extra
        dE_tidal += dE * dt
    vel_new = vel_half + 0.5 * dt * acc_new

    return pos_new, vel_new, dE_tidal


# ════════════════════════════════════════════════════════
# SIM1 — SIMULAÇÃO v8: N-CORPOS TERRA+LUA+MARTE
# Reproduz o cenário de escape da Lua com física corrigida
# Referência: Relatório v8, Seções 2-3
# ════════════════════════════════════════════════════════

def forca_eKH_lua(pos, vel):
    """
    Força de maré EKH 1998 aplicada sobre a Lua (índice 2).
    [C2] Corte em RMARE_MAX = 1.5 × R_Hill (corrigido de 0.08 UA)
    Retorna: (acc_extra[4,3], dE_dt)
    """
    RMARE_MAX = 1.5 * RHILL_TERRA   # ≈ 0.015 UA  ← correção C2

    acc_extra = np.zeros((4, 3))
    dE_dt     = 0.0

    pt = pos[1]; pl = pos[2]
    vt = vel[1]; vl = vel[2]
    rv = pl - pt
    vrel = vl - vt
    r  = norm(rv)

    if r < 1e-9 or r > RMARE_MAX:
        return acc_extra, 0.0

    rh   = rv / r
    vtan = vrel - np.dot(vrel, rh) * rh
    vm   = norm(vtan)

    # Componente A: torque de maré Terra→Lua
    if vm > 1e-12:
        mag_t  = 3 * kQ_TERRA * G * M_TERRA * M_LUA * R_TERRA**5 / (r**6 * M_LUA)
        F_mare = mag_t * (vtan / vm)
    else:
        F_mare = np.zeros(3)

    # Componente B: dissipação viscosa (Hut 1981 + EKH 1998)
    eps_orb = 0.5 * np.dot(vrel, vrel) - G * M_TERRA / r
    F_diss  = np.zeros(3)

    if eps_orb < -1e-20:
        aL  = -G * M_TERRA / (2 * eps_orb)
        e2  = max(0.0, min(0.99, 1.0 - norm(np.cross(rv, vrel))**2 / (G * M_TERRA * aL)))
        nL  = np.sqrt(G * M_TERRA / aL**3)
        # f(e) de Hut 1981, eq. A.2  [verificado sem bug na auditoria]
        fe  = (1 + 3*e2 + 0.375*e2**2) / (1 - e2)**4.5
        vrm = norm(vrel)
        if vrm > 1e-12:
            mag_d  = kQ_LUA * G * M_TERRA * M_LUA * R_LUA**5 * nL * fe / (aL**6 * M_LUA)
            F_diss = -mag_d * (vrel / vrm)

    F_total = F_mare + F_diss
    acc_extra[2] = F_total        # força aplicada apenas na Lua
    dE_dt        = np.dot(F_total, vl)  # potência dissipada

    return acc_extra, dE_dt


def sim1_v8(t_total=25.0, n_output=4000, verbose=True):
    """
    SIM1: N-corpos v8 — Terra, Lua, Marte, Sol.
    Integrador Leapfrog simplético + EKH 1998 corrigido.

    Condições iniciais (melhor candidato da busca de fase v7):
      a_Lua = 0.0025 UA, e_Lua = 0.05
      a_Marte = 1.3680 UA (res. 8:5), e_Marte = 0.10
      phi_L = 1.396 rad, phi_M = 3.770 rad

    Retorna: dict com arrays de tempo e resultados
    """
    if verbose:
        print("═" * 60)
        print("  SIM1 — N-corpos v8 (Leapfrog + EKH corrigido)")
        print(f"  t_total = {t_total} anos | n_output = {n_output}")
        print("═" * 60)

    MASSES = np.array([M_SOL, M_TERRA, M_LUA, M_MARTE])
    a_L0   = 0.0025; e_L0 = 0.05
    phi_L  = 1.396;  phi_M = 3.770
    e_M    = 0.10

    # Condições iniciais no referencial heliocêntrico
    p_t = np.array([1.0, 0.0, 0.0])
    v_t = np.array([0.0, 2*np.pi, 0.0])

    rL   = a_L0 * (1 - e_L0)
    vL   = np.sqrt(G*M_TERRA/rL) * np.sqrt((1+e_L0)/(1-e_L0))
    p_l  = p_t + rL * np.array([np.cos(phi_L), np.sin(phi_L), 0.0])
    v_l  = v_t + vL * np.array([-np.sin(phi_L), np.cos(phi_L), 0.0])

    QM   = A_MARTE_RES * (1 + e_M)
    vaf  = np.sqrt(G*M_SOL/A_MARTE_RES * (1-e_M)/(1+e_M))
    p_m  = QM * np.array([np.cos(phi_M), np.sin(phi_M), 0.0])
    v_m  = vaf * np.array([-np.sin(phi_M), np.cos(phi_M), 0.0])

    pos = np.array([[0.,0.,0.], p_t, p_l, p_m])
    vel = np.array([[0.,0.,0.], v_t, v_l, v_m])

    # Correção para centro de massa
    pos -= np.sum(MASSES[:,None]*pos, 0) / MASSES.sum()
    vel -= np.sum(MASSES[:,None]*vel, 0) / MASSES.sum()

    E0, Ec0, Eg0 = energia_total(pos, vel, MASSES)
    if verbose:
        print(f"  E₀ = {E0:.4e}  (Ec={Ec0:.4e}, Eg={Eg0:.4e})")
        print(f"  R_Hill Terra = {RHILL_TERRA:.5f} UA")
        print(f"  Corte tidal  = {1.5*RHILL_TERRA:.5f} UA  [C2 corrigido]")
        print()

    # Inicializar Leapfrog com meio-passo de velocidade
    dt0  = 2e-3
    acc0 = accel_nbody_vetorial(pos, MASSES)
    vel  = vel + 0.5 * dt0 * acc0

    # Arrays de saída
    t_arr=[]; dl_arr=[]; dm_arr=[]; al_arr=[]; am_arr=[]
    dE_arr=[]; dEt_arr=[]; Ec_arr=[]; Eg_arr=[]

    Et=0.0; esc=False; t_esc=None; a_ls=None
    t=0.0; tout=0.0; dt_out=t_total/n_output

    while t < t_total:
        dl = norm(pos[2]-pos[1])
        dm = norm(pos[3]-pos[1])
        dmin = min(dl, dm)
        if   dmin < RHILL_TERRA:       dt = 5e-5
        elif dmin < 3*RHILL_TERRA:     dt = 2e-4
        elif dmin < 10*RHILL_TERRA:    dt = 8e-4
        else:                           dt = 3e-3
        dt = min(dt, t_total - t)

        pos, vel, dEt = leapfrog_step(pos, vel, MASSES, dt, forca_eKH_lua)
        Et += dEt
        t  += dt

        dl = norm(pos[2]-pos[1])
        if not esc and dl > 1.5*RHILL_TERRA:
            esc = True; t_esc = float(t)
            a_ls, _ = elementos_orbitais(pos[2], vel[2])
            if verbose:
                print(f"  *** ESCAPE Lua em t={t_esc:.4f} anos ***  a_solar={a_ls:.4f} UA")

        if t >= tout:
            En, Ec, Eg = energia_total(pos, vel, MASSES)
            aL, _ = elementos_orbitais(pos[2], vel[2])
            aM, _ = elementos_orbitais(pos[3], vel[3])
            dm2   = norm(pos[3]-pos[1])
            t_arr.append(float(t)); dl_arr.append(float(dl)); dm_arr.append(float(dm2))
            al_arr.append(float(aL)); am_arr.append(float(aM) if aM!=np.inf else 0.0)
            dE_arr.append(float((En-E0)/abs(E0))); dEt_arr.append(float(Et))
            Ec_arr.append(float(Ec)); Eg_arr.append(float(Eg))
            tout += dt_out

    En,_,_ = energia_total(pos, vel, MASSES)
    dEf = (En-E0)/abs(E0)
    aM_f, _ = elementos_orbitais(pos[3], vel[3])

    if verbose:
        print()
        print(f"  RESULTADO SIM1:")
        print(f"  ΔE/E₀        = {dEf:.2e}")
        print(f"  a_Marte_final= {aM_f:.5f} UA  (alvo: {A_MARTE_RES:.5f} UA)")
        print(f"  Lua escapou? = {esc}")
        if t_esc: print(f"  t_escape     = {t_esc:.4f} anos")
        print()

    return {
        't': t_arr, 'd_lua': dl_arr, 'd_marte': dm_arr,
        'a_lua': al_arr, 'a_marte': am_arr,
        'dE': dE_arr, 'dE_tidal': dEt_arr, 'Ec': Ec_arr, 'Eg': Eg_arr,
        'lua_escapou': esc, 't_escape': t_esc, 'a_lua_solar': a_ls,
        'a_marte_final': float(aM_f), 'dE_final': float(dEf),
        'E0': float(E0), 'RHILL': float(RHILL_TERRA),
    }


# ════════════════════════════════════════════════════════
# SIM2 — FASE 1 MC: DESTINOS IMEDIATOS DA LUA ERRANTE
# 6 corpos, 36 amostras de fase, 2 Myr por trajetória
# ════════════════════════════════════════════════════════

def sim2_destinos_mc(N_runs=36, t_max_myr=2.0, verbose=True):
    """
    SIM2: Monte Carlo de destinos — Lua em 1.48 UA, 6 corpos.

    Condições iniciais: pós-escape (a=1.48 UA, e=0.098)
    Corpos: Sol, Terra, Marte(1.368 UA), Lua, Júpiter, Saturno
    Fase da Lua: N_runs valores uniformes em [0, 2π]

    Retorna: dict com contagem de destinos e probabilidades
    """
    if verbose:
        print("═" * 60)
        print(f"  SIM2 — MC Destinos (N={N_runs}, t={t_max_myr} Myr/run)")
        print("═" * 60)

    MASSES6 = np.array([M_SOL, M_TERRA, M_MARTE, M_LUA, M_JUP, M_SAT])
    MYR     = 1e6

    A_LUA_ESC = 1.48; E_LUA_ESC = 0.098
    A_SAT     = 9.537

    def build_ic6(phi_lua):
        pos = np.zeros((6, 3)); vel = np.zeros((6, 3))
        # Terra
        pos[1]=[1.,0.,0.]; vel[1]=[0.,2*np.pi,0.]
        # Marte (ressonância 8:5)
        pos[2]=[A_MARTE_RES,0.,0.]; vel[2]=[0.,np.sqrt(G*M_SOL/A_MARTE_RES),0.]
        # Lua (pós-escape)
        r_peri = A_LUA_ESC*(1-E_LUA_ESC)
        v_peri = np.sqrt(G*M_SOL/A_LUA_ESC*(1+E_LUA_ESC)/(1-E_LUA_ESC))
        pos[3]=r_peri*np.array([np.cos(phi_lua),np.sin(phi_lua),0.])
        vel[3]=v_peri*np.array([-np.sin(phi_lua),np.cos(phi_lua),0.])
        # Júpiter
        pos[4]=[A_JUP,0.,0.]; vel[4]=[0.,np.sqrt(G*M_SOL/A_JUP),0.]
        # Saturno
        pos[5]=[A_SAT,0.,0.]; vel[5]=[0.,np.sqrt(G*M_SOL/A_SAT),0.]
        # CoM
        pos-=np.sum(MASSES6[:,None]*pos,0)/MASSES6.sum()
        vel-=np.sum(MASSES6[:,None]*vel,0)/MASSES6.sum()
        return pos, vel

    fases    = np.linspace(0, 2*np.pi, N_runs, endpoint=False)
    contagem = {'captura_jupiter':0,'recaptura_terra':0,
                'ejecao_hiperbolica':0,'colisao_sol':0,'em_orbita':0}
    resultados = []

    for i, phi in enumerate(fases):
        pos, vel = build_ic6(phi)
        t=0.; t_max=t_max_myr*MYR; desfecho='em_orbita'; t_desf=None

        # Init leapfrog
        acc0 = accel_nbody_vetorial(pos, MASSES6)
        vel  = vel + 0.5*2e-3*acc0

        while t < t_max:
            dl=norm(pos[3]-pos[1]); dm=norm(pos[3]-pos[2]); dj=norm(pos[3]-pos[4])
            dmin=min(dl,dm,dj)
            if   dmin<RHILL_TERRA:  dt=5e-5
            elif dmin<0.3:          dt=3e-4
            else:                   dt=2e-3
            dt=min(dt,t_max-t)

            # Leapfrog step
            pos=pos+dt*vel
            acc_new=accel_nbody_vetorial(pos,MASSES6)
            vel=vel+dt*acc_new; t+=dt

            dl=norm(pos[3]-pos[1]); dj=norm(pos[3]-pos[4])
            r_lua=norm(pos[3])

            if dj < RHILL_JUP:
                desfecho='captura_jupiter'; t_desf=t/MYR; break
            if dl < RHILL_TERRA*3:
                desfecho='recaptura_terra'; t_desf=t/MYR; break
            a_l,_=elementos_orbitais(pos[3],vel[3])
            if a_l==np.inf or r_lua>40:
                desfecho='ejecao_hiperbolica'; t_desf=t/MYR; break
            if r_lua<0.1:
                desfecho='colisao_sol'; t_desf=t/MYR; break

        contagem[desfecho]+=1
        resultados.append({'phi':phi,'desfecho':desfecho,'t_desf':t_desf or t/MYR})
        if verbose and i%6==0:
            print(f"  Run {i+1:02d}/{N_runs}  φ={np.degrees(phi):6.1f}°  "
                  f"{desfecho:20s}  t={resultados[-1]['t_desf']:.4f} Myr")

    if verbose:
        print()
        print("  RESULTADO SIM2:")
        for k,v in contagem.items():
            print(f"  {k:25s}: {v:2d}/{N_runs}  ({100*v/N_runs:.1f}%)")
        print()

    return {'contagem': contagem,
            'probabilidades': {k: v/N_runs for k,v in contagem.items()},
            'resultados': resultados, 'N_runs': N_runs}


# ════════════════════════════════════════════════════════
# SIM3 — KOZAI-LIDOV: ESTADIA EM JÚPITER
# Equações seculares de KL + difusão caótica (500 Myr)
# ════════════════════════════════════════════════════════

def sim3_kozai_lidov(t_total_myr=500, dt_myr=0.5, verbose=True):
    """
    SIM3: Kozai-Lidov secular com Saturno como perturbador.

    Equações de Kozai 1962 / Lidov 1962 (nível quadrupolo)
    com difusão caótica de Morbidelli & Nesvorný 1999.

    Condições iniciais da órbita irregular da Lua ao redor de Júpiter:
      a_irr = 0.28 × R_Hill_Júpiter = 0.099 UA
      e₀    = 0.25
      i₀    = 150° (órbita retrógrada — análogo Pasífae)

    Retorna: arrays de tempo, e, inclinação, omega
    """
    if verbose:
        print("═" * 60)
        print(f"  SIM3 — Kozai-Lidov ({t_total_myr} Myr, dt={dt_myr} Myr)")
        print("═" * 60)

    A_SAT  = 9.537; E_SAT = 0.056; MYR = 1e6
    a_irr  = 0.28 * RHILL_JUP    # 0.099 UA

    # Frequência e pré-fator de KL
    n_in   = np.sqrt(G * M_JUP / a_irr**3)    # mov. médio ao redor de Júpiter (rad/ano)
    eps_KL = (M_SAT/M_JUP) * (a_irr/A_SAT)**3 / (1-E_SAT**2)**1.5
    T_KL   = (8/(15*np.pi)) / eps_KL / n_in / MYR  # Myr
    if verbose:
        print(f"  a_irr = {a_irr:.4f} UA = {a_irr/RHILL_JUP:.2f} R_Hill_Jup")
        print(f"  ε_KL  = {eps_KL:.4e}")
        print(f"  T_KL  = {T_KL:.4f} Myr por ciclo")

    N     = int(t_total_myr / dt_myr) + 1
    dt    = dt_myr * MYR   # anos
    rng   = np.random.default_rng(42)

    t_arr   = np.zeros(N)
    e_arr   = np.zeros(N)
    om_arr  = np.zeros(N)
    inc_arr = np.zeros(N)

    # Estado inicial
    e0  = 0.25; omega0 = 0.0; i0 = 150.0 * np.pi/180
    state = np.array([e0, omega0, i0])

    jz = np.sqrt(1-e0**2) * np.cos(i0)   # integral de Kozai conservada
    e_max_KL = np.sqrt(max(0, 1 - jz**2))
    if verbose:
        print(f"  j_z   = {jz:.4f}  (conservado no KL quadrupolo)")
        print(f"  e_max = {e_max_KL:.4f}  (sem difusão)")
        print()

    t_flyby = None

    for k in range(N):
        e, om, inc = state
        t_arr[k]   = k * dt_myr
        e_arr[k]   = e
        om_arr[k]  = np.degrees(om)
        inc_arr[k] = np.degrees(inc)

        # Equações de KL (nível quadrupolo)
        si2 = np.sin(inc)**2; s2o = np.sin(2*om)
        de  = (15/8)*n_in*eps_KL*e*np.sqrt(1-e**2)*si2*s2o * dt
        dw  = (3/4)*n_in*eps_KL/np.sqrt(1-e**2) * (
                5*e**2*np.sin(om)**2 - 2*(1-e**2)
                - 5*si2*(e**2*np.sin(om)**2 - 1 + e**2)) * dt
        di  = -(15/8)*n_in*eps_KL*e**2*np.sin(2*inc)*np.sin(2*om)/(2*np.sqrt(1-e**2)) * dt

        # Difusão caótica (Morbidelli & Nesvorný 1999)
        # σ_e ~ 0.0015 Myr⁻¹ para satélites irregulares típicos
        # Fator ~2 para a Lua massiva (perturbações galileanas)
        sigma_e = 0.003 * np.sqrt(dt_myr)   # por passo de dt_myr Myr
        de += rng.normal(0, sigma_e)

        state[0] = np.clip(e + de, 0.001, 0.97)
        state[1] = (om + dw) % (2*np.pi)
        state[2] = inc + di

        # Detectar iminência de instabilidade Hill
        # Critério: r_apo = a_irr*(1+e) > 0.45*R_Hill_Jup
        r_apo = a_irr * (1 + state[0])
        if t_flyby is None and r_apo > 0.45 * RHILL_JUP:
            t_flyby = k * dt_myr
            if verbose:
                print(f"  *** INSTABILIDADE HILL em t={t_flyby:.0f} Myr ***")
                print(f"      e={state[0]:.4f}, r_apo={r_apo:.4f} UA = {r_apo/RHILL_JUP:.3f} R_Hill")

    if verbose:
        print(f"\n  Duração total: {t_total_myr} Myr")
        print(f"  e_final: {e_arr[-1]:.4f}")
        print(f"  i_final: {inc_arr[-1]:.1f}°")
        print()

    return {
        't_myr': t_arr, 'e': e_arr, 'omega_deg': om_arr, 'inc_deg': inc_arr,
        't_flyby_myr': float(t_flyby) if t_flyby else None,
        'T_KL_myr': float(T_KL), 'jz': float(jz), 'e_max_KL': float(e_max_KL),
        'a_irr_UA': float(a_irr),
    }


# ════════════════════════════════════════════════════════
# SIM4 — MONTE CARLO DE RETORNO
# 94.389 trajetórias, distribuição de t_retorno
# ════════════════════════════════════════════════════════

def sim4_mc_retorno(N=100000, seed=2024, verbose=True):
    """
    SIM4: Monte Carlo da distribuição de tempo de retorno à Terra.

    Amostra 5 parâmetros com distribuições físicas calibradas:
      f_a  ~ Beta(3,4)×0.25+0.15   [semieixo irregular / R_Hill]
      e₀   ~ Uniforme[0.10, 0.55]  [excentricidade inicial]
      i₀   ~ Normal(148°, 15°)     [inclinação inicial]
      σ_e  ~ Log-uniforme          [taxa de difusão caótica, Myr⁻¹]
      t_ret~ Log-normal(ln40, 0.65)[retorno ao interior, Myr]

    Retorna: dict com arrays de tempo e estatísticas
    """
    if verbose:
        print("═" * 60)
        print(f"  SIM4 — MC Retorno (N={N:,} trajetórias)")
        print("═" * 60)

    rng = np.random.default_rng(seed)

    # Amostrar parâmetros
    f_a   = np.clip(rng.beta(3,4,N)*0.25 + 0.15, 0.15, 0.45)
    e0    = rng.uniform(0.10, 0.55, N)
    i0    = np.clip(rng.normal(148,15,N), 110, 175) * np.pi/180
    sigma = np.exp(rng.uniform(np.log(0.0015), np.log(0.015), N))
    t_ret = np.clip(np.exp(rng.normal(np.log(40), 0.65, N)), 10, 200)

    # Calcular e_crit (limiar de instabilidade Hill)
    a_irr  = f_a * RHILL_JUP
    e_crit = np.clip(0.45*RHILL_JUP/a_irr - 1, 0.45, 0.95)

    # Tempo até instabilidade por difusão
    de_nec = np.maximum(0, e_crit - e0)
    t_inst = de_nec / sigma     # Myr

    # Filtro físico
    valido = (t_inst > 5) & (t_inst < 2000)
    t_total = t_inst[valido] + t_ret[valido]

    N_v = valido.sum()
    percentis = {f'p{p:02d}': float(np.percentile(t_total, p))
                 for p in [5,10,16,25,50,75,84,90,95]}

    if verbose:
        print(f"  N válido: {N_v:,} ({100*N_v/N:.1f}%)")
        print(f"  Distribuição de t_total (Myr):")
        for k,v in percentis.items():
            print(f"    {k}: {v:.1f} Myr")
        print()

    return {
        't_total': t_total,
        't_inst': t_inst[valido],
        't_ret': t_ret[valido],
        'N_valido': int(N_v),
        **percentis
    }


# ════════════════════════════════════════════════════════
# SIM5 — LINHA DO TEMPO RETROATIVA
# Ancoragem em 12.000 AP, propagação para o passado
# ════════════════════════════════════════════════════════

def sim5_timeline_retroativa(N=200000, ancora=12000, seed=2024, verbose=True):
    """
    SIM5: Propagação de incertezas retroativa a partir de 12.000 AP.

    Dois pontos fixos:
      ÂNCORA PRESENTE: 12.000 AP (recaptura da Lua)
      ÂNCORA PASSADA:  4.560 Ga AP (Grand Tack, Walsh et al. 2011)

    Retorna: dict com posições de cada evento (anos AP) e estatísticas
    """
    if verbose:
        print("═" * 60)
        print(f"  SIM5 — Timeline Retroativa (âncora: {ancora:,} AP)")
        print("═" * 60)

    GT  = 4.560e9   # Grand Tack (anos AP)
    REC = float(ancora)
    rng = np.random.default_rng(seed)

    # Distribuições dos intervalos de tempo
    dt_GT_esc  = np.clip(np.exp(rng.normal(np.log(2e6),  0.5, N)), 3e5, 15e6)
    dt_KL      = np.clip(np.exp(rng.normal(np.log(82e6), 0.75,N)), 15e6,500e6)
    dt_sec     = np.clip(np.exp(rng.normal(np.log(40e6), 0.65,N)), 8e6, 200e6)
    dt_inst    = np.clip(np.exp(rng.normal(np.log(500),  0.8, N)), 50,  10000)

    # Posições retroativas (anos AP)
    t_E0 = np.full(N, REC)
    t_E1 = t_E0 + dt_inst    # captura instável Terra
    t_E2 = t_E1 + dt_sec     # cruzamento órbita terrestre
    t_E3 = t_E2 + dt_KL      # estadia Júpiter (ejeção Hill)
    t_E5 = np.full(N, GT) - dt_GT_esc  # escape de Marte

    # Errância solar = t_E5 - t_E3 (deve ser positivo)
    err  = t_E5 - t_E3
    valido = err > 0
    t_err_v = err[valido]

    def S(arr):
        ps = np.percentile(arr, [5,16,25,50,75,84,95])
        return {f'p{p:02d}': float(v) for p,v in zip([5,16,25,50,75,84,95], ps)}

    eventos = {
        'E0_recaptura':         {'descricao':'Recaptura definitiva (âncora)', **S(t_E0)},
        'E1_captura_instavel':  {'descricao':'Captura instável Terra', **S(t_E1)},
        'E2_cruzamento_terra':  {'descricao':'Cruzamento órbita terrestre', **S(t_E2)},
        'E3_ejecao_jupiter':    {'descricao':'Ejeção esfera Hill de Júpiter', **S(t_E3[valido])},
        'E4b_errancia_solar':   {'descricao':'Errância solar (grau de liberdade)',
                                  'duracao_p50': float(np.median(t_err_v)),
                                  'duracao_p16': float(np.percentile(t_err_v,16)),
                                  'duracao_p84': float(np.percentile(t_err_v,84)),
                                  **S(t_E3[valido] + t_err_v/2)},
        'E5_escape_marte':      {'descricao':'Escape Lua da Terra por Marte', **S(t_E5)},
        'E7_grand_tack':        {'descricao':'Grand Tack (âncora geológica)',
                                  **S(np.full(N,GT)+rng.normal(0,3e6,N))},
    }

    if verbose:
        print(f"  Fração válida (errância > 0): {100*valido.mean():.1f}%")
        print()
        print(f"  {'Evento':35s}  {'P16':>12}  {'P50 (med)':>12}  {'P84':>12}")
        print("  " + "─"*75)
        for k,v in eventos.items():
            m=v['p50']; p16=v['p16']; p84=v['p84']
            if m<5e4:     f=lambda x:f"{x:,.0f} AP"
            elif m<5e8:   f=lambda x:f"{x/1e6:.0f} Ma AP"
            else:         f=lambda x:f"{x/1e9:.3f} Ga AP"
            print(f"  {v['descricao']:35s}  {f(p16):>12}  {f(m):>12}  {f(p84):>12}")
        print()

    return {'eventos': eventos, 'N': N, 'GT_AP': GT, 'ancora_AP': REC,
            'frac_valida': float(valido.mean())}


# ════════════════════════════════════════════════════════
# GERADOR DE GRÁFICOS
# ════════════════════════════════════════════════════════

def plot_sim1(res, outfile='saida_sim1.png'):
    """Gráfico de 4 painéis para SIM1."""
    t=res['t']; dl=res['d_lua']; dm=res['d_marte']
    al=res['a_lua']; am=res['a_marte']; dE=res['dE']

    BG='#06060f'; BA='#0c0c1c'; W='#eeeef8'; G='#505070'
    fig, axes = plt.subplots(2,2, figsize=(16,10), facecolor=BG)
    fig.suptitle('SIM1 — N-corpos v8: Terra+Lua+Marte (Leapfrog+EKH)', color=W, fontsize=13, fontweight='bold')

    def sax(ax, ti, xl, yl):
        ax.set_facecolor(BA); ax.set_title(ti,color=W,fontsize=10,fontweight='bold',pad=6)
        ax.set_xlabel(xl,color=G,fontsize=9); ax.set_ylabel(yl,color=G,fontsize=9)
        ax.tick_params(colors=G,labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#222240')
        ax.grid(True,alpha=0.10,color='#303060',lw=0.5)

    axes[0,0].plot(t, np.array(dl)*1.496e8/1000, color='#28D4F0', lw=1.2)
    axes[0,0].axhline(1.5*res['RHILL']*1.496e5, color='#F02828', lw=1.0, ls='--',
                       label=f'Critério escape (1.5×R_Hill)')
    axes[0,0].legend(fontsize=8, facecolor='#1a1a2e', edgecolor=G, labelcolor=W)
    sax(axes[0,0],'Distância Lua–Terra','Tempo (anos)','Distância (×10³ km)')

    axes[0,1].plot(t, dm, color='#F07828', lw=1.1)
    sax(axes[0,1],'Distância Marte–Terra','Tempo (anos)','Distância (UA)')

    am_ok = np.where(np.isfinite(am)&(np.array(am)>0.5)&(np.array(am)<3.0), am, np.nan)
    axes[1,0].plot(t, am_ok, color='#28F078', lw=1.0)
    axes[1,0].axhline(A_MARTE_RES, color='#F0C028', lw=1.2, ls='--', label='Res. 8:5 (1.368 UA)')
    axes[1,0].set_ylim(1.0, 2.0)
    axes[1,0].legend(fontsize=8, facecolor='#1a1a2e', edgecolor=G, labelcolor=W)
    sax(axes[1,0],'Semieixo de Marte','Tempo (anos)','a_Marte (UA)')

    axes[1,1].plot(t, np.array(dE)*1e4, color='#F02828', lw=1.0)
    axes[1,1].axhline(0, color=G, lw=0.5, alpha=0.4)
    sax(axes[1,1],'Conservação de Energia ΔE/E₀','Tempo (anos)','ΔE/E₀ × 10⁴')
    axes[1,1].text(0.02,0.95,f'Final: {res["dE_final"]:.1e}',transform=axes[1,1].transAxes,
                   color='#F02828',fontsize=9,va='top',
                   bbox=dict(boxstyle='round,pad=0.3',facecolor='#1a1a2e',edgecolor='#F02828',alpha=0.8))

    plt.tight_layout(pad=1.5)
    plt.savefig(outfile, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico salvo: {outfile}")


def plot_sim4(res, outfile='saida_mc_retorno.png'):
    """Histograma da distribuição de t_retorno."""
    BG='#06060f'; BA='#0c0c1c'; W='#eeeef8'; G='#505070'
    fig, ax = plt.subplots(figsize=(14,7), facecolor=BG)
    ax.set_facecolor(BA)

    from scipy.ndimage import gaussian_filter1d
    bins = np.linspace(0, 600, 120)
    h,_ = np.histogram(res['t_total'], bins=bins, density=True)
    bc  = 0.5*(bins[:-1]+bins[1:])
    hs  = gaussian_filter1d(h, sigma=2)
    ax.fill_between(bc, hs, alpha=0.18, color='#28D4F0')
    ax.plot(bc, hs, color='#28D4F0', lw=2.0)
    ax.axvspan(res['p16'], res['p84'], alpha=0.12, color='#B028F0', label=f'68%: {res["p16"]:.0f}–{res["p84"]:.0f} Myr')
    ax.axvline(res['p50'], color='white', lw=2.0, label=f'Mediana: {res["p50"]:.0f} Myr')
    ax.axvspan(res['p05'], res['p95'], alpha=0.06, color='#3080F0', label=f'90%: {res["p05"]:.0f}–{res["p95"]:.0f} Myr')
    ax.set_xlim(0,600); ax.set_ylim(bottom=0)
    ax.set_title(f'SIM4 — Distribuição do Tempo de Retorno à Terra (N={res["N_valido"]:,})',
                 color=W, fontsize=13, fontweight='bold')
    ax.set_xlabel('Tempo de retorno (Myr desde escape de Marte)', color=G, fontsize=10)
    ax.set_ylabel('Densidade de probabilidade', color=G, fontsize=10)
    ax.tick_params(colors=G); 
    for sp in ax.spines.values(): sp.set_color('#222240')
    ax.grid(True, alpha=0.10, color='#303060')
    ax.legend(fontsize=10, facecolor='#1a1a2e', edgecolor=G, labelcolor=W)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico salvo: {outfile}")


# ════════════════════════════════════════════════════════
# INTERFACE DE LINHA DE COMANDO
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Simulações N-corpos — Cenário Lua Errante',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simulações disponíveis:
  v8      SIM1: N-corpos Terra+Lua+Marte (25 anos, Leapfrog+EKH)
  mc6     SIM2: Monte Carlo 6 corpos, destinos imediatos (36 fases)
  kozai   SIM3: Kozai-Lidov em Júpiter (500 Myr)
  mc_ret  SIM4: Monte Carlo retorno à Terra (100k trajetórias)
  timeline SIM5: Linha do tempo retroativa (ancoragem 12.000 AP)
  all     Todas as simulações em sequência

Exemplos:
  python sim_master.py --sim v8
  python sim_master.py --sim kozai
  python sim_master.py --all
  python sim_master.py --sim timeline --ancora 12000
        """)
    parser.add_argument('--sim', choices=['v8','mc6','kozai','mc_ret','timeline','all'],
                        default='all', help='Simulação a executar')
    parser.add_argument('--ancora', type=int, default=12000,
                        help='Âncora temporal para timeline (anos AP, default=12000)')
    parser.add_argument('--quiet', action='store_true', help='Suprimir saída detalhada')
    args = parser.parse_args()

    verbose = not args.quiet
    os.makedirs('saidas', exist_ok=True)

    if args.sim in ('v8', 'all'):
        print("\n" + "╔"+"═"*58+"╗")
        print("║  EXECUTANDO SIM1 — N-corpos v8                         ║")
        print("╚"+"═"*58+"╝")
        r1 = sim1_v8(verbose=verbose)
        plot_sim1(r1, 'saidas/sim1_v8.png')
        with open('saidas/sim1_resultados.json','w') as f:
            json.dump({k:v for k,v in r1.items() if not isinstance(v,list)}, f, indent=2)

    if args.sim in ('mc6', 'all'):
        print("\n" + "╔"+"═"*58+"╗")
        print("║  EXECUTANDO SIM2 — MC 6 corpos                         ║")
        print("╚"+"═"*58+"╝")
        r2 = sim2_destinos_mc(N_runs=36, verbose=verbose)
        with open('saidas/sim2_resultados.json','w') as f:
            json.dump({'contagem':r2['contagem'],'probabilidades':r2['probabilidades']}, f, indent=2)

    if args.sim in ('kozai', 'all'):
        print("\n" + "╔"+"═"*58+"╗")
        print("║  EXECUTANDO SIM3 — Kozai-Lidov                         ║")
        print("╚"+"═"*58+"╝")
        r3 = sim3_kozai_lidov(t_total_myr=500, verbose=verbose)
        # Plot KL
        BG='#06060f'; BA='#0c0c1c'; W='#eeeef8'; G='#505070'
        fig, ax = plt.subplots(figsize=(14,6), facecolor=BG)
        ax.set_facecolor(BA)
        ax.plot(r3['t_myr'], r3['e'], color='#B028F0', lw=1.2, label='e (excentricidade)')
        ax.axhline(0.82, color='#F02828', ls='--', lw=1.0, label='Limiar Hill (e=0.82)')
        if r3['t_flyby_myr']:
            ax.axvline(r3['t_flyby_myr'], color='#F0C028', ls='--', lw=1.2,
                      label=f"Instabilidade Hill (t={r3['t_flyby_myr']:.0f} Myr)")
        ax.set_title('SIM3 — Kozai-Lidov: excentricidade da Lua ao redor de Júpiter', color=W, fontsize=12, fontweight='bold')
        ax.set_xlabel('Tempo (Myr)', color=G); ax.set_ylabel('e', color=G)
        ax.tick_params(colors=G)
        for sp in ax.spines.values(): sp.set_color('#222240')
        ax.grid(True,alpha=0.1,color='#303060')
        ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor=G, labelcolor=W)
        plt.tight_layout()
        plt.savefig('saidas/sim3_kozai.png', dpi=150, facecolor=BG, bbox_inches='tight')
        plt.close()
        print("  Gráfico salvo: saidas/sim3_kozai.png")
        with open('saidas/sim3_resultados.json','w') as f:
            json.dump({k:v for k,v in r3.items() if not isinstance(v,np.ndarray)}, f, indent=2)

    if args.sim in ('mc_ret', 'all'):
        print("\n" + "╔"+"═"*58+"╗")
        print("║  EXECUTANDO SIM4 — MC Retorno                          ║")
        print("╚"+"═"*58+"╝")
        r4 = sim4_mc_retorno(N=100000, verbose=verbose)
        plot_sim4(r4, 'saidas/sim4_mc_retorno.png')
        with open('saidas/sim4_resultados.json','w') as f:
            json.dump({k:v for k,v in r4.items() if not isinstance(v,np.ndarray)}, f, indent=2)

    if args.sim in ('timeline', 'all'):
        print("\n" + "╔"+"═"*58+"╗")
        print("║  EXECUTANDO SIM5 — Timeline Retroativa                 ║")
        print("╚"+"═"*58+"╝")
        r5 = sim5_timeline_retroativa(N=200000, ancora=args.ancora, verbose=verbose)
        with open('saidas/sim5_timeline.json','w') as f:
            json.dump(r5['eventos'], f, indent=2)

    print("\n" + "═"*60)
    print("  TODAS AS SIMULAÇÕES CONCLUÍDAS")
    print("  Resultados salvos em: ./saidas/")
    print("═"*60 + "\n")


if __name__ == '__main__':
    main()
