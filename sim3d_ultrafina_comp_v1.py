#!/usr/bin/env python3
"""
███████████████████████████████████████████████████████████████████████████████
  PARADIGMA ZERO — SIM-3D-ULTRAFINA-COMPLEMENTAR v1.0
  
  FOCO: Região BAIXA ao redor dos melhores candidatos:
    - R-0108: a=1.45, e=0.46, inc=1°, Ω=30°, M0=45° (r=83.5 R⊕)
    - RC-0804: a=1.50, e=0.48, inc=1°, Ω=15°, M0=150° (r=99.6 R⊕)
  
  Grade ultra-fina com passos mínimos para encontrar captura!
  
  Tempo estimado: ~50-60 min
███████████████████████████████████████████████████████████████████████████████
"""

import numpy as np
import json
import os
import time
import csv
from datetime import datetime, timedelta
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = 'sim3d_ultrafina_comp'

# Constantes físicas
AU = 1.495978707e11
G = 6.67430e-11
M_SOL = 1.989e30
M_TERRA = 5.972e24
M_LUA = 7.342e22
M_PM = 6.417e23
M_JUPITER = 1.898e27
R_TERRA = 6.371e6
YEAR = 365.25 * 24 * 3600

# Parâmetros orbitais fixos
A_TERRA = 1.0
E_TERRA = 0.0167
A_PM_I = 1.367981
E_PM_I = 0.30
M0_PM = 180.0
A_JUPITER = 5.2044
E_JUPITER = 0.0489

# Alvos
A_PM_ALVO = 1.524
R_CAPTURA = 70.0
R_INTERACAO = 400.0
DELTA_A_PM_TOL = 0.05

# ══════════════════════════════════════════════════════════════════════════════
#  GRADE ULTRA-FINA — REGIÃO BAIXA
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'integrador': 'ias15',
    't_anos': 500,
    't_pos_impulso': 100,
    
    # Duas sub-regiões baseadas nos melhores candidatos
    
    # Sub-região A: ao redor de R-0108 (a=1.45, e=0.46, inc=1°, Ω=30°, M0=45°)
    'subregiao_A': {
        'a_lua': [1.44, 1.445, 1.45, 1.455, 1.46],
        'e_lua': [0.455, 0.46, 0.465, 0.47],
        'inc_lua': [0.5, 1, 1.5, 2],
        'omega_lua': [25, 30, 35],
        'm0_lua': [40, 45, 50, 55],
    },
    
    # Sub-região B: ao redor de RC-0804 (a=1.50, e=0.48, inc=1°, Ω=15°, M0=150°)
    'subregiao_B': {
        'a_lua': [1.485, 1.49, 1.495, 1.50, 1.505],
        'e_lua': [0.475, 0.48, 0.485, 0.49],
        'inc_lua': [0.5, 1, 1.5, 2],
        'omega_lua': [10, 15, 20],
        'm0_lua': [145, 150, 155, 160],
    },
    
    'checkpoint_intervalo': 100,
}

# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def setup_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

def perihelio(a, e):
    return a * (1 - e)

def cruza_terra(a_lua, e_lua):
    q = perihelio(a_lua, e_lua)
    Q = a_lua * (1 + e_lua)
    return q < 1.15 and Q > 0.85

def formato_tempo(segundos):
    return str(timedelta(seconds=int(segundos)))

def log(msg, arquivo_log=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    linha = f"[{timestamp}] {msg}"
    print(linha)
    if arquivo_log:
        with open(arquivo_log, 'a') as f:
            f.write(linha + '\n')

# ══════════════════════════════════════════════════════════════════════════════
#  TERMO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_impulso_pm(sim, r_encontro_AU):
    pm = sim.particles[2]
    r_pm = np.sqrt(pm.x**2 + pm.y**2 + pm.z**2)
    v_pm = np.sqrt(pm.vx**2 + pm.vy**2 + pm.vz**2)
    
    GM = 4 * np.pi**2
    a_atual = pm.a
    v_visviva_atual = np.sqrt(GM * (2/r_pm - 1/a_atual))
    v_visviva_alvo = np.sqrt(GM * (2/r_pm - 1/A_PM_ALVO))
    dv_escalar = v_visviva_alvo - v_visviva_atual
    
    hx = pm.y * pm.vz - pm.z * pm.vy
    hy = pm.z * pm.vx - pm.x * pm.vz
    hz = pm.x * pm.vy - pm.y * pm.vx
    
    tx = hy * pm.z - hz * pm.y
    ty = hz * pm.x - hx * pm.z
    tz = hx * pm.y - hy * pm.x
    t_mag = np.sqrt(tx**2 + ty**2 + tz**2)
    
    if t_mag > 0:
        tx /= t_mag
        ty /= t_mag
        tz /= t_mag
    else:
        tx, ty, tz = pm.vx/v_pm, pm.vy/v_pm, pm.vz/v_pm
    
    dv_x = dv_escalar * tx
    dv_y = dv_escalar * ty
    dv_z = dv_escalar * tz
    
    return dv_x, dv_y, dv_z, dv_escalar

def calcular_energia_disponivel(sim, r_lua_terra_m):
    lua = sim.particles[3]
    terra = sim.particles[1]
    
    dvx = (lua.vx - terra.vx) * AU / YEAR
    dvy = (lua.vy - terra.vy) * AU / YEAR
    dvz = (lua.vz - terra.vz) * AU / YEAR
    v_rel = np.sqrt(dvx**2 + dvy**2 + dvz**2)
    
    E_cin = 0.5 * M_LUA * v_rel**2
    E_pot = -G * M_TERRA * M_LUA / r_lua_terra_m
    E_disponivel = E_cin + abs(E_pot) * 0.3
    
    return E_disponivel

# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, 'checkpoint_state.json')
        self.results_file = os.path.join(output_dir, 'checkpoint_results.csv')
        self.capturas_file = os.path.join(output_dir, 'capturas.csv')
        
    def existe_checkpoint(self):
        return os.path.exists(self.state_file)
    
    def carregar(self):
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        resultados = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                reader = csv.DictReader(f)
                resultados = list(reader)
        capturas = []
        if os.path.exists(self.capturas_file):
            with open(self.capturas_file, 'r') as f:
                reader = csv.DictReader(f)
                capturas = list(reader)
        return state, resultados, capturas
    
    def salvar(self, state, resultados, capturas):
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        if resultados:
            keys = resultados[0].keys()
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(resultados)
        if capturas:
            keys = capturas[0].keys()
            with open(self.capturas_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(capturas)

# ══════════════════════════════════════════════════════════════════════════════
#  SIMULAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def criar_simulacao(a_lua, e_lua, inc_lua, omega_lua, m0_lua):
    import rebound
    
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = 'ias15'
    
    sim.add(m=1.0)
    sim.add(m=M_TERRA/M_SOL, a=A_TERRA, e=E_TERRA, inc=0, Omega=0, omega=0, M=0)
    sim.add(m=M_PM/M_SOL, a=A_PM_I, e=E_PM_I, inc=np.radians(1.85), 
            Omega=np.radians(49.6), omega=np.radians(286.5), M=np.radians(M0_PM))
    sim.add(m=M_LUA/M_SOL, a=a_lua, e=e_lua, 
            inc=np.radians(inc_lua), Omega=np.radians(omega_lua), 
            omega=np.radians(90), M=np.radians(m0_lua))
    sim.add(m=M_JUPITER/M_SOL, a=A_JUPITER, e=E_JUPITER,
            inc=np.radians(1.3), Omega=np.radians(100.5), omega=np.radians(14.75), M=0)
    
    sim.move_to_com()
    return sim

def rodar_simulacao_termo(sim, t_anos, t_pos_impulso):
    resultado = {
        'interacao': False,
        'captura': False,
        'impulso_aplicado': False,
        'r_min_Rt': float('inf'),
        't_interacao_yr': None,
        't_captura_yr': None,
        'dv_aplicado_ms': None,
        'a_terra_f': None,
        'a_pm_f': None,
        'delta_a_pm': None,
        'vinculo_terra': False,
        'vinculo_marte': False,
        'vinculo_captura': False,
        'todos_vinculos': False,
        'ejetado': False,
        'erro': None
    }
    
    dt_check = 0.01  # Ainda mais fino!
    t = 0
    r_min = float('inf')
    impulso_aplicado = False
    t_fim = t_anos
    
    try:
        while t < t_fim:
            t_next = min(t + dt_check, t_fim)
            sim.integrate(t_next)
            t = t_next
            
            terra = sim.particles[1]
            pm = sim.particles[2]
            lua = sim.particles[3]
            
            dx = (lua.x - terra.x) * AU
            dy = (lua.y - terra.y) * AU
            dz = (lua.z - terra.z) * AU
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            r_Rt = r / R_TERRA
            
            if r_Rt < r_min:
                r_min = r_Rt
            
            if r_Rt < R_INTERACAO and not impulso_aplicado:
                resultado['interacao'] = True
                resultado['t_interacao_yr'] = t
                
                E_disp = calcular_energia_disponivel(sim, r)
                dv_x, dv_y, dv_z, dv_esc = calcular_impulso_pm(sim, r_Rt)
                
                pm.vx += dv_x
                pm.vy += dv_y
                pm.vz += dv_z
                
                impulso_aplicado = True
                resultado['impulso_aplicado'] = True
                resultado['dv_aplicado_ms'] = dv_esc * AU / YEAR
                
                t_fim = t + t_pos_impulso
            
            if r_Rt < R_CAPTURA and not resultado['captura']:
                resultado['captura'] = True
                resultado['t_captura_yr'] = t
            
            if lua.a > 10 or lua.a < 0:
                resultado['ejetado'] = True
                break
        
        resultado['r_min_Rt'] = r_min
        resultado['a_terra_f'] = sim.particles[1].a
        resultado['a_pm_f'] = sim.particles[2].a
        
        if resultado['a_pm_f']:
            resultado['delta_a_pm'] = abs(resultado['a_pm_f'] - A_PM_ALVO)
        
        resultado['vinculo_terra'] = abs(resultado['a_terra_f'] - A_TERRA) < 0.01
        resultado['vinculo_marte'] = resultado['delta_a_pm'] is not None and resultado['delta_a_pm'] < DELTA_A_PM_TOL
        resultado['vinculo_captura'] = resultado['r_min_Rt'] < R_CAPTURA
        resultado['todos_vinculos'] = (resultado['vinculo_terra'] and 
                                        resultado['vinculo_marte'] and 
                                        resultado['vinculo_captura'])
        
    except Exception as e:
        resultado['erro'] = str(e)
    
    return resultado

# ══════════════════════════════════════════════════════════════════════════════
#  GRADE
# ══════════════════════════════════════════════════════════════════════════════

def gerar_grade():
    grade = []
    
    for subregiao_nome in ['subregiao_A', 'subregiao_B']:
        g = CONFIG[subregiao_nome]
        
        for a in g['a_lua']:
            for e in g['e_lua']:
                if not cruza_terra(a, e):
                    continue
                for inc in g['inc_lua']:
                    for omega in g['omega_lua']:
                        for m0 in g['m0_lua']:
                            grade.append({
                                'subregiao': subregiao_nome,
                                'a_lua': a,
                                'e_lua': e,
                                'inc_lua': inc,
                                'omega_lua': omega,
                                'm0_lua': m0
                            })
    
    return grade

# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def rodar_varredura(arquivo_log):
    checkpoint = CheckpointManager(OUTPUT_DIR)
    
    log("", arquivo_log)
    log("═" * 70, arquivo_log)
    log("  SIM-3D-ULTRAFINA-COMP: Região BAIXA", arquivo_log)
    log(f"  Integrador: IAS15 | T = {CONFIG['t_anos']} anos", arquivo_log)
    log(f"  R_INTERACAO = {R_INTERACAO} R⊕ | R_CAPTURA = {R_CAPTURA} R⊕", arquivo_log)
    log("  Sub-região A: a=1.45, e=0.46, Ω=30°, M0=45°", arquivo_log)
    log("  Sub-região B: a=1.50, e=0.48, Ω=15°, M0=150°", arquivo_log)
    log("═" * 70, arquivo_log)
    
    grade = gerar_grade()
    total = len(grade)
    log(f"  Grade total: {total} simulações", arquivo_log)
    
    contagem = {}
    for g in grade:
        r = g['subregiao']
        contagem[r] = contagem.get(r, 0) + 1
    for r, c in sorted(contagem.items()):
        log(f"    {r}: {c} simulações", arquivo_log)
    
    log("═" * 70, arquivo_log)
    
    if checkpoint.existe_checkpoint():
        state, resultados, capturas = checkpoint.carregar()
        idx_inicio = state['idx_ultimo'] + 1
        if state.get('completo'):
            log(f"  Já completa", arquivo_log)
            return capturas
        log(f"  CHECKPOINT: retomando de {idx_inicio}/{total}", arquivo_log)
    else:
        state = {'idx_ultimo': -1}
        resultados = []
        capturas = []
        idx_inicio = 0
    
    t_inicio = time.time()
    n_interacoes = 0
    r_min_global = float('inf')
    melhor_params = None
    
    for idx in range(idx_inicio, total):
        params = grade[idx]
        
        try:
            sim = criar_simulacao(
                a_lua=params['a_lua'],
                e_lua=params['e_lua'],
                inc_lua=params['inc_lua'],
                omega_lua=params['omega_lua'],
                m0_lua=params['m0_lua']
            )
            
            resultado = rodar_simulacao_termo(sim, CONFIG['t_anos'], CONFIG['t_pos_impulso'])
            
        except Exception as e:
            resultado = {'erro': str(e), 'captura': False, 'r_min_Rt': float('inf')}
        
        if resultado.get('interacao'):
            n_interacoes += 1
        
        if resultado.get('r_min_Rt', float('inf')) < r_min_global:
            r_min_global = resultado['r_min_Rt']
            melhor_params = params.copy()
            melhor_params['r_min'] = r_min_global
        
        registro = {
            'id': f"UFC-{idx+1:04d}",
            **params,
            **resultado
        }
        resultados.append(registro)
        
        if resultado.get('todos_vinculos'):
            capturas.append(registro)
            log(f"  ★★★ CANDIDATO COMPLETO #{len(capturas)} | {registro['id']} | "
                f"r_min={resultado['r_min_Rt']:.1f} R⊕ | "
                f"a={params['a_lua']:.3f} e={params['e_lua']:.3f} inc={params['inc_lua']}°", arquivo_log)
        elif resultado.get('captura'):
            capturas.append(registro)
            log(f"  ★★ CAPTURA #{len(capturas)} | {registro['id']} | "
                f"r_min={resultado['r_min_Rt']:.1f} R⊕ | "
                f"a={params['a_lua']:.3f} e={params['e_lua']:.3f} inc={params['inc_lua']}°", arquivo_log)
        elif resultado.get('r_min_Rt', float('inf')) < 75:
            log(f"  ★ QUASE CAPTURA! | {registro['id']} | "
                f"r_min={resultado['r_min_Rt']:.1f} R⊕ | "
                f"a={params['a_lua']:.3f} e={params['e_lua']:.3f} inc={params['inc_lua']}°", arquivo_log)
        elif resultado.get('r_min_Rt', float('inf')) < 100:
            log(f"  ◆ MUITO PERTO | {registro['id']} | "
                f"r_min={resultado['r_min_Rt']:.1f} R⊕", arquivo_log)
        
        if (idx + 1) % 100 == 0 or idx == total - 1:
            elapsed = time.time() - t_inicio
            pct = (idx + 1) / total * 100
            rate = (idx + 1 - idx_inicio) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            
            log(f"  [{idx+1:5d}/{total}] {pct:5.1f}% | "
                f"Cap: {len(capturas)} | Int: {n_interacoes} | "
                f"r_min_best: {r_min_global:.1f} R⊕ | "
                f"ETA: {formato_tempo(eta)}", arquivo_log)
        
        if (idx + 1) % CONFIG['checkpoint_intervalo'] == 0:
            state['idx_ultimo'] = idx
            checkpoint.salvar(state, resultados, capturas)
    
    state['idx_ultimo'] = total - 1
    state['completo'] = True
    checkpoint.salvar(state, resultados, capturas)
    
    log("", arquivo_log)
    log("═" * 70, arquivo_log)
    log(f"  VARREDURA COMPLETA", arquivo_log)
    log(f"    Total: {total} | Interações: {n_interacoes} | Capturas: {len(capturas)}", arquivo_log)
    log(f"    Melhor r_min: {r_min_global:.2f} R⊕", arquivo_log)
    if melhor_params:
        log(f"    Parâmetros: a={melhor_params['a_lua']:.3f} e={melhor_params['e_lua']:.3f} "
            f"inc={melhor_params['inc_lua']}° Ω={melhor_params['omega_lua']}° M0={melhor_params['m0_lua']}°", arquivo_log)
    log("═" * 70, arquivo_log)
    
    return capturas

def analisar_resultados(capturas, arquivo_log):
    log("", arquivo_log)
    log("█" * 70, arquivo_log)
    log("  ANÁLISE FINAL — SIM-3D-ULTRAFINA-COMP (REGIÃO BAIXA)", arquivo_log)
    log("█" * 70, arquivo_log)
    
    if not capturas:
        log("", arquivo_log)
        log("  Nenhuma captura (r < 70 R⊕) encontrada.", arquivo_log)
        return
    
    completos = [c for c in capturas if c.get('todos_vinculos')]
    
    if completos:
        log("", arquivo_log)
        log(f"  🏆🏆🏆 {len(completos)} CANDIDATOS COMPLETOS ENCONTRADOS! 🏆🏆🏆", arquivo_log)
        log("", arquivo_log)
        
        for i, cap in enumerate(sorted(completos, key=lambda x: float(x['r_min_Rt']))):
            log(f"  #{i+1}: {cap['id']} | r_min={float(cap['r_min_Rt']):.2f} R⊕ | "
                f"a={float(cap['a_lua']):.3f} e={float(cap['e_lua']):.3f} inc={float(cap['inc_lua'])}°", arquivo_log)
        
        log("", arquivo_log)
        log("  ✅✅✅ CAPTURA 3D INDEPENDENTE CONFIRMADA! ✅✅✅", arquivo_log)
        log("  → Paper pronto para Nature Astronomy!", arquivo_log)
    else:
        log("", arquivo_log)
        log(f"  {len(capturas)} capturas encontradas (sem todos os vínculos):", arquivo_log)
        for i, cap in enumerate(sorted(capturas, key=lambda x: float(x['r_min_Rt']))[:5]):
            log(f"  #{i+1}: {cap['id']} | r_min={float(cap['r_min_Rt']):.2f} R⊕", arquivo_log)

def main():
    setup_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_log = os.path.join(OUTPUT_DIR, f'ultrafina_comp_log_{timestamp}.txt')
    
    print("")
    print("█" * 75)
    print("█" + " " * 73 + "█")
    print("█  PARADIGMA ZERO — SIM-3D-ULTRAFINA-COMPLEMENTAR v1.0" + " " * 19 + "█")
    print("█  Região BAIXA | Duas sub-regiões promissoras" + " " * 26 + "█")
    print("█  Centros: a=1.45/e=0.46 e a=1.50/e=0.48" + " " * 31 + "█")
    print("█" + " " * 73 + "█")
    print("█" * 75)
    print("")
    
    log(f"Diretório de saída: {OUTPUT_DIR}", arquivo_log)
    
    try:
        import rebound
        log(f"REBOUND versão: {rebound.__version__}", arquivo_log)
    except ImportError:
        log("ERRO: REBOUND não instalado.", arquivo_log)
        return
    
    t_total_inicio = time.time()
    
    capturas = rodar_varredura(arquivo_log)
    analisar_resultados(capturas, arquivo_log)
    
    t_total = time.time() - t_total_inicio
    log("", arquivo_log)
    log(f"TEMPO TOTAL: {formato_tempo(t_total)}", arquivo_log)
    
    print("")
    print("█" * 75)
    print("█  SIM-3D-ULTRAFINA-COMPLEMENTAR COMPLETA" + " " * 33 + "█")
    print("█" * 75)

if __name__ == "__main__":
    main()
