# Simulações N-corpos — Cenário Lua Errante
## Guia de Replicação Completo

---

## Dependências

```bash
pip install numpy matplotlib scipy
# Python >= 3.10 recomendado
```

Testado com:
- Python 3.12
- NumPy 1.26
- Matplotlib 3.8
- SciPy 1.11

---

## Arquivo principal

```
sim_master.py   — código unificado de todas as 5 simulações
```

---

## Como executar

```bash
# Simulação específica
python sim_master.py --sim v8        # SIM1: N-corpos Terra+Lua+Marte
python sim_master.py --sim mc6       # SIM2: Monte Carlo 6 corpos (36 fases)
python sim_master.py --sim kozai     # SIM3: Kozai-Lidov em Júpiter (500 Myr)
python sim_master.py --sim mc_ret    # SIM4: Monte Carlo retorno (100k trajetórias)
python sim_master.py --sim timeline  # SIM5: Linha do tempo retroativa

# Todas as simulações em sequência
python sim_master.py --all

# Com âncora temporal diferente
python sim_master.py --sim timeline --ancora 11700

# Modo silencioso
python sim_master.py --all --quiet
```

---

## Saídas geradas

```
saidas/
  sim1_v8.png             — gráfico 4 painéis: distâncias, semieixos, energia
  sim1_resultados.json    — métricas numéricas do SIM1
  sim2_resultados.json    — probabilidades dos 5 destinos possíveis
  sim3_kozai.png          — excentricidade vs tempo (Kozai-Lidov)
  sim3_resultados.json    — parâmetros KL e tempo de instabilidade
  sim4_mc_retorno.png     — distribuição de probabilidade de t_retorno
  sim4_resultados.json    — percentis P5–P95 do tempo de retorno
  sim5_timeline.json      — posições de cada evento com intervalos
```

---

## Premissas axiomáticas (não testadas — tratadas como entrada)

1. Terra e Lua têm mesma origem isotópica
2. Marte expulsou a Lua via perturbação gravitacional
3. Ressonância 8:5 Terra-Marte: a_Marte = (8/5)^(2/3) = 1.3680 UA

---

## Correções implementadas na v8 vs v7 original

| Código | Correção | Impacto |
|--------|----------|---------|
| C1 | RK4 → Leapfrog simplético | Elimina drift de energia |
| C2 | Corte tidal: 0.08→0.015 UA | Remove amplificação 53.000× |
| C4 | ΔE separado por componente | Separa físico de numérico |
| C5 | Calor tidal rastreado | Grandeza física independente |

---

## Parâmetros físicos principais

```python
G        = 4π²            # UA³ M☉⁻¹ ano⁻²
M_Lua    = 3.694e-8 M☉    # IAU 2015
k₂_Lua   = 0.80           # oceano magmático (fluido quase completo)
Q_Lua    = 2.0             # limite dissipativo
k₂/Q     = 0.400          # 1667× Lua sólida atual
R_Hill   = 0.0100 UA      # esfera de influência da Terra
Corte EKH= 0.0150 UA      # 1.5 × R_Hill [CORRIGIDO]
```

---

## Referências dos análogos observacionais

| Simulação | Referência | Dado fornecido |
|-----------|-----------|----------------|
| SIM1 | Eggleton, Kiseleva, Hut 1998 (ApJ 499) | Equações EKH |
| SIM1 | Hut 1981 (A&A 99) | Função f(e) |
| SIM2 | Nesvorný et al. 2007 (AJ 133) | Captura por Júpiter (22%) |
| SIM3 | Carruba et al. 2002 (Icarus 158) | Kozai-Lidov em irregulares |
| SIM3 | Gladman et al. 1997 (Science) | Instabilidade Hill |
| SIM4 | Levison & Duncan 1994 (Icarus) | Retorno ao interior (37.5%) |
| SIM4 | Morbidelli & Nesvorný 1999 | Taxa de difusão σ_e |
| SIM5 | Walsh et al. 2011 (Nature 475) | Grand Tack = 4.560 Ga AP |
| SIM5 | Bottke et al. 2012 (Nature 485) | LHB: 66–120 Ma AP |

---

## Reprodutibilidade

Todas as simulações são determinísticas com seed fixo.
Para reproduzir resultados exatos dos relatórios:
- SIM4: seed=2024 (padrão)
- SIM5: seed=2024 (padrão)

Para variar o seed e explorar o espaço de parâmetros:
```python
from sim_master import sim4_mc_retorno
resultados = sim4_mc_retorno(N=100000, seed=9999)
```

---

## Limitações conhecidas

1. **Integrador Leapfrog**: simplético para forças conservativas.
   EKH (não-conservativo) adicionado como perturbação — erro ~0.01%/Myr
2. **Encontros d < 0.001 UA**: recomenda-se IAS15 (REBOUND) para maior precisão
3. **Simulação 2D**: inclinações orbitais não modeladas
4. **N=36 no SIM2**: suficiente para probabilidades de ordem de grandeza
5. **Errância solar**: grau de liberdade irredutível — não constrangida por dados

