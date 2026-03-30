# PARADIGM-ZERO-V5.7
# Paradigma Zero — Gravitational Capture of the Moon

**A Dynamically Consistent Scenario for Gravitational Capture of the Moon: Computational Evidence and Observational Convergence from 32 Independent Datasets**

> Alexssandro A. — Independent Researcher, Osasco, SP, Brazil  
> Preprint submitted for peer review — March 2026

---

## Abstract

The origin of the Moon remains a central unsolved problem in planetary science. This work presents a systematic, unguided N-body exploration comprising **~65,000 integrations across 26 simulation stages**, executed with REBOUND/IAS15, proposing that the Moon was gravitationally captured by the Earth at approximately **12,000 BP**.

Post-hoc evaluation against **32 independent peer-reviewed observational datasets** produces no contradiction. Five independently verified retrodictions (R1–R5) are documented — all published *after* completion of the simulations.

---

## Key Results

| Candidate | rmin | ΔT | Earth Orbit (M1) | Proto-Mars (M2) |
|-----------|------|----|------------------|-----------------|
| P1-06 | 15.51 R⊕ | 17.8 yr | 1.0004 AU ✅ | 1.4954 AU ✅ |
| P1-03 | 9.84 R⊕ | 122.6 yr | 0.9996 AU ✅ | 1.5169 AU ✅ |

- **Terrestrial obliquity** delivered analytically: Etorque/Etilt = **1.0005** for both candidates
- **LLR kinematic paradox** resolved: ta* = 12.19 kyr for P1-03 (ratio 1.02×)
- **Body Z hypothesis** rejected computationally across 936 orbital configurations — not by theoretical assumption

---

## Falsifiable Predictions

**S-06 — Farside Seismicity** (testable 2027, FSS+LITMS at Schrödinger Basin):
- Confirmation: < 10% of deep moonquakes on the farside
- Falsification: > 30% farside distribution

**L-07 — K-Ar Isotopic Reset**:
- K-Ar reset between 12,000–13,000 BP in Apollo samples from tsunami impact zones
- U-Pb in zircon remains unchanged (closure temperature ~700–900°C vs ~300°C for K-Ar)

---

## Observational Convergence (Selected)

| ID | Observable | Status |
|----|-----------|--------|
| L-10 | Oxygen isotope identity (Δ¹⁷O = 0.2 ± 1.6 ppm) | ✅ MATCH |
| L-02a/b | Lunar mantle thermal asymmetry 100–200 K | ✅ MATCH (R1, R2) |
| T-01 | Terrestrial obliquity 23.44° | ✅ MATCH |
| T-02 | LLR kinematic paradox | ✅ MATCH |
| Ma-01 | Mars semi-major axis 1.523679 AU | ✅ MATCH |
| Ma-02 | Mars inclination 1.85° | ✅ MATCH |
| T-07 | YDB synchronous age ~12,800 BP | ✅ MATCH |
| S-06 | Farside seismicity (2027) | 🔮 PREDICTION |

---

## Verified Retrodictions (Post-Simulation Publications)

| # | Reference | Finding | PZ Prediction |
|---|-----------|---------|---------------|
| R1 | Park et al., *Nature* (May 2025) | Mantle thermal asymmetry k3=0.0163 | Asymmetric tidal pressure at rmin < 20 R⊕ |
| R2 | He et al., *Nature Geoscience* (Sep 2025) | Farside mantle ~100°C cooler | Persistent gradient from asymmetric capture |
| R3 | Fischer et al., *PNAS* (Dec 2024) | Δ¹⁷O = 0.2 ± 1.6 ppm | Co-formation in same nebular reservoir |
| R4 | Cai et al., *Science Advances* (Jan 2025) | Dynamo ~2–4 µT at 2.0 Ga | Tidal extension of core convection |
| R5 | Barboni et al., *Nature Comm.* (Mar 2025) | Pervasive impact reworking of lunar clasts | Recent energetic history (~12,000 BP) |

---

## Simulation Chain

```
SIM1  →  Lunar escape from primordial geocentric orbit (tidal + 8:5 MMR)
SIM2  →  Monte Carlo destiny mapping (N=36; 5 dynamical fates)
SIM3  →  Kozai–Lidov secular oscillation as Jovian irregular satellite
SIM4  →  Monte Carlo return trajectory (N=94,389; median 135 Myr)
SIM5  →  Retroactive timeline with dual anchors (N=200,000)
SIM-A →  Variable-dissipation tidal model (120 runs; resolves M4 for P1-03)
SIM-3D → 3D scanning and validation (63,200 integrations)
```

**Total: > 65,000 N-body integrations across 26 stages**

---

## Repository Structure

```
/
├── README.md
├── paper/
│   └── paradigma_zero_v5.7.pdf        ← Full preprint
├── simulations/
│   ├── SIM1_lunar_escape/
│   ├── SIM2_monte_carlo_destiny/
│   ├── SIM3_kozai_lidov/
│   ├── SIM4_return_trajectories/
│   ├── SIM5_retroactive_timeline/
│   ├── SIM-A_tidal_model/
│   └── SIM-3D_validation/
├── data/
│   └── convergence_matrix.csv
└── figures/
    └── ...
```

---

## Methods

- **Integrator:** REBOUND v4.6.0 with IAS15 (capture phases) and WHFast (long-term MMR screening)
- **Framework:** TERMO — Tidal Energy and Resonance Migration Operator (hybrid analytical + N-body)
- **Tidal model:** Eggleton, Kiseleva & Hut (1998) equilibrium tide; Hut (1981) eccentricity functions
- **Lunar tidal state:** k₂/Q = 0.400 (global magma ocean; fluid-body limit)
- **System modeled:** Sun, Earth, Proto-Mars, Moon, Jupiter (5-body)

---

## Citation

```bibtex
@article{alexssandro2026paradigmazero,
  title   = {Paradigma Zero: A Dynamically Consistent Scenario for Gravitational
             Capture of the Moon — Computational Evidence and Observational
             Convergence from 32 Independent Datasets},
  author  = {Alexssandro A.},
  year    = {2026},
  month   = {March},
  note    = {Preprint submitted for peer review},
  url     = {https://github.com/alewztrong/PARADIGM-ZERO-V5.7}
}
```

---

## Contact

**Alexssandro N.**  
Independent Researcher — Osasco, SP, Brazil  
📧 alex7ndo@gmail.com

---

*"The convergence between an unguided computational exploration and independent observational constraints documents a scenario that was not constructed to fit the data, but found within them."*
