# Quantum Monte Carlo for Multidimensional Call Option Pricing (MIQAE)

Quantum Monte Carlo pricing of **best-of call options on 3 underlying assets** (extensible to d dimensions), using the **Modified Iterative Quantum Amplitude Estimation (MIQAE)** algorithm and Qiskit.

> Author: Rémi Machinal — Supervisor: Alain Bélanger — July 2025
> The full report (in French) is available in [`rapport_quantique.pdf`](./rapport_quantique.pdf).

---

## Overview

Classical Monte Carlo needs O(1/ε²) samples to reach precision ε, which becomes costly for high-dimensional or exotic products. Quantum Amplitude Estimation (QAE) reaches O(1/ε), a quadratic speedup. The original QAE relies on Quantum Phase Estimation and the Quantum Fourier Transform, which are too noisy for current hardware, so this project uses **MIQAE** (Fukuzawa et al., 2023), a QFT-free variant with proven asymptotic optimality.

The pricer targets the payoff

```
max(S1 − K1, S2 − K2, S3 − K3, 0)
```

with possibly different strikes, where the underlyings follow a **multivariate log-normal (Black–Scholes) model**. The payoff is encoded in the amplitude of a single objective qubit through:

- discretization of each future price on 2ⁿ values (n qubits per asset),
- strike loading, quantum subtractors (S − K) with carry qubits for the sign,
- quantum comparators (`QNMSub`) to select the largest of the three payoffs,
- controlled small rotations on the objective qubit (linear regime around π/4),
- MIQAE to estimate the amplitude, then rescaling to a monetary value and discounting by e^(−rT).

## Repository structure

```
.
├── pricing.py          # Script reproducing the report's experiments
├── qfinance/
│   ├── __init__.py
│   ├── helper.py            # Utility functions (e.g. loadNumber)
│   ├── option_pricing.py    # OptionParams, OptionPricing, BestOfCallPricer (3-asset payoff)
│   ├── qArithmetic.py       # Quantum subtraction / comparison gates
│   └── ModifiedIQAE/
│       └── mod_iae_updated.py   # ModifiedIterativeAmplitudeEstimation
└── README.md
```

## Implementation notes

- Code adapted from [jianjun-dot/quantum-finance](https://github.com/jianjun-dot/quantum-finance) (Chen, Li, Neufeld, 2024). **The modification is limited to the 3-dimensional payoff implementation in `option_pricing.py`.**
- MIQAE (`ModifiedIterativeAmplitudeEstimation`) follows Algorithms 1 (`estimate`) and 2 (`_find_next_k`) of Fukuzawa et al., with Chernoff or Clopper–Pearson confidence intervals and an early-stopping check.
- Qubit count for d assets with n qubits each: `(n+1)/2 · d² + (3+n)/2 · d + n` (30 qubits for d = 3, n = 3).

## Results

Averages over 10 runs, n = 3 qubits per asset, α = 0.05, ε = 0.005, r = 0.045, T = 40/365:

| S1, S2, S3     | Estimate | Exact | Confidence interval |
|----------------|----------|-------|---------------------|
| 2.7, 2.2, 2.5  | 0.543    | 0.181 | [0.320, 0.766]      |
| 2.0, 2.1, 1.9  | 0.163    | 0.149 | [0.014, 0.311]      |

With strikes K = (2.1, 1.9, 1.8), σᵢ = 0.4, r = 0.04 and S = (2, 2, 2): estimate 0.344 vs. exact 0.343.

## Limitations

- A discretization on only 8 values per asset is too coarse. Estimates degrade when initial asset prices are far apart, and rounding strikes to integers can lead to wrong comparisons between payoffs.
- The circuit is too deep (and needs ~30 qubits) for current NISQ devices, so results come from simulation. Practical use likely requires fault-tolerant hardware.
- Higher dimensions increase qubit count and controlled-gate depth quickly.

## Possible improvements

- Finer, better-chosen discretization range `[low, high]`
- Tuning of `ε`, `offset` and `step` according to hardware precision
- Replacing the log-normal gate `P(X)` with another asset model

## References

1. Brassard, Høyer, Mosca, Tapp — *Quantum Amplitude Amplification and Estimation*, Contemporary Mathematics 305 (2002)
2. Chen, Li, Neufeld — *Quantum Monte Carlo algorithm for solving Black–Scholes PDEs for high-dimensional option pricing in finance and its complexity analysis*, arXiv (2024)
3. Fukuzawa, Ho, Irani, Zion — *Modified Iterative Quantum Amplitude Estimation is Asymptotically Optimal*, ALENEX (2023)
4. Wang et al. — *Chebyshev-type early stopping for iterative quantum amplitude estimation*, arXiv:2304.01409 (2023)
5. Grinko, Gacon, Zoufal, Woerner — *Iterative Quantum Amplitude Estimation*, npj Quantum Information 7 (2021)
