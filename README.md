# ising_odd_RG_flow

Extension of [Di Carlo's IsingCG](https://github.com/lucadic/IsingCG) to track Z2-odd operators under renormalization group (RG) flow in the 2D Ising model. Developed as part of a master's thesis on the hierarchy problem and GOOFY symmetries.

---

## Attribution

This repository is an extension of:

> L. Di Carlo, *Coarse-graining the Ising model with the inverse Ising problem*, arXiv:2401.04811 (2024).
> Code: https://github.com/lucadic/IsingCG

The base method — Monte Carlo sampling, Kadanoff block-spin transformation, and pseudo-likelihood inverse Ising inference — is entirely due to Di Carlo. The present repository modifies two of his files (IsingRG.py and MeasureRGFlow.ipynb) to include Z2-odd operators in both the forward simulation and the inverse inference.

**I do not claim ownership of the original code.** My contribution is the extension described below.

---

## Physics motivation

The 2D Ising model has a global Z2 symmetry (sigma -> -sigma) which forbids odd operators such as a magnetic field h or a three-spin coupling h3. By Kadanoff's argument, symmetry-forbidden operators should remain zero under coarse-graining when the symmetry is exact, and grow when the symmetry is explicitly broken.

This code tests that argument numerically. The extended Hamiltonian is:

    H[sigma] = - sum_{d=1}^{4} K_d sum_{|i-j|=d} sigma_i sigma_j
               - h   sum_i sigma_i
               - h3  sum_{<ijk>} sigma_i sigma_j sigma_k

where the three-spin operator couples each spin to the four L-shaped triplets formed by one horizontal and one vertical neighbour. Both h and h3 are Z2-odd.

The procedure is:
1. Simulate the Ising model at near-critical couplings with a chosen value of h (and h3 = 0).
2. Apply Kadanoff blocking at block sizes b in {2, 3, 4, 5, 6, 8}.
3. Use pseudo-likelihood inference to extract the effective couplings (K, h, h3) of the blocked system.
4. Track how h and h3 evolve with b.

When h = 0 (Z2 exact), the inferred h and h3 remain close to zero at all block sizes. When h != 0 (Z2 broken), they grow with b, confirming that the symmetry is no longer protecting them.

---

## My contribution

Relative to Di Carlo's original, the following changes were made:

**src/IsingRG_3spin.py** (modified from Di Carlo's IsingRG.py):

- Make_MontecarloStep: added h and h3 to the local field used in the Metropolis acceptance step.
- Make_MonteCarloStepS: threads h and h3 through the JAX scan.
- Sample: threads h and h3 through the JAX scan.
- Pseudo_Loss: added h and h3 contributions to the local field H_n before computing the pseudo-likelihood.
- Pseudo_Loss_fn_and_grad: added gradients dF/dh and dF/dh3. Return signature extended from (loss, grad_K) to (loss, grad_K, grad_h, grad_h3).
- ComputeThreeSpinField (new): computes the three-spin field for all sites simultaneously using jnp.roll.

The gradients for the new couplings are:

    dF/dh  = -2/(Ns L^2) sum_{t,n} tau_n^t
    dF/dh3 = -2/(Ns L^2) sum_{t,n} Theta_n(t) tau_n^t

where tau_n^t = sigma_n(t) / (1 + exp(2 H_n sigma_n(t))).

**notebooks/MeasureRGFlow_odd_operators.ipynb** (modified from Di Carlo's MeasureRGFlow.ipynb):

- All simulation and inference calls updated to pass h and h3.
- Loss landscapes plotted in the (K1, h), (K1, h3), and (h, h3) planes.
- RG flow tracked for h0 in {0, 1e-5, 1e-4, 1e-3, 5e-3} with h3 = 0.

---

## Repository structure

    ising_odd_RG_flow/
    |-- README.md
    |-- CITATION.cff
    |-- LICENSE
    |-- requirements.txt
    |-- src/
    |   |-- IsingRG_3spin.py
    |-- notebooks/
    |   |-- MeasureRGFlow_odd_operators.ipynb
    |-- figures/

---

## Simulation parameters

| Parameter | Value |
|-----------|-------|
| Lattice size | L = 240 (square, periodic boundary conditions) |
| Near-critical couplings | K1 = 0.203, K2 = 0.078 |
| Equilibrium configurations | Ns = 300 |
| Block sizes | b in {2, 3, 4, 5, 6, 8} |
| Learning rate | eta = 0.025 |
| Gradient descent iterations | ~7000 per block size |
| Values of h0 simulated | 0, 1e-5, 1e-4, 1e-3, 5e-3 |

---

## How to run

**On Google Colab (recommended):**

Open notebooks/MeasureRGFlow_odd_operators.ipynb directly in Colab. Upload src/IsingRG_3spin.py to the Colab session or mount your Drive. No local installation needed.

**Locally:**

    git clone https://github.com/juanagranadosr/ising_odd_RG_flow.git
    cd ising_odd_RG_flow
    pip install -r requirements.txt
    jupyter notebook notebooks/MeasureRGFlow_odd_operators.ipynb

---

## Reference

If you use this code, please cite Di Carlo's original work:

    @article{dicarlo2024,
      author  = {Di Carlo, Luca},
      title   = {Coarse-graining the Ising model with the inverse Ising problem},
      year    = {2024},
      url     = {https://arxiv.org/abs/2401.04811}
    }

See CITATION.cff for how to cite this repository.

---

## Thesis

This code was developed as part of a master's thesis:

> J. Granados, *The hierarchy problem and GOOFY symmetries*, Master's thesis, Universitat de Barcelona, 2025.

(Link to be added upon publication.)
