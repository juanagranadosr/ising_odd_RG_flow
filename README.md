# ising_odd_RG_flow

Extension of [Di Carlo's IsingCG](https://github.com/lucadic/IsingCG) to track Z2-odd operators under renormalization group (RG) flow in the 2D Ising model. Developed as part of a master's thesis, *Symmetry Protection and the Hierarchy Problem: From Lattice Models to Goofy*, published by the Universitat de Barcelona: https://hdl.handle.net/2445/231453

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

**notebooks/MeasureRGFlow_odd_operators_template.ipynb** and **notebooks/runs/h0_\*/MeasureRGFlow_odd_operators.ipynb** (modified from Di Carlo's MeasureRGFlow.ipynb):

- All simulation and inference calls updated to pass h and h3.
- Loss landscapes plotted in the (K1, h), (K1, h3), and (h, h3) planes.
- RG flow tracked for h0 in {0, 1e-5, 1e-4, 1e-3, 5e-3} with h3 = 0. Each h0 value was run as a separate notebook (one per subfolder under `notebooks/runs/`) since the full run required Google Colab's compute rather than a local machine. `notebooks/MeasureRGFlow_odd_operators_template.ipynb` is the generic version (edit the `h` value near the top and rerun) used to produce each of these runs.

---

## Repository structure

    ising_odd_RG_flow/
    |-- README.md
    |-- CITATION.cff
    |-- LICENSE
    |-- requirements.txt
    |-- manuscript/                     # analytical work: full thesis and defense slides
    |   |-- TFM_JUANA_GRANADOS.pdf
    |   |-- TFM_present.pdf
    |-- src/
    |   |-- IsingRG_3spin.py           # extended Hamiltonian library
    |   |-- IsingRG.py                 # thin `from IsingRG_3spin import *` shim, see below
    |-- notebooks/
    |   |-- MeasureRGFlow_odd_operators_template.ipynb   # generic notebook: set h near the top and run
    |   |-- runs/                       # executed runs, one subfolder per h0 value, run on Google Colab
    |       |-- h0_0/MeasureRGFlow_odd_operators.ipynb
    |       |-- h0_1e-5/MeasureRGFlow_odd_operators.ipynb
    |       |-- h0_1e-4/MeasureRGFlow_odd_operators.ipynb
    |       |-- h0_1e-3/MeasureRGFlow_odd_operators.ipynb
    |       |-- h0_5e-3/MeasureRGFlow_odd_operators.ipynb
    |-- figures/                        # final figures used in the thesis
        |-- RG_flow.png
        |-- h.png
        |-- h3.png
        |-- ising_blocking.png
        |-- loss_landscapes/
        |-- GD_convergence/

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
| Compute environment | Google Colab (all runs executed there; not run locally) |

---

## How to run

**To inspect existing results:** each notebook under `notebooks/runs/h0_*/` already contains its output cells, so the RG flow results can be inspected directly on GitHub or by opening the file, without re-running anything.

**To reproduce or extend the results:** use `notebooks/MeasureRGFlow_odd_operators_template.ipynb`, the generic version of the notebook. Set `h` (and `h3`) near the top to the value you want to simulate and run the whole notebook — this is exactly how each of the `notebooks/runs/h0_*/` notebooks was produced. The full runs (L = 240, ~7000 gradient descent iterations per block size) were done on Google Colab; reproducing them locally requires comparable compute.

Every notebook loads the library with:

    ising_cg_path = '/content/drive/ColabNotebooks/IsingCG-main'  # <- change this
    sys.path.append(ising_cg_path)
    from IsingRG import *

`ising_cg_path` is a leftover from the original author's Drive layout. Point it at this repo's `src/` folder instead (e.g. `/content/ising_odd_RG_flow/src` on Colab after cloning, or the local absolute path to `src/`). `from IsingRG import *` will then resolve via `src/IsingRG.py`, a one-line shim that re-exports `src/IsingRG_3spin.py` — kept so the notebooks' import statements didn't need to be edited.

**On Google Colab** (used for the original runs, since local hardware wasn't powerful enough for them):

Open notebooks/MeasureRGFlow_odd_operators_template.ipynb directly in Colab, clone or upload this repository into the session (or mount your Drive), and update `ising_cg_path` as above to point at its `src/` folder.

**Locally:**

    git clone https://github.com/juanagranadosr/ising_odd_RG_flow.git
    cd ising_odd_RG_flow
    pip install -r requirements.txt
    jupyter notebook notebooks/MeasureRGFlow_odd_operators_template.ipynb

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

> J. Granados Rodríguez, *Symmetry Protection and the Hierarchy Problem: From Lattice Models to Goofy*, Master's thesis (advisor: Jordi Salvadó Serra), Universitat de Barcelona, 2026. https://hdl.handle.net/2445/231453

The published version is available from the Universitat de Barcelona's institutional repository (Dipòsit Digital de la UB) at the link above. A local copy, including the analytical derivations, is in [manuscript/TFM_JUANA_GRANADOS.pdf](manuscript/TFM_JUANA_GRANADOS.pdf); defense slides are in [manuscript/TFM_present.pdf](manuscript/TFM_present.pdf).

**Note on licensing:** the thesis document itself (the PDF and slides in `manuscript/`) is published under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) via the UB repository — noncommercial use, no derivatives, with attribution. This is separate from the MIT license covering the code in this repository (see [LICENSE](LICENSE)); only the code is MIT-licensed.
