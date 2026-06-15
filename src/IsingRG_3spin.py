"""
IsingRG_3spin.py
================
Extension of Di Carlo's IsingRG.py (https://github.com/lucadic/IsingCG)
to include two Z2-odd operators in the Ising Hamiltonian:

    H[σ] = -Σ_{d=1}^{4} K_d Σ_{|i-j|=d} σ_i σ_j
            - h   Σ_i σ_i
            - h_3 Σ_{<ijk>} σ_i σ_j σ_k

where h is a uniform magnetic field (single-spin operator) and h_3
couples each spin to the four L-shaped triplets formed by one
horizontal and one vertical neighbour.

Under the Z2 symmetry σ → -σ of the standard Ising model, both h
and h_3 are forbidden (they are Z2-odd). The goal of this code is
to track these operators under Kadanoff block-spin renormalization
group flow, and verify numerically that they remain protected when
Z2 is exact and grow when Z2 is explicitly broken.

Attribution
-----------
The base code (Monte Carlo sampling, Kadanoff blocking, and
pseudo-likelihood inverse Ising inference) is due to:

    L. Di Carlo, arXiv:2401.04811 (2024)
    https://github.com/lucadic/IsingCG

Modified functions (relative to Di Carlo's original):
    - Make_MontecarloStep      : added h and h3 to the local field
    - Make_MonteCarloStepS     : threads h and h3 through the scan
    - Sample                   : threads h and h3 through the scan
    - Pseudo_Loss              : added h and h3 contributions
    - Pseudo_Loss_fn_and_grad  : added gradients w.r.t. h and h3

New functions added:
    - ComputeThreeSpinField    : computes the three-spin field for
                                 all sites simultaneously using
                                 jnp.roll

Unmodified functions (reproduced from Di Carlo's original):
    - ComputeLocalField
    - Apply_Majority_Rule
    - Block_configuration
    - PadConfiguration
    - build_filter
    - ComputeLocalFields

Author of extension: Juana Granados
Repository: https://github.com/juanagranadosr/ising_odd_RG_flow
"""

import jax
import jax.numpy as jnp
from flax.linen import avg_pool
from functools import partial
from jax.scipy.signal import convolve2d


# ==============================================================
# UNMODIFIED FUNCTIONS (from Di Carlo's original IsingRG.py)
# ==============================================================

def ComputeLocalField(configuration, pos, d):
    """
    Compute the sum of spins at Manhattan distance d from position pos.

    This is used in the single-spin-flip Metropolis step to evaluate
    the local field acting on one spin at a time.

    Parameters
    ----------
    configuration : jnp.ndarray, shape (L, L)
        Current spin configuration.
    pos : tuple of int
        (x0, y0) position of the spin being considered for a flip.
    d : int
        Interaction distance (d=1 nearest neighbours, d=2 next-nearest, ...).

    Returns
    -------
    h : float
        Sum of all spins at Manhattan distance d from pos,
        with periodic boundary conditions.

    Notes
    -----
    Unchanged from Di Carlo's original.
    """
    x0, y0 = pos
    L = configuration.shape[-1]
    h = 0
    for cursor in range(1, d):
        h += configuration[(x0 + cursor) % L, (y0 + d - cursor + L) % L]
        h += configuration[(x0 - cursor) % L, (y0 + d - cursor + L) % L]
        h += configuration[(x0 + cursor) % L, (y0 - d + cursor + L) % L]
        h += configuration[(x0 - cursor) % L, (y0 - d + cursor + L) % L]
    h += configuration[(x0 + d) % L, y0]
    h += configuration[(x0 - d + L) % L, y0]
    h += configuration[x0, (y0 + d) % L]
    h += configuration[x0, (y0 - d + L) % L]
    return h


# ==============================================================
# MODIFIED FUNCTIONS (extended from Di Carlo's original)
# ==============================================================

@jax.jit
def Make_MontecarloStep(configuration, rng, K, h, h3):
    """
    Perform one single-spin-flip Metropolis step on the configuration.

    Proposes flipping a randomly chosen spin and accepts or rejects
    the flip according to the Metropolis criterion. The local energy
    change includes contributions from spin-spin couplings K, a
    uniform magnetic field h, and a three-spin coupling h3.

    The full local field acting on spin σ_n is:

        H_n = Σ_d K_d * (sum of spins at distance d from n)
              + h
              + h3 * (σ_{n+x̂}σ_{n+ŷ} + σ_{n-x̂}σ_{n+ŷ}
                      + σ_{n+x̂}σ_{n-ŷ} + σ_{n-x̂}σ_{n-ŷ})

    Parameters
    ----------
    configuration : jnp.ndarray, shape (L, L)
        Current spin configuration with values in {-1, +1}.
    rng : jax.random.PRNGKey
        JAX random key for this step.
    K : jnp.ndarray, shape (num_distances,)
        Spin-spin coupling constants. K[d-1] is the coupling at
        Manhattan distance d.
    h : float
        Uniform magnetic field (Z2-odd, single-spin operator).
        Set to 0.0 to recover the Z2-symmetric Ising model.
    h3 : float
        Three-spin coupling constant (Z2-odd).
        Set to 0.0 to recover the Z2-symmetric Ising model.

    Returns
    -------
    configuration : jnp.ndarray, shape (L, L)
        Updated spin configuration after the Metropolis step.
    magnetization : float
        Mean magnetization of the updated configuration.

    Notes
    -----
    Modified from Di Carlo's original: added h and h3 contributions
    to the local field H before computing the flip probability.
    """
    L = configuration.shape[-1]
    rng_pos, rng_p_acc = jax.random.split(rng, 2)
    pos = tuple(jax.random.randint(rng_pos, (2,), minval=0, maxval=L))
    p = jax.random.uniform(rng_p_acc)

    # Spin-spin coupling contribution (Di Carlo's original)
    H = jnp.zeros((1,))
    for distance in range(len(K)):
        H += K[distance] * ComputeLocalField(configuration, pos, distance + 1)

    # Magnetic field contribution [ADDED: not in Di Carlo's original]
    H += h

    # Three-spin coupling contribution [ADDED: not in Di Carlo's original]
    x0, y0 = pos
    H += h3 * (configuration[(x0+1)%L, y0] * configuration[x0, (y0+1)%L]
          + configuration[(x0-1)%L, y0] * configuration[x0, (y0+1)%L]
          + configuration[(x0+1)%L, y0] * configuration[x0, (y0-1)%L]
          + configuration[(x0-1)%L, y0] * configuration[x0, (y0-1)%L])

    flip_probability = jnp.exp(-2 * H * configuration[pos])
    configuration = jnp.where(
        p < flip_probability,
        configuration.at[pos].multiply(-1),
        configuration
    )
    return configuration, configuration.sum() / (L * L)


def Make_MonteCarloStepS(configuration, rng, num_flips, K, h, h3):
    """
    Evolve the spin configuration for num_flips Metropolis steps.

    Applies Make_MontecarloStep repeatedly using jax.lax.scan
    for efficiency.

    Parameters
    ----------
    configuration : jnp.ndarray, shape (L, L)
        Current spin configuration.
    rng : jax.random.PRNGKey
        JAX random key.
    num_flips : int
        Number of single-spin-flip attempts.
    K : jnp.ndarray
        Spin-spin coupling constants.
    h : float
        Uniform magnetic field (Z2-odd).
    h3 : float
        Three-spin coupling constant (Z2-odd).

    Returns
    -------
    configuration : jnp.ndarray, shape (L, L)
        Configuration after num_flips steps.
    configuration : jnp.ndarray, shape (L, L)
        Same configuration (returned twice for jax.lax.scan
        compatibility).

    Notes
    -----
    Modified from Di Carlo's original: h and h3 threaded through
    the partial call to Make_MontecarloStep.
    """
    rng_new, rng = jax.random.split(rng, 2)
    rng_s = jax.random.split(rng_new, num_flips)
    configuration, rng = jax.lax.scan(
        partial(Make_MontecarloStep, K=K, h=h, h3=h3),
        configuration, rng_s
    )
    return configuration, configuration


def Sample(initial_configuration, rng, num_flips, num_samples, K, h, h3):
    """
    Sample equilibrium configurations from the Ising model.

    Starting from initial_configuration, runs the Metropolis chain
    and returns num_samples configurations, each separated by
    num_flips single-spin-flip attempts.

    Parameters
    ----------
    initial_configuration : jnp.ndarray, shape (L, L)
        Starting spin configuration.
    rng : jax.random.PRNGKey
        JAX random key.
    num_flips : int
        Number of Metropolis steps between recorded samples.
        Should be large enough for decorrelation (typically ~ 30*L*L).
    num_samples : int
        Number of configurations to return.
    K : jnp.ndarray
        Spin-spin coupling constants.
    h : float
        Uniform magnetic field (Z2-odd).
        Set to 0.0 for the standard Z2-symmetric Ising model.
    h3 : float
        Three-spin coupling constant (Z2-odd).
        Set to 0.0 for the standard Z2-symmetric Ising model.

    Returns
    -------
    sampled_configurations : jnp.ndarray, shape (num_samples, L, L)
        Array of sampled spin configurations.

    Notes
    -----
    Modified from Di Carlo's original: h and h3 threaded through
    the partial call to Make_MonteCarloStepS.
    """
    rng_sampling = jax.random.split(rng, num_samples)
    final_configuration, sampled_configurations = jax.lax.scan(
        partial(Make_MonteCarloStepS, K=K, h=h, h3=h3, num_flips=num_flips),
        initial_configuration, rng_sampling
    )
    return sampled_configurations


# ==============================================================
# UNMODIFIED FUNCTIONS (from Di Carlo's original IsingRG.py)
# ==============================================================

@jax.jit
def Apply_Majority_Rule(x, rng):
    """
    Apply the majority rule to a single blocked spin value.

    Returns +1 if x > 0, -1 if x < 0, and a random sign if x == 0.

    Parameters
    ----------
    x : float
        Average spin value in a block.
    rng : jax.random.PRNGKey
        Random key used to break ties when x == 0.

    Returns
    -------
    jnp.ndarray
        Blocked spin value in {-1, +1}.

    Notes
    -----
    Unchanged from Di Carlo's original.
    """
    return jnp.where(x > 0, 1,
                     jnp.where(x == 0,
                                jax.random.randint(rng, (1,), minval=0, maxval=2) * 2 - 1,
                                -1))


def Block_configuration(s, block_size, rng):
    """
    Apply Kadanoff block-spin transformation to a set of configurations.

    Divides each L x L configuration into non-overlapping blocks of
    size block_size x block_size, computes the average spin in each
    block, and applies the majority rule to obtain a coarse-grained
    configuration of size (L/block_size) x (L/block_size).

    Parameters
    ----------
    s : jnp.ndarray, shape (num_samples, L, L)
        Array of spin configurations to coarse-grain.
    block_size : int
        Linear size of each block. Must divide L exactly.
    rng : jax.random.PRNGKey
        Random key for tie-breaking in Apply_Majority_Rule.

    Returns
    -------
    blocked_majority : jnp.ndarray, shape (num_samples, L//block_size, L//block_size)
        Coarse-grained spin configurations.

    Notes
    -----
    Unchanged from Di Carlo's original.
    """
    L = s.shape[-1]
    if s.shape[-2] != L:
        print("The system must be square size")
        return
    if L % block_size != 0:
        print("The block size is not compatible with the system size")
        return
    Block_shape = (block_size, block_size)
    blocked = avg_pool(
        s.transpose(1, 2, 0),
        window_shape=Block_shape,
        strides=Block_shape
    ).transpose(2, 0, 1)

    B, L1, L2 = blocked.shape
    blocked_flat = blocked.reshape(B * L1 * L2)
    rng_s = jax.random.split(rng, B * L1 * L2)
    blocked_flat_majority = jax.vmap(
        Apply_Majority_Rule, in_axes=(0, 0), out_axes=(0)
    )(blocked_flat, rng_s)
    blocked_majority = blocked_flat_majority.reshape((B, L1, L2))
    return blocked_majority


def PadConfiguration(s, padding):
    """
    Pad a single configuration with periodic boundary conditions.

    Parameters
    ----------
    s : jnp.ndarray, shape (L, L)
        Spin configuration to pad.
    padding : int
        Number of sites to add on each side.

    Returns
    -------
    padded_lattice : jnp.ndarray, shape (L + 2*padding, L + 2*padding)
        Periodically padded configuration.

    Notes
    -----
    Unchanged from Di Carlo's original.
    """
    upper_pad = s[:, -padding:]
    lower_pad = s[:, :padding]
    partial_lattice = jnp.concatenate([upper_pad, s, lower_pad], axis=1)
    left_pad = partial_lattice[-padding:, :]
    right_pad = partial_lattice[:padding, :]
    padded_lattice = jnp.concatenate([left_pad, partial_lattice, right_pad], axis=0)
    return padded_lattice


@jax.jit
def build_filter(K):
    """
    Build a convolutional filter encoding the spin-spin couplings K.

    The filter has size (2*len(K)+1) x (2*len(K)+1). Entry (i,j)
    contains K[d-1] if the Manhattan distance from the centre is d,
    and 0 otherwise.

    Example for K = [k1, k2]:
        [[0   0   k2  0   0 ],
         [0   k2  k1  k2  0 ],
         [k2  k1  0   k1  k2],
         [0   k2  k1  k2  0 ],
         [0   0   k2  0   0 ]]

    Parameters
    ----------
    K : jnp.ndarray
        Coupling constants. K[d-1] is the coupling at distance d.

    Returns
    -------
    Filter : jnp.ndarray, shape (2*len(K)+1, 2*len(K)+1)
        Convolutional filter for computing local fields.

    Notes
    -----
    Unchanged from Di Carlo's original.
    """
    D_max = len(K)
    Filter = jnp.zeros((len(K) * 2 + 1, len(K) * 2 + 1))
    Lf, Lf = Filter.shape
    for i in range(Lf):
        for j in range(Lf):
            distance = abs(i - D_max) + abs(j - D_max)
            if 0 < distance <= len(K):
                Filter = Filter.at[i, j].set(K[distance - 1])
    return Filter


@jax.jit
def ComputeLocalFields(padded_lattice, filter):
    """
    Compute the local spin-spin field at every site simultaneously.

    Uses 2D convolution with the coupling filter to compute
    H_n = Σ_d K_d Σ_{|m-n|=d} σ_m  for all sites n at once.

    Parameters
    ----------
    padded_lattice : jnp.ndarray, shape (L + 2*padding, L + 2*padding)
        Periodically padded spin configuration.
    filter : jnp.ndarray
        Convolutional filter from build_filter(K).

    Returns
    -------
    jnp.ndarray, shape (L, L)
        Local spin-spin field at each site.

    Notes
    -----
    Unchanged from Di Carlo's original.
    """
    return convolve2d(padded_lattice, filter, mode='valid')


# ==============================================================
# NEW FUNCTION (added in this extension, not in Di Carlo's original)
# ==============================================================

def ComputeThreeSpinField(St):
    """
    Compute the three-spin field Θ_n for all sites simultaneously.

    For each site n, Θ_n is the sum of products of pairs of
    orthogonal neighbours forming L-shaped triplets:

        Θ_n = σ_{n+x̂} σ_{n+ŷ}
              + σ_{n-x̂} σ_{n+ŷ}
              + σ_{n+x̂} σ_{n-ŷ}
              + σ_{n-x̂} σ_{n-ŷ}

    The three-spin interaction energy at site n is:
        h3 * σ_n * Θ_n

    so Θ_n is the effective field conjugate to σ_n from the
    three-spin operator (after factoring out σ_n itself).

    Uses jnp.roll for efficient vectorised computation over all
    sites with periodic boundary conditions.

    Parameters
    ----------
    St : jnp.ndarray, shape (num_samples, L, L)
        Array of spin configurations.

    Returns
    -------
    jnp.ndarray, shape (num_samples, L, L)
        Three-spin field Θ_n at every site n, for every sample.

    Notes
    -----
    New function, not present in Di Carlo's original.
    Added to support the pseudo-likelihood inference of h3.
    """
    right = jnp.roll(St, -1, axis=2)  # σ_{n+x̂}
    left  = jnp.roll(St, +1, axis=2)  # σ_{n-x̂}
    up    = jnp.roll(St, -1, axis=1)  # σ_{n+ŷ}
    down  = jnp.roll(St, +1, axis=1)  # σ_{n-ŷ}
    return right*up + left*up + right*down + left*down


# ==============================================================
# MODIFIED FUNCTIONS (extended from Di Carlo's original)
# ==============================================================

@jax.jit
def Pseudo_Loss(K, h, h3, St):
    """
    Compute the pseudo-likelihood loss function F = -Σ_n f_n.

    The pseudo-likelihood loss is:

        F = - (1/N_s L^2) Σ_{t,n} log P(σ_n(t) | σ_{\\n}(t))

    where the conditional probability is:

        P(σ_n | σ_{\\n}) = sigmoid(2 σ_n H_n)

    and the local field H_n includes spin-spin, magnetic field,
    and three-spin contributions:

        H_n = Σ_d K_d * (local spin-spin field)
              + h
              + h3 * Θ_n

    Minimising F over (K, h, h3) solves the inverse Ising problem:
    it finds the couplings that best explain the observed configurations.

    Parameters
    ----------
    K : jnp.ndarray
        Spin-spin coupling constants.
    h : float
        Uniform magnetic field (Z2-odd).
    h3 : float
        Three-spin coupling constant (Z2-odd).
    St : jnp.ndarray, shape (num_samples, L, L)
        Array of observed spin configurations.

    Returns
    -------
    float
        Value of the pseudo-likelihood loss F.

    Notes
    -----
    Modified from Di Carlo's original: added h and h3 contributions
    to H_n_t before computing the normalization.
    """
    Filter = build_filter(K)
    padded_configurations = jax.vmap(PadConfiguration, in_axes=(0, None))(St, len(K))
    H_n_t = jax.vmap(ComputeLocalFields, in_axes=(0, None), out_axes=0)(
        padded_configurations, Filter
    )
    H_n_t += h                              # magnetic field [ADDED]
    H_n_t += h3 * ComputeThreeSpinField(St) # three-spin operator [ADDED]

    normalization = 1. / (1 + jnp.exp(-2 * jnp.multiply(H_n_t, St)))
    ln = jnp.average(jnp.log(normalization), axis=1)
    return -jnp.average(ln)


@jax.jit
def Pseudo_Loss_fn_and_grad(K, St, h, h3):
    """
    Compute the pseudo-likelihood loss and its gradients.

    Returns F and the gradients ∂F/∂K_d, ∂F/∂h, ∂F/∂h3, which
    are used in gradient descent to solve the inverse Ising problem.

    The gradients are derived analytically. Defining:

        τ_n^t = σ_n(t) / (1 + exp(2 H_n σ_n(t)))

    the gradients are:

        ∂F/∂K_d  = -2/(N_s L^2) Σ_{t,n} [Σ_{|m-n|=d} σ_m(t)] τ_n^t
        ∂F/∂h    = -2/(N_s L^2) Σ_{t,n} τ_n^t
        ∂F/∂h3   = -2/(N_s L^2) Σ_{t,n} Θ_n(t) τ_n^t

    where Θ_n(t) is the three-spin field from ComputeThreeSpinField.

    Parameters
    ----------
    K : jnp.ndarray
        Spin-spin coupling constants.
    St : jnp.ndarray, shape (num_samples, L, L)
        Array of observed spin configurations.
    h : float
        Uniform magnetic field (Z2-odd).
    h3 : float
        Three-spin coupling constant (Z2-odd).

    Returns
    -------
    loss : float
        Value of the pseudo-likelihood loss F.
    grad_K : jnp.ndarray, shape (len(K),)
        Gradient of F with respect to each K_d.
    grad_h : float
        Gradient of F with respect to h.
    grad_h3 : float
        Gradient of F with respect to h3.

    Notes
    -----
    Modified from Di Carlo's original: added computation of
    grad_h and grad_h3, and added h and h3 contributions to H_n_t.
    The return signature is extended from (loss, grad_K) to
    (loss, grad_K, grad_h, grad_h3).
    """
    Filter = build_filter(K)
    padded_configurations = jax.vmap(PadConfiguration, in_axes=(0, None))(St, len(K))
    H_n_t = jax.vmap(ComputeLocalFields, in_axes=(0, None), out_axes=0)(
        padded_configurations, Filter
    )
    H_n_t += h                               # magnetic field [ADDED]
    three_spin_field = ComputeThreeSpinField(St)
    H_n_t += h3 * three_spin_field           # three-spin operator [ADDED]

    normalization_1 = 1. / (1 + jnp.exp(-2 * jnp.multiply(H_n_t, St)))
    ln = jnp.average(jnp.log(normalization_1), axis=1)

    # τ_n^t = σ_n / (1 + exp(2 H_n σ_n))
    normalization_2 = 1. / (1 + jnp.exp(2 * jnp.multiply(H_n_t, St)))
    temp = jnp.multiply(normalization_2, St)

    # Gradient w.r.t. each K_d (Di Carlo's original formula)
    grad = []
    for d in range(len(K)):
        _uniform_couplings = [0 for i in range(d + 1)]
        _uniform_couplings[-1] = 1
        Flat_filter_d = build_filter(_uniform_couplings)
        padded_configurations = jax.vmap(
            PadConfiguration, in_axes=(0, None)
        )(St, d + 1)
        Avg_d = jax.vmap(
            ComputeLocalFields, in_axes=(0, None), out_axes=0
        )(padded_configurations, Flat_filter_d)
        grad_d = -2 * jnp.average(jnp.multiply(Avg_d, temp))
        grad.append(grad_d)

    # Gradient w.r.t. h [ADDED]
    grad_h = -2 * jnp.average(temp)

    # Gradient w.r.t. h3 [ADDED]
    grad_h3 = -2 * jnp.average(jnp.multiply(three_spin_field, temp))

    return jnp.average(jnp.array(-ln)), jnp.array(grad), jnp.array(grad_h), jnp.array(grad_h3)
