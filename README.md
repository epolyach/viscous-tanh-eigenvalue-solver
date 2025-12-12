# Viscous tanh-flow eigenvalue solver (Legendre truncation)

This repository contains Matlab and C implementations for computing viscous eigenmodes of the hyperbolic-tangent shear flow
\(U(y)=\tanh(y)\) using a Legendre-polynomial expansion (mapping \(u=\tanh y\in[-1,1]\)) and a truncated generalized eigenvalue problem.

The formulation matches the “EIGENVALUE PROBLEM” section in `viscous_hydro3.pdf` (not included here). In particular, after expansion over Legendre polynomials,
the viscous Orr–Sommerfeld problem reduces to a sparse/banded generalized eigenproblem of the form

\[ \sigma\,\bar R\,\bar A = \bar Q\,\bar A + \mu\,\bar T\,\bar A \]

which is solved by shift/targeted methods.

## Repository layout

- `Matlab/`
  - `viscous_tanh_Legendre_Ver2.m` — double-precision solver (uses `eigs`).
  - `viscous_tanh_Legendre_Ver2_mp.m` — multiprecision solver (requires Advanpix Multiprecision Toolbox).
- `C/`
  - `viscous_eigenvalue_solver.c`, `viscous_eigenvalue_solver.h` — MPFR-based shift-and-invert / inverse-iteration solver.
  - `Makefile` — build helper.

## Matlab usage

1. Open `Matlab/viscous_tanh_Legendre_Ver2.m`.
2. Set parameters at the top of the script:
   - `k_value` (wavenumber),
   - `mu` (viscosity parameter),
   - `target_eigenvalue` (target \(\sigma\) for `eigs`),
   - `Nmax` (truncation size).
3. Run the script.

### Matlab outputs

When run from the `Matlab/` directory, the script writes:

- `EF_k=<k>_mu=<mu>_Nmax=<Nmax>_Precision=17.mat`
  - Matlab `result` struct including eigenvalue, coefficients, eigenfunction \(\psi\) and vorticity \(\zeta\).
- `EP-k=<k>_mu=<mu>_Nmax=<Nmax>_Precision=17.csv`
  - Columns: `y, u, Re\psi, Im\psi, Re\zeta, Im\zeta`.

The multiprecision script `viscous_tanh_Legendre_Ver2_mp.m` follows the same logic but uses Advanpix `mp` arithmetic.

## C usage

### Dependencies

- C compiler (gcc/clang)
- MPFR + GMP development libraries

On Ubuntu, you typically need:

- `libmpfr-dev`
- `libgmp-dev`

### Build

```bash
cd C
make
```

### Run

```bash
cd C
./viscous_eigenvalue_solver -k 1.05 -mu 1e-7 -t 0.03143596668 -n 40000
```

Options (see `-h`):

- `-k` wavenumber
- `-mu` viscosity
- `-t` “Landau” eigenvalue used as a base for the spectral shift (a real target \(\sigma\) is constructed internally)
- `-n` truncation size (`Nmax`)

### C outputs

The solver writes a file named like:

- `EVC_k=<k>_mu=<mu>_Nmax=<Nmax>_Precision_=<PREC>_Digits=<DIGITS>.dat`

The first line is the computed eigenvalue; subsequent lines are the eigenvector components (normalized by the first component if possible).

## Notes

- Convergence and runtime can be very sensitive to `mu`, the target/shift, and `Nmax`.
- For small `mu`, the required `Nmax` can become very large.
