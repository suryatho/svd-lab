# svd-lab

Truncated SVD written from scratch in Rust — power iteration with deflation, no
linear algebra library — applied as a lossy compressor across several data
domains to see where low-rank structure actually exists.

This is a learning project, written while working through Steve Brunton's
YouTube series
[Singular Value Decomposition (Data-Driven Science and Engineering)](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv)
alongside an optimization textbook. That series covers SVD across the whole
range of its applications — image compression through to PCA in control theory
— which is what suggested building one implementation and pointing it at
several different data domains rather than at just one.

It is not a general-purpose compressor; it is a low-rank approximation codec,
evaluated honestly, including the cases where it should perform badly.

## Status

| Milestone | State |
| --- | --- |
| `core::matrix` — flat row-major matrix, matvec / transpose-matvec | done |
| `core::svd` — power iteration + deflation, truncated SVD | done |
| `core::metrics` — Frobenius error, PSNR, compression ratio | done |
| `loaders::synthetic` — ground-truth matrices with a known spectrum | done |
| Correctness tests against Eckart-Young | done |
| `loaders::robot_traj` — MuJoCo qpos/qvel CSV | stub |
| `loaders::image` — uncompressed BMP | stub |
| `loaders::spectrogram`, `loaders::video` | stub, stretch goal |
| CLI, `std::simd` matvec + criterion bench, LaTeX writeup | not started |

## Layout

```
core/          # pure std linear algebra
  matrix.rs    #   storage layout + the two matvec kernels everything sits on
  svd.rs       #   the mathematics, documented against Eckart-Young
  metrics.rs   #   error, PSNR, compression ratio
  rng.rs       #   seeded xorshift64*, so runs are reproducible
loaders/       # one module per data domain -> Matrix
  synthetic.rs #   A = U diag(sigma) V^T + noise, with factors returned
  tests/       #   correctness gate: no real-data loader until these pass
```

## Approach

The leading singular triplet comes from maximizing the Rayleigh quotient of
`B = A^T A` — `B` is never formed, it is applied as two matrix-vector products,
`Bv = A^T(Av)`. Once `v` converges, `sigma = ||Av||` and `u = Av/sigma`.
Subtracting `sigma * u v^T` deflates the matrix so the next pass finds the next
triplet. `core/svd.rs` documents the derivation in full.

Correctness is checked against theory rather than against another library: by
Eckart-Young the rank-`k` truncation is the *optimal* rank-`k` approximation,
with residual exactly `sqrt(sum_{i>k} sigma_i^2)`. The tests construct matrices
with a chosen spectrum and confirm the measured residual matches that
prediction at every `k`. Singular vectors are compared sign-agnostically, since
`(sigma, u, v)` and `(sigma, -u, -v)` are the same rank-one term.

A negative control is included on purpose. A 120x120 i.i.d. Gaussian matrix has
a flat spectrum (`sigma_1/sigma_12 = 1.21`, rank-12 keeps 31% of the energy);
a geometrically decaying matrix of the same size keeps 99.90% at the same rank.
Low-rank structure is a property of the data, not of the method.

## Requirements

Nightly Rust, pinned by `rust-toolchain.toml`. Nightly is used only so that a
`std::simd` matvec can be added later; everything else is plain `std`, and the
core math has no crates.io dependencies by design.

```bash
cargo test
```
