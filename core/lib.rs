//! Core linear algebra for svd-lab: a flat dense matrix type, a truncated SVD
//! computed by power iteration with deflation, and the metrics used to judge
//! the resulting low-rank approximations.
//!
//! Pure `std`, no external crates. The nightly toolchain is pinned only so
//! that `matrix::matvec_simd` (a later milestone) can use `std::simd`.
//!
//! Reading order:
//! - [`matrix`] — storage layout and the two matvec kernels everything is
//!   built on.
//! - [`svd`] — the mathematics, documented against Eckart-Young and the
//!   Rayleigh quotient.
//! - [`metrics`] — error, PSNR, compression ratio.

pub mod matrix;
pub mod metrics;
pub mod rng;
pub mod svd;

pub use matrix::Matrix;
pub use svd::{SvdConfig, TruncatedSvd, truncated_svd, truncated_svd_with};
