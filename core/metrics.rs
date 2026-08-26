//! Quality and cost metrics for a low-rank approximation.
//!
//! Two axes matter for a codec: how much smaller the representation got
//! ([`compression_ratio`]) and how much was lost ([`frobenius_error`],
//! [`relative_error`], [`psnr`]). Sweeping `k` and recording both is what
//! produces the rate-distortion-style curves for the writeup.

use crate::matrix::Matrix;

/// Frobenius norm `||A||_F = sqrt(Σ_ij a_ij^2)`.
///
/// Equivalently `sqrt(Σ_i σ_i^2)` — the Frobenius norm is the 2-norm of the
/// singular value spectrum, which is why Eckart-Young's error formula is a
/// tail sum of squared singular values.
pub fn frobenius_norm(a: &Matrix) -> f64 {
    a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Absolute reconstruction error `||A - B||_F`.
///
/// # Panics
/// If the matrices have different shapes.
pub fn frobenius_error(a: &Matrix, b: &Matrix) -> f64 {
    assert_shapes_match(a, b);
    a.as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Relative reconstruction error `||A - B||_F / ||A||_F`.
///
/// This is the scale-free number to report and to plot against `k`. Returns
/// `0.0` when `A` is the zero matrix and `B` matches it, `INFINITY` otherwise.
///
/// # Panics
/// If the matrices have different shapes.
pub fn relative_error(a: &Matrix, b: &Matrix) -> f64 {
    let denom = frobenius_norm(a);
    let num = frobenius_error(a, b);
    if denom == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    num / denom
}

/// Mean squared error over all entries.
///
/// # Panics
/// If the matrices have different shapes, or are empty.
pub fn mse(a: &Matrix, b: &Matrix) -> f64 {
    assert_shapes_match(a, b);
    assert!(!a.is_empty(), "mse: matrices are empty");
    let sum: f64 = a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    sum / a.len() as f64
}

/// Peak signal-to-noise ratio in dB, for image-like data.
///
/// ```text
///     PSNR = 20 * log10(max_value / RMSE)
/// ```
///
/// `max_value` is the peak of the data's dynamic range, *not* the observed
/// maximum — pass `255.0` for 8-bit grayscale, `1.0` for normalized data.
/// Returns `INFINITY` for an exact reconstruction.
///
/// PSNR is only meaningful for the image domain; for robot trajectories or
/// synthetic matrices, use [`relative_error`] instead.
///
/// # Panics
/// If the matrices have different shapes, are empty, or `max_value <= 0`.
pub fn psnr(a: &Matrix, b: &Matrix, max_value: f64) -> f64 {
    assert!(max_value > 0.0, "psnr: max_value must be positive");
    let m = mse(a, b);
    if m == 0.0 {
        return f64::INFINITY;
    }
    20.0 * (max_value / m.sqrt()).log10()
}

/// Number of `f64`s needed to store a rank-`k` factorization of an `m x n`
/// matrix: `U` is `m x k`, `V^T` is `k x n`, plus the `k` singular values.
pub fn factored_storage(m: usize, n: usize, k: usize) -> usize {
    k * (m + n + 1)
}

/// Compression ratio of a rank-`k` factorization: dense elements divided by
/// factored elements.
///
/// ```text
///     ratio = (m * n) / (k * (m + n + 1))
/// ```
///
/// Greater than 1 means the factorization is smaller than the dense matrix.
/// Note this crosses below 1 once `k > mn / (m + n + 1)` — for a square
/// `n x n` matrix that is roughly `k > n/2`, so high-rank approximations of
/// square data are *larger* than just storing the matrix. Worth stating
/// plainly in the discussion section.
///
/// Returns `INFINITY` for `k == 0` (nothing stored).
pub fn compression_ratio(m: usize, n: usize, k: usize) -> f64 {
    if k == 0 {
        return f64::INFINITY;
    }
    (m * n) as f64 / factored_storage(m, n, k) as f64
}

/// The largest `k` at which a rank-`k` factorization still costs less than the
/// dense matrix. Useful for annotating sweep plots with a "break-even" line.
pub fn break_even_rank(m: usize, n: usize) -> usize {
    (m * n) / (m + n + 1)
}

/// Eckart-Young's predicted error for a rank-`k` truncation, given the full
/// singular value spectrum: `sqrt(Σ_{i>k} σ_i^2)`.
///
/// Comparing this against the measured [`frobenius_error`] validates the SVD
/// implementation — see `tests/svd_correctness.rs`.
pub fn tail_energy(spectrum: &[f64], k: usize) -> f64 {
    spectrum.iter().skip(k).map(|s| s * s).sum::<f64>().sqrt()
}

/// Fraction of total squared "energy" captured by the leading `k` singular
/// values: `Σ_{i≤k} σ_i^2 / Σ_i σ_i^2`.
///
/// This is the natural way to read a spectrum-decay plot: a matrix with real
/// low-rank structure reaches 0.99 at small `k`, noise does not.
///
/// Returns `1.0` for an all-zero spectrum.
pub fn energy_fraction(spectrum: &[f64], k: usize) -> f64 {
    let total: f64 = spectrum.iter().map(|s| s * s).sum();
    if total == 0.0 {
        return 1.0;
    }
    let kept: f64 = spectrum.iter().take(k).map(|s| s * s).sum();
    kept / total
}

fn assert_shapes_match(a: &Matrix, b: &Matrix) {
    assert_eq!(
        (a.rows(), a.cols()),
        (b.rows(), b.cols()),
        "shape mismatch: {}x{} vs {}x{}",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frobenius_norm_of_a_known_matrix() {
        // 3-4-5 in each row: ||A||_F = sqrt(25 + 25) = sqrt(50)
        let a = Matrix::from_rows(&[vec![3.0, 4.0], vec![4.0, -3.0]]);
        assert!((frobenius_norm(&a) - 50.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn identical_matrices_have_zero_error_and_infinite_psnr() {
        let a = Matrix::from_fn(4, 5, |i, j| (i * j) as f64);
        assert_eq!(frobenius_error(&a, &a), 0.0);
        assert_eq!(relative_error(&a, &a), 0.0);
        assert_eq!(psnr(&a, &a, 255.0), f64::INFINITY);
    }

    #[test]
    fn relative_error_is_scale_invariant() {
        let a = Matrix::from_fn(6, 6, |i, j| ((i + j) as f64).sin());
        let b = Matrix::from_fn(6, 6, |i, j| ((i + j) as f64).sin() * 0.9);
        let a10 = Matrix::from_fn(6, 6, |i, j| 10.0 * ((i + j) as f64).sin());
        let b10 = Matrix::from_fn(6, 6, |i, j| 10.0 * ((i + j) as f64).sin() * 0.9);
        assert!((relative_error(&a, &b) - relative_error(&a10, &b10)).abs() < 1e-12);
    }

    #[test]
    fn psnr_matches_the_definition() {
        // Every entry off by exactly 1 => MSE = 1, RMSE = 1.
        let a = Matrix::zeros(4, 4);
        let b = Matrix::from_fn(4, 4, |_, _| 1.0);
        let got = psnr(&a, &b, 255.0);
        let want = 20.0 * 255.0_f64.log10(); // ~48.13 dB
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    #[test]
    fn compression_ratio_matches_the_formula() {
        // 100x100 at k=5: 10000 / (5 * 201) = 9.950...
        let r = compression_ratio(100, 100, 5);
        assert!((r - 10000.0 / 1005.0).abs() < 1e-12);
        assert!(r > 1.0);
    }

    #[test]
    fn compression_ratio_drops_below_one_past_break_even() {
        let (m, n) = (100, 100);
        let be = break_even_rank(m, n);
        assert!(compression_ratio(m, n, be) >= 1.0);
        assert!(compression_ratio(m, n, be + 1) < 1.0);
    }

    #[test]
    fn zero_rank_costs_nothing() {
        assert_eq!(compression_ratio(10, 10, 0), f64::INFINITY);
        assert_eq!(factored_storage(10, 10, 0), 0);
    }

    #[test]
    fn tail_energy_and_fraction_are_complementary() {
        let spectrum = [4.0, 3.0, 2.0, 1.0]; // total energy 16+9+4+1 = 30
        assert!((tail_energy(&spectrum, 2) - 5.0_f64.sqrt()).abs() < 1e-12);
        assert!((energy_fraction(&spectrum, 2) - 25.0 / 30.0).abs() < 1e-12);
        assert!((energy_fraction(&spectrum, 4) - 1.0).abs() < 1e-12);
        assert_eq!(tail_energy(&spectrum, 4), 0.0);
    }
}
