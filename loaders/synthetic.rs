//! Synthetic matrices with known low-rank structure.
//!
//! This is the ground-truth domain: we *construct* `A = U diag(σ) V^T + E`
//! with orthonormal `U`, `V` and a spectrum `σ` we chose, so the true answer
//! is known exactly and the SVD implementation can be checked against it
//! rather than against another library.
//!
//! Building the factors orthonormal (rather than just filling `U` and `V` with
//! random entries and taking `U V^T`) is what makes the singular values known:
//! if `U^T U = I` and `V^T V = I`, then `U diag(σ) V^T` *is* a singular value
//! decomposition, so its singular values are exactly the `σ` we picked. With
//! random non-orthonormal factors the product still has rank `r`, but its
//! spectrum is whatever it happens to be — fine for a rank test, useless for a
//! singular value test.
//!
//! Adding noise `E` breaks that exactness on purpose: a noisy matrix is
//! full-rank, its leading `r` singular values are perturbed slightly upward,
//! and the remaining `min(m,n) - r` sit at the noise floor instead of zero.
//! Weyl's inequality bounds the perturbation by `|σ_i(A + E) - σ_i(A)| ≤
//! ||E||_2`, which is what the noisy test asserts against.

use svd_core::Matrix;
use svd_core::matrix;
use svd_core::rng::Rng;

/// A generated matrix together with the factors it was built from.
#[derive(Clone, Debug)]
pub struct Synthetic {
    /// The generated `m x n` matrix, `U diag(σ) V^T + E`.
    pub a: Matrix,
    /// Ground-truth left factor, `m x rank`, orthonormal columns.
    pub u: Matrix,
    /// Ground-truth singular values, length `rank`, descending. Exact for
    /// `A` only when `noise_stddev == 0.0`.
    pub singular_values: Vec<f64>,
    /// Ground-truth right factor, `n x rank`, orthonormal columns.
    /// (Note: `V`, not `V^T` — column `i` is `v_i`.)
    pub v: Matrix,
    /// Standard deviation of the i.i.d. Gaussian noise that was added.
    pub noise_stddev: f64,
}

impl Synthetic {
    /// The true rank of the noiseless part.
    pub fn rank(&self) -> usize {
        self.singular_values.len()
    }

    /// The noiseless matrix `U diag(σ) V^T`, rebuilt from the stored factors.
    pub fn clean(&self) -> Matrix {
        svd_core::svd::reconstruct(&self.u, &self.singular_values, &self.v.transpose())
    }

    /// Ground-truth left singular vector `i`.
    pub fn left_vector(&self, i: usize) -> Vec<f64> {
        self.u.column(i)
    }

    /// Ground-truth right singular vector `i`.
    pub fn right_vector(&self, i: usize) -> Vec<f64> {
        self.v.column(i)
    }
}

/// Generate `A = U diag(σ) V^T + E` with a caller-chosen spectrum.
///
/// `sigmas` should be positive and descending; it is sorted descending
/// defensively so the ground truth always matches the convention the SVD
/// returns. `noise_stddev` is the standard deviation of i.i.d. `N(0, s^2)`
/// entries added to every element — pass `0.0` for an exactly rank-`r` matrix.
///
/// # Panics
/// If `sigmas.len() > min(m, n)`, if any sigma is negative, or if either
/// dimension is zero.
pub fn with_spectrum(
    m: usize,
    n: usize,
    sigmas: &[f64],
    noise_stddev: f64,
    seed: u64,
) -> Synthetic {
    assert!(m > 0 && n > 0, "dimensions must be non-zero");
    assert!(
        sigmas.len() <= m.min(n),
        "rank {} exceeds min(m, n) = {}",
        sigmas.len(),
        m.min(n)
    );
    assert!(
        sigmas.iter().all(|&s| s >= 0.0),
        "singular values must be non-negative"
    );
    assert!(noise_stddev >= 0.0, "noise stddev must be non-negative");

    let rank = sigmas.len();
    let mut ordered = sigmas.to_vec();
    ordered.sort_by(|a, b| b.partial_cmp(a).expect("singular values must not be NaN"));

    let mut rng = Rng::new(seed);
    let u = random_orthonormal(m, rank, &mut rng);
    let v = random_orthonormal(n, rank, &mut rng);

    // A = Σ_i σ_i u_i v_i^T, accumulated one rank-one term at a time.
    let mut a = Matrix::zeros(m, n);
    for (i, &s) in ordered.iter().enumerate() {
        a.add_rank_one(s, &u.column(i), &v.column(i));
    }

    if noise_stddev > 0.0 {
        for x in a.as_mut_slice() {
            *x += noise_stddev * rng.next_normal();
        }
    }

    Synthetic {
        a,
        u,
        singular_values: ordered,
        v,
        noise_stddev,
    }
}

/// Generate a rank-`rank` matrix with a default well-separated spectrum
/// `σ_i = rank - i` (so `rank, rank-1, ..., 1`).
///
/// The gaps matter: power iteration converges at rate `(σ_{i+1}/σ_i)^2`, so a
/// clearly separated spectrum keeps the tests fast and unambiguous. Use
/// [`with_spectrum`] directly to probe the slow, nearly-degenerate case.
pub fn low_rank(m: usize, n: usize, rank: usize, noise_stddev: f64, seed: u64) -> Synthetic {
    let sigmas: Vec<f64> = (0..rank).map(|i| (rank - i) as f64).collect();
    with_spectrum(m, n, &sigmas, noise_stddev, seed)
}

/// Generate a matrix with a geometrically decaying spectrum,
/// `σ_i = scale * decay^i`.
///
/// This is the interesting middle case for the writeup: mathematically
/// full-rank, but numerically compressible, which is what real data
/// (images, trajectories) actually looks like. `decay` should be in `(0, 1)`.
///
/// # Panics
/// If `decay` is not in `(0, 1)` or `scale` is not positive.
pub fn geometric_decay(
    m: usize,
    n: usize,
    rank: usize,
    scale: f64,
    decay: f64,
    noise_stddev: f64,
    seed: u64,
) -> Synthetic {
    assert!(decay > 0.0 && decay < 1.0, "decay must lie in (0, 1)");
    assert!(scale > 0.0, "scale must be positive");
    let sigmas: Vec<f64> = (0..rank).map(|i| scale * decay.powi(i as i32)).collect();
    with_spectrum(m, n, &sigmas, noise_stddev, seed)
}

/// A pure i.i.d. Gaussian matrix — the *incompressible* control case.
///
/// Its spectrum follows the Marchenko-Pastur law: spread across a broad band
/// with no sharp decay, so no rank-`k` truncation is a good approximation.
/// Included deliberately, since the point of the project is to show where SVD
/// fails as well as where it works.
pub fn noise(m: usize, n: usize, stddev: f64, seed: u64) -> Matrix {
    let mut rng = Rng::new(seed);
    Matrix::from_fn(m, n, |_, _| stddev * rng.next_normal())
}

/// Draw an `n x k` matrix with orthonormal columns.
///
/// Fills with Gaussians (whose column span is uniformly distributed) and
/// orthonormalizes by modified Gram-Schmidt. The classical algorithm loses
/// orthogonality when columns are nearly dependent, so each column is
/// projected twice — the "twice is enough" result: one reorthogonalization
/// pass restores orthogonality to machine precision.
///
/// # Panics
/// If `k > n`.
fn random_orthonormal(n: usize, k: usize, rng: &mut Rng) -> Matrix {
    assert!(
        k <= n,
        "cannot have {k} orthonormal columns in dimension {n}"
    );
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);

    for _ in 0..k {
        // Redraw if a column collapses during orthogonalization (measure-zero
        // in theory, cheap to guard against in practice).
        let col = loop {
            let mut c = rng.normal_vec(n);

            for _pass in 0..2 {
                for prev in &cols {
                    let proj = matrix::dot(&c, prev);
                    for (ci, &pi) in c.iter_mut().zip(prev.iter()) {
                        *ci -= proj * pi;
                    }
                }
            }

            if matrix::normalize(&mut c) > 1e-8 {
                break c;
            }
        };
        cols.push(col);
    }

    let mut q = Matrix::zeros(n, k);
    for (j, col) in cols.iter().enumerate() {
        q.set_column(j, col);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use svd_core::matrix::matmul;

    #[test]
    fn generated_factors_are_orthonormal() {
        let s = low_rank(40, 25, 6, 0.0, 11);
        for (name, f) in [("U", &s.u), ("V", &s.v)] {
            let gram = matmul(&f.transpose(), f);
            for i in 0..6 {
                for j in 0..6 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (gram[(i, j)] - want).abs() < 1e-10,
                        "{name}^T {name} off at ({i},{j}): {}",
                        gram[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn noiseless_matrix_equals_its_factors() {
        let s = low_rank(20, 14, 4, 0.0, 3);
        let rebuilt = s.clean();
        for (x, y) in s.a.as_slice().iter().zip(rebuilt.as_slice()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn shapes_are_as_documented() {
        let s = low_rank(30, 17, 5, 0.0, 1);
        assert_eq!((s.a.rows(), s.a.cols()), (30, 17));
        assert_eq!((s.u.rows(), s.u.cols()), (30, 5));
        assert_eq!((s.v.rows(), s.v.cols()), (17, 5));
        assert_eq!(s.singular_values.len(), 5);
        assert_eq!(s.rank(), 5);
    }

    #[test]
    fn spectrum_is_stored_descending_even_if_supplied_out_of_order() {
        let s = with_spectrum(12, 10, &[1.0, 9.0, 4.0], 0.0, 5);
        assert_eq!(s.singular_values, vec![9.0, 4.0, 1.0]);
    }

    #[test]
    fn geometric_decay_has_the_requested_ratios() {
        let s = geometric_decay(20, 20, 5, 10.0, 0.5, 0.0, 8);
        assert_eq!(s.singular_values, vec![10.0, 5.0, 2.5, 1.25, 0.625]);
    }

    #[test]
    fn noise_is_added_only_when_requested() {
        let clean = low_rank(15, 15, 3, 0.0, 2);
        let noisy = low_rank(15, 15, 3, 0.1, 2);
        assert_ne!(clean.a.as_slice(), noisy.a.as_slice());
        // Same seed => same underlying factors, so only the noise differs.
        assert_eq!(clean.u.as_slice(), noisy.u.as_slice());
    }

    #[test]
    fn generation_is_reproducible() {
        let a = low_rank(18, 12, 4, 0.05, 77);
        let b = low_rank(18, 12, 4, 0.05, 77);
        assert_eq!(a.a.as_slice(), b.a.as_slice());
    }

    #[test]
    fn pure_noise_matrix_has_the_right_shape_and_scale() {
        let e = noise(50, 40, 2.0, 9);
        assert_eq!((e.rows(), e.cols()), (50, 40));
        let var = e.as_slice().iter().map(|x| x * x).sum::<f64>() / e.len() as f64;
        assert!((var - 4.0).abs() < 0.5, "sample variance was {var}");
    }
}
