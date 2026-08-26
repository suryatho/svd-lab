//! Truncated SVD by power iteration and deflation.
//!
//! # What we are computing, and why it is the right thing
//!
//! Every real `m x n` matrix `A` factors as `A = U Σ V^T`, where `U` is
//! `m x m` orthogonal, `V` is `n x n` orthogonal, and `Σ` is `m x n` diagonal
//! with non-negative entries `σ_1 ≥ σ_2 ≥ ... ≥ σ_p ≥ 0`, `p = min(m, n)`.
//! Equivalently, `A` is a sum of `p` rank-one pieces:
//!
//! ```text
//!     A = Σ_{i=1..p} σ_i u_i v_i^T
//! ```
//!
//! **Eckart-Young-Mirsky.** Truncating that sum after `k` terms,
//!
//! ```text
//!     A_k = Σ_{i=1..k} σ_i u_i v_i^T,
//! ```
//!
//! gives the *best possible* rank-`k` approximation to `A` in the Frobenius
//! norm (and in the spectral norm): for every matrix `B` with `rank(B) ≤ k`,
//!
//! ```text
//!     ||A - B||_F  ≥  ||A - A_k||_F  =  sqrt( Σ_{i>k} σ_i^2 ).
//! ```
//!
//! That identity is the whole justification for this project's compression
//! scheme, and it is also a free correctness check: after computing `k`
//! triplets we can compare the measured residual against the tail sum of the
//! singular values we *didn't* keep. `tests/svd_correctness.rs` does exactly
//! that.
//!
//! The practical consequence is that SVD compresses well exactly when the
//! singular value spectrum decays fast — the tail `Σ_{i>k} σ_i^2` is what you
//! throw away. A matrix with a flat spectrum (say, i.i.d. noise) has no small
//! tail at any `k`, so no rank-`k` approximation is good. That is the
//! "low-rank structure" requirement, stated quantitatively.
//!
//! # Getting the leading triplet: the Rayleigh quotient
//!
//! The first singular value has a variational definition:
//!
//! ```text
//!     σ_1 = max_{||v|| = 1} ||A v||.
//! ```
//!
//! Squaring, and writing `B = A^T A` (symmetric positive semi-definite), the
//! quantity being maximized is the Rayleigh quotient of `B`:
//!
//! ```text
//!     ||A v||^2 = v^T A^T A v = v^T B v = R_B(v)      for ||v|| = 1.
//! ```
//!
//! Stationary points of `R_B` on the unit sphere satisfy `B v = λ v`: taking
//! the gradient of `v^T B v - λ(v^T v - 1)` and setting it to zero gives
//! `2 B v - 2 λ v = 0`. So the maximizer of `R_B` is the top eigenvector of
//! `B`, with `λ_max = σ_1^2`. Right singular vectors of `A` *are* the
//! eigenvectors of `A^T A`, and the eigenvalues are the squared singular
//! values. Maximizing the Rayleigh quotient and finding the leading singular
//! triplet are the same problem.
//!
//! # Power iteration
//!
//! To find the top eigenvector of `B` we use power iteration:
//!
//! ```text
//!     v <- B v / ||B v||       repeatedly.
//! ```
//!
//! Why it converges: expand the (random) start vector in `B`'s orthonormal
//! eigenbasis, `v^{(0)} = Σ_i c_i w_i` with `B w_i = λ_i w_i`. Then
//!
//! ```text
//!     B^t v^{(0)} = Σ_i c_i λ_i^t w_i
//!                 = λ_1^t ( c_1 w_1 + Σ_{i>1} c_i (λ_i/λ_1)^t w_i ).
//! ```
//!
//! Every ratio `λ_i / λ_1 < 1` (assuming `λ_1` is simple and `c_1 ≠ 0`), so
//! after normalizing, the non-leading terms decay geometrically and
//! `v^{(t)} -> ±w_1`. The convergence rate is governed by the *gap*:
//! error shrinks like `(λ_2/λ_1)^t = (σ_2/σ_1)^{2t}`. A matrix with two nearly
//! equal leading singular values will converge slowly — worth remembering when
//! reading timing numbers.
//!
//! Note we never form `B = A^T A` explicitly (that would cost `O(mn^2)` and
//! square the condition number). We apply it as two matrix-vector products:
//! `B v = A^T (A v)`. That is why `matrix.rs` provides exactly `matvec` and
//! `matvec_t` and nothing else.
//!
//! Once `v` has converged, the rest of the triplet falls out of the definition
//! `A v = σ u` with `||u|| = 1`:
//!
//! ```text
//!     σ = ||A v||,        u = A v / σ.
//! ```
//!
//! # Deflation: triplets 2 through k
//!
//! Having `(σ_1, u_1, v_1)`, subtract that rank-one piece off:
//!
//! ```text
//!     A^{(2)} = A - σ_1 u_1 v_1^T.
//! ```
//!
//! Because `{u_i}` and `{v_i}` are each orthonormal sets, this subtraction
//! annihilates the leading term of the SVD sum and leaves the others exactly
//! intact: `A^{(2)} = Σ_{i≥2} σ_i u_i v_i^T`. So `A^{(2)}`'s singular values
//! are `σ_2 ≥ σ_3 ≥ ...` with the same singular vectors, and running power
//! iteration again on `A^{(2)}` yields the *second* triplet of the original
//! `A`. Repeat `k` times.
//!
//! Caveat worth knowing (and worth a sentence in the writeup): deflation
//! accumulates round-off. Each subtraction leaves an `O(ε ||A||)` residue in
//! the removed direction, and later triplets inherit the error from all
//! earlier ones, so orthogonality of the computed `u_i` degrades as `k` grows.
//! For the moderate `k` this project uses that is fine; production codes use
//! block methods (randomized range finders, Lanczos) partly to avoid it.
//!
//! # Sign ambiguity
//!
//! `(σ, u, v)` and `(σ, -u, -v)` describe the same rank-one term, so the
//! singular vectors are only determined up to a simultaneous sign flip (and,
//! for repeated singular values, up to a rotation within the eigenspace). We
//! pin the sign to a fixed convention below so that runs are reproducible, but
//! *any comparison against a reference must be sign-agnostic* — the tests
//! compare `|<u_hat, u_true>|` against 1, never `u_hat` against `u_true`.

use crate::matrix::{self, Matrix};
use crate::rng::Rng;

/// Tuning knobs for the power iteration.
#[derive(Clone, Debug)]
pub struct SvdConfig {
    /// Hard cap on power iterations per singular triplet.
    pub max_iters: usize,
    /// Convergence threshold on the change in `v` between iterations
    /// (measured sign-agnostically).
    pub tol: f64,
    /// Seed for the random starting vectors. Fixed by default so results are
    /// reproducible across runs.
    pub seed: u64,
    /// Relative floor for accepting a singular value. A triplet whose `σ` is
    /// below `sigma_floor * σ_1` is treated as numerical noise and extraction
    /// stops — this is what makes `k > rank(A)` return fewer than `k`
    /// triplets rather than garbage directions.
    pub sigma_floor: f64,
    /// How many times to retry with a fresh random start if a start vector
    /// lands in the null space of `A^T A`.
    pub restarts: usize,
}

impl Default for SvdConfig {
    fn default() -> Self {
        SvdConfig {
            max_iters: 2_000,
            tol: 1e-12,
            seed: 0x5EED_5EED,
            sigma_floor: 1e-12,
            restarts: 3,
        }
    }
}

impl SvdConfig {
    /// Builder-style seed override.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder-style tolerance override.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Builder-style iteration-cap override.
    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }
}

/// The rank-`k` factorization `A ≈ U diag(σ) V^T`.
///
/// `u` is `m x k_eff`, `sigma` has length `k_eff`, `v_t` is `k_eff x n`, where
/// `k_eff ≤ k` — see [`truncated_svd`] for when it is strictly less.
#[derive(Clone, Debug)]
pub struct TruncatedSvd {
    /// Left singular vectors as columns, `m x k_eff`.
    pub u: Matrix,
    /// Singular values in descending order, length `k_eff`.
    pub sigma: Vec<f64>,
    /// Right singular vectors as *rows*, `k_eff x n`.
    pub v_t: Matrix,
    /// Power iterations actually spent on each triplet — useful for spotting
    /// the slow convergence that a small `σ_i / σ_{i-1}` gap causes.
    pub iterations: Vec<usize>,
}

impl TruncatedSvd {
    /// The rank actually recovered, `k_eff`.
    pub fn rank(&self) -> usize {
        self.sigma.len()
    }

    /// Number of rows of the original matrix.
    pub fn rows(&self) -> usize {
        self.u.rows()
    }

    /// Number of columns of the original matrix.
    pub fn cols(&self) -> usize {
        self.v_t.cols()
    }

    /// Rebuild the dense rank-`k` approximation `A_k = U diag(σ) V^T`.
    pub fn reconstruct(&self) -> Matrix {
        reconstruct(&self.u, &self.sigma, &self.v_t)
    }

    /// Left singular vector `i` as an owned vector.
    pub fn left_vector(&self, i: usize) -> Vec<f64> {
        self.u.column(i)
    }

    /// Right singular vector `i` as an owned vector.
    pub fn right_vector(&self, i: usize) -> Vec<f64> {
        self.v_t.row(i).to_vec()
    }

    /// Eckart-Young's predicted residual from the *discarded* tail, given the
    /// full spectrum. Only meaningful if you have the singular values beyond
    /// `k`; provided for the correctness tests.
    pub fn predicted_error(full_spectrum: &[f64], k: usize) -> f64 {
        full_spectrum
            .iter()
            .skip(k)
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt()
    }
}

/// Rebuild `U diag(σ) V^T` as a dense matrix.
///
/// Accumulates the rank-one terms directly rather than forming `diag(σ)`,
/// which keeps it `O(k m n)` with no temporaries.
///
/// # Panics
/// If `u.cols()`, `sigma.len()`, and `v_t.rows()` disagree.
pub fn reconstruct(u: &Matrix, sigma: &[f64], v_t: &Matrix) -> Matrix {
    assert_eq!(
        u.cols(),
        sigma.len(),
        "reconstruct: U.cols() must equal sigma.len()"
    );
    assert_eq!(
        v_t.rows(),
        sigma.len(),
        "reconstruct: V^T.rows() must equal sigma.len()"
    );

    let mut a = Matrix::zeros(u.rows(), v_t.cols());
    for (i, &s) in sigma.iter().enumerate() {
        // A += σ_i * u_i * v_i^T
        let ui = u.column(i);
        a.add_rank_one(s, &ui, v_t.row(i));
    }
    a
}

/// Truncated SVD of `a`, keeping at most the `k` leading singular triplets.
///
/// Uses [`SvdConfig::default`]; see [`truncated_svd_with`] for control over
/// tolerance, iteration cap, and seed.
pub fn truncated_svd(a: &Matrix, k: usize) -> TruncatedSvd {
    truncated_svd_with(a, k, &SvdConfig::default())
}

/// Truncated SVD of `a` with explicit configuration.
///
/// Returns *at most* `k` triplets. Fewer are returned when `A` runs out of
/// numerical rank: either `k` exceeds `min(m, n)`, or the residual matrix
/// after deflation has collapsed to (near) zero, which is exactly what happens
/// when `k > rank(A)`. Callers should read `result.rank()` rather than
/// assuming `k`.
///
/// Cost is `O(k · iters · m · n)` — each power iteration is one `matvec` plus
/// one `matvec_t`, and each deflation is one `O(mn)` pass.
pub fn truncated_svd_with(a: &Matrix, k: usize, cfg: &SvdConfig) -> TruncatedSvd {
    let (m, n) = (a.rows(), a.cols());
    let k_max = k.min(m).min(n);

    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(k_max);
    let mut sigmas: Vec<f64> = Vec::with_capacity(k_max);
    let mut v_rows: Vec<Vec<f64>> = Vec::with_capacity(k_max);
    let mut iterations: Vec<usize> = Vec::with_capacity(k_max);

    // The residual. Starts as A and loses one rank-one piece per triplet:
    //     A^{(1)} = A,   A^{(i+1)} = A^{(i)} - σ_i u_i v_i^T.
    let mut residual = a.clone();
    let mut rng = Rng::new(cfg.seed);

    // Absolute cutoff, fixed from σ_1 once we have it. Using a *relative*
    // floor rather than an absolute one keeps the behaviour scale-invariant:
    // multiplying A by 1000 should not change which triplets we accept.
    let mut cutoff = 0.0_f64;

    for _ in 0..k_max {
        let Some(triplet) = dominant_triplet(&residual, cfg, &mut rng) else {
            break; // residual is numerically zero; nothing left to extract
        };

        if sigmas.is_empty() {
            cutoff = triplet.sigma * cfg.sigma_floor;
        } else if triplet.sigma <= cutoff {
            break; // below the noise floor set by σ_1 — stop rather than fit noise
        }

        // Deflate before moving on, so the next iteration sees Σ_{j>i} σ_j u_j v_j^T.
        residual.sub_rank_one(triplet.sigma, &triplet.u, &triplet.v);

        u_cols.push(triplet.u);
        sigmas.push(triplet.sigma);
        v_rows.push(triplet.v);
        iterations.push(triplet.iters);
    }

    // Pack the collected vectors into U (columns) and V^T (rows).
    let k_eff = sigmas.len();
    let mut u = Matrix::zeros(m, k_eff);
    for (i, col) in u_cols.iter().enumerate() {
        u.set_column(i, col);
    }
    let mut v_t = Matrix::zeros(k_eff, n);
    for (i, row) in v_rows.iter().enumerate() {
        v_t.row_mut(i).copy_from_slice(row);
    }

    TruncatedSvd {
        u,
        sigma: sigmas,
        v_t,
        iterations,
    }
}

/// Just the singular values (no vectors), for spectrum-decay plots.
///
/// Same cost as the full [`truncated_svd`] — the vectors have to be computed
/// to deflate — this only saves the caller from carrying them around.
pub fn singular_values(a: &Matrix, k: usize) -> Vec<f64> {
    truncated_svd(a, k).sigma
}

/// One converged singular triplet of the matrix it was extracted from.
struct Triplet {
    sigma: f64,
    /// Unit-norm left singular vector, length `m`.
    u: Vec<f64>,
    /// Unit-norm right singular vector, length `n`.
    v: Vec<f64>,
    iters: usize,
}

/// Power-iterate to the dominant singular triplet of `a`.
///
/// Returns `None` if `a` is numerically the zero matrix (every start vector
/// maps to zero), which is the signal for the caller to stop deflating.
fn dominant_triplet(a: &Matrix, cfg: &SvdConfig, rng: &mut Rng) -> Option<Triplet> {
    let (m, n) = (a.rows(), a.cols());
    if m == 0 || n == 0 {
        return None;
    }

    // Scratch buffers, reused across all iterations so the inner loop does no
    // allocation.
    let mut av = vec![0.0; m]; // holds A v
    let mut bv = vec![0.0; n]; // holds A^T A v
    let mut v = vec![0.0; n];

    // `restarts + 1` attempts: a random start can (with probability zero in
    // exact arithmetic, but not never in practice for structured matrices)
    // land in the null space of A^T A. Redrawing is cheaper than being clever.
    for _ in 0..=cfg.restarts {
        // Gaussian start vector. An isotropic distribution guarantees the
        // component along the leading eigenvector, `c_1` in the convergence
        // argument above, is non-zero with probability 1.
        rng.fill_normal(&mut v);
        if matrix::normalize(&mut v) == 0.0 {
            continue;
        }

        let mut iters = 0;
        let mut converged = false;

        for _ in 0..cfg.max_iters {
            // One application of B = A^T A, as two matvecs. Never form B.
            matrix::matvec(a, &v, &mut av); //  av <- A v
            matrix::matvec_t(a, &av, &mut bv); //  bv <- A^T (A v) = B v

            let bv_norm = matrix::normalize(&mut bv);
            if bv_norm == 0.0 {
                // v lies in ker(A^T A) = ker(A). Either A is the zero matrix
                // or we got unlucky; the restart loop sorts out which.
                break;
            }
            iters += 1;

            // Sign-agnostic convergence test: v^{(t)} -> ±w_1, and which sign
            // it settles on depends on the start vector, so a plain
            // ||v_new - v_old|| can sit at ~2 forever on a flipping sequence.
            // (With B PSD the iterate does not actually alternate, but the
            // min() costs nothing and makes the test robust.)
            let delta = matrix::distance(&bv, &v).min(
                bv.iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| (a + b) * (a + b))
                    .sum::<f64>()
                    .sqrt(),
            );

            v.copy_from_slice(&bv);

            if delta < cfg.tol {
                converged = true;
                break;
            }
        }

        if iters == 0 {
            continue; // degenerate start; try another
        }
        let _ = converged; // hitting max_iters is not an error, just slower convergence

        // Recover σ and u from the converged v:  A v = σ u,  ||u|| = 1.
        // σ = ||A v|| is precisely the Rayleigh-quotient value sqrt(v^T B v)
        // at the maximizer, i.e. σ_1 of this (possibly deflated) matrix.
        matrix::matvec(a, &v, &mut av);
        let mut u = av.clone();
        let sigma = matrix::normalize(&mut u);
        if sigma == 0.0 {
            continue;
        }

        // Pin the sign so repeated runs agree: make the largest-magnitude
        // entry of u positive, flipping v to match. (σ u v^T is invariant
        // under flipping both, so this changes nothing mathematically.)
        let pivot = u
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| {
                x.abs()
                    .partial_cmp(&y.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        if u[pivot] < 0.0 {
            matrix::scale(&mut u, -1.0);
            matrix::scale(&mut v, -1.0);
        }

        return Some(Triplet { sigma, u, v, iters });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::matmul;

    /// `diag(3, 2, 1)` has an SVD we can write down by hand.
    #[test]
    fn diagonal_matrix_has_its_diagonal_as_singular_values() {
        let a = Matrix::from_diagonal(&[3.0, 2.0, 1.0]);
        let svd = truncated_svd(&a, 3);
        assert_eq!(svd.rank(), 3);
        for (got, want) in svd.sigma.iter().zip([3.0, 2.0, 1.0]) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    /// A rank-one matrix: only one triplet exists, and it is exact.
    #[test]
    fn rank_one_matrix_is_recovered_exactly() {
        let u = [1.0, 2.0, 2.0]; // norm 3
        let v = [0.0, 3.0, 4.0]; // norm 5
        let a = Matrix::from_fn(3, 3, |i, j| u[i] * v[j]);

        let svd = truncated_svd(&a, 1);
        assert_eq!(svd.rank(), 1);
        // σ_1 = ||u|| * ||v|| = 15
        assert!(
            (svd.sigma[0] - 15.0).abs() < 1e-9,
            "sigma was {}",
            svd.sigma[0]
        );

        let approx = svd.reconstruct();
        for (x, y) in approx.as_slice().iter().zip(a.as_slice()) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    /// Asking for more triplets than the matrix has rank must not invent them.
    #[test]
    fn extraction_stops_at_numerical_rank() {
        let u = [1.0, 0.5, -2.0, 0.25];
        let v = [3.0, 1.0, -1.0];
        let a = Matrix::from_fn(4, 3, |i, j| u[i] * v[j]); // rank 1

        let svd = truncated_svd(&a, 3);
        assert_eq!(svd.rank(), 1, "recovered sigmas: {:?}", svd.sigma);
    }

    /// Requesting k larger than min(m, n) is clamped, not an error.
    #[test]
    fn k_is_clamped_to_min_dimension() {
        let a = Matrix::from_fn(3, 5, |i, j| ((i * 5 + j) as f64).sin());
        let svd = truncated_svd(&a, 99);
        assert!(svd.rank() <= 3, "rank was {}", svd.rank());
        assert_eq!(svd.u.rows(), 3);
        assert_eq!(svd.v_t.cols(), 5);
    }

    /// Singular values must come out in descending order — deflation depends
    /// on it, and so does the Eckart-Young tail-sum identity.
    #[test]
    fn singular_values_are_descending() {
        let a = Matrix::from_fn(20, 12, |i, j| ((i * 31 + j * 17) as f64).sin());
        let svd = truncated_svd(&a, 8);
        for w in svd.sigma.windows(2) {
            assert!(w[0] >= w[1] - 1e-12, "not descending: {:?}", svd.sigma);
        }
    }

    /// U's columns and V^T's rows should each be orthonormal sets.
    #[test]
    fn factors_are_orthonormal() {
        let a = Matrix::from_fn(30, 18, |i, j| {
            ((i as f64) * 0.3).cos() * ((j as f64) * 0.7).sin()
        });
        let svd = truncated_svd(&a, 6);

        let utu = matmul(&svd.u.transpose(), &svd.u);
        let vtv = matmul(&svd.v_t, &svd.v_t.transpose());
        for i in 0..svd.rank() {
            for j in 0..svd.rank() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((utu[(i, j)] - want).abs() < 1e-8, "U^T U off at ({i},{j})");
                assert!((vtv[(i, j)] - want).abs() < 1e-8, "V^T V off at ({i},{j})");
            }
        }
    }

    /// `A v = σ u` is the defining relation; check it directly.
    #[test]
    fn triplets_satisfy_the_defining_relation() {
        let a = Matrix::from_fn(25, 15, |i, j| {
            ((i + 1) as f64).sqrt() * ((j + 2) as f64).ln()
        });
        let svd = truncated_svd(&a, 4);
        for i in 0..svd.rank() {
            let av = matrix::matvec_alloc(&a, &svd.right_vector(i));
            let su: Vec<f64> = svd
                .left_vector(i)
                .iter()
                .map(|x| svd.sigma[i] * x)
                .collect();
            let err = matrix::distance(&av, &su) / svd.sigma[0];
            assert!(
                err < 1e-8,
                "triplet {i}: ||A v - sigma u|| / sigma_1 = {err}"
            );
        }
    }

    #[test]
    fn same_seed_gives_identical_results() {
        let a = Matrix::from_fn(16, 9, |i, j| ((i * 7 + j * 3) as f64).cos());
        let cfg = SvdConfig::default().with_seed(1234);
        let first = truncated_svd_with(&a, 5, &cfg);
        let second = truncated_svd_with(&a, 5, &cfg);
        assert_eq!(first.sigma, second.sigma);
        assert_eq!(first.u, second.u);
        assert_eq!(first.v_t, second.v_t);
    }

    #[test]
    fn zero_matrix_yields_no_triplets() {
        let a = Matrix::zeros(6, 4);
        let svd = truncated_svd(&a, 3);
        assert_eq!(svd.rank(), 0);
        assert!(svd.reconstruct().as_slice().iter().all(|&x| x == 0.0));
    }
}
