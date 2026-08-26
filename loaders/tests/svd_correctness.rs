//! Correctness of the truncated SVD against synthetic ground truth.
//!
//! These tests live in the `loaders` crate because they need both the solver
//! (`svd_core`) and the ground-truth generator (`svd_loaders::synthetic`).
//! They are the gate for milestone 5: no real-data loader gets built until
//! every assertion here holds.
//!
//! What is checked, and why each one is a distinct claim:
//!
//! 1. **Singular values.** The recovered `σ` match the ones we constructed.
//! 2. **Singular vectors, sign-agnostically.** `(σ, u, v)` and `(σ, -u, -v)`
//!    are the same rank-one term, so we compare `|<u_hat, u_true>|` to 1 and
//!    never compare the vectors elementwise.
//! 3. **Reconstruction.** `A_k` reproduces a rank-`k` matrix to machine
//!    precision.
//! 4. **Eckart-Young.** The measured residual `||A - A_k||_F` equals the
//!    predicted tail `sqrt(Σ_{i>k} σ_i^2)`. This is the strongest check —
//!    it says we found the *optimal* rank-`k` approximation, not merely a
//!    rank-`k` one that happens to fit.
//! 5. **Optimality against competitors.** No other rank-`k` matrix beats it.
//! 6. **Degenerate and adversarial cases.** Rank overshoot, noise, and a
//!    spectrum with no decay at all.

use svd_core::matrix::{self, Matrix};
use svd_core::metrics;
use svd_core::svd::{self, SvdConfig};
use svd_loaders::synthetic;

/// Cosine of the angle between two vectors, ignoring sign.
///
/// This is the right way to compare singular vectors: the decomposition only
/// determines `u_i` up to a simultaneous sign flip with `v_i`, so a value of
/// 1 means "same direction", regardless of orientation.
fn abs_cosine(x: &[f64], y: &[f64]) -> f64 {
    let denom = matrix::norm(x) * matrix::norm(y);
    assert!(denom > 0.0, "cannot compare against a zero vector");
    (matrix::dot(x, y) / denom).abs()
}

// ---------------------------------------------------------------------------
// 1-3: exact recovery on a noiseless, exactly rank-r matrix
// ---------------------------------------------------------------------------

#[test]
fn recovers_a_known_rank_five_matrix() {
    let truth = synthetic::with_spectrum(80, 50, &[20.0, 12.0, 7.0, 3.0, 1.0], 0.0, 42);
    let got = svd::truncated_svd(&truth.a, 5);

    assert_eq!(got.rank(), 5, "expected 5 triplets, got {}", got.rank());

    // (1) singular values
    for (i, (&want, &have)) in truth
        .singular_values
        .iter()
        .zip(got.sigma.iter())
        .enumerate()
    {
        let rel = (have - want).abs() / want;
        assert!(
            rel < 1e-9,
            "sigma[{i}]: got {have}, want {want} (rel err {rel:.3e})"
        );
    }

    // (2) singular vectors, up to sign
    for i in 0..5 {
        let cu = abs_cosine(&got.left_vector(i), &truth.left_vector(i));
        let cv = abs_cosine(&got.right_vector(i), &truth.right_vector(i));
        assert!(cu > 1.0 - 1e-8, "u[{i}]: |cos| = {cu}");
        assert!(cv > 1.0 - 1e-8, "v[{i}]: |cos| = {cv}");
    }

    // (2b) the signs must be *consistent*: whatever orientation was chosen for
    // u_i, v_i must carry the matching one, or the rank-one term flips.
    for i in 0..5 {
        let su = matrix::dot(&got.left_vector(i), &truth.left_vector(i)).signum();
        let sv = matrix::dot(&got.right_vector(i), &truth.right_vector(i)).signum();
        assert_eq!(su, sv, "u[{i}] and v[{i}] disagree on sign");
    }

    // (3) full-rank reconstruction is exact
    let rel = metrics::relative_error(&truth.a, &got.reconstruct());
    assert!(rel < 1e-10, "relative reconstruction error was {rel:.3e}");
}

#[test]
fn recovers_a_wide_matrix_as_well_as_a_tall_one() {
    // Power iteration runs on A^T A regardless of orientation; make sure the
    // bookkeeping is right in both.
    for (m, n) in [(60usize, 25usize), (25, 60)] {
        let truth = synthetic::low_rank(m, n, 4, 0.0, 7);
        let got = svd::truncated_svd(&truth.a, 4);

        assert_eq!(got.rank(), 4, "{m}x{n}");
        assert_eq!(got.u.rows(), m);
        assert_eq!(got.u.cols(), 4);
        assert_eq!(got.v_t.rows(), 4);
        assert_eq!(got.v_t.cols(), n);

        for (i, (&want, &have)) in truth
            .singular_values
            .iter()
            .zip(got.sigma.iter())
            .enumerate()
        {
            assert!(
                (have - want).abs() / want < 1e-9,
                "{m}x{n} sigma[{i}]: {have} vs {want}"
            );
        }
        assert!(
            metrics::relative_error(&truth.a, &got.reconstruct()) < 1e-10,
            "{m}x{n}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4: Eckart-Young
// ---------------------------------------------------------------------------

#[test]
fn truncation_error_matches_the_eckart_young_tail() {
    // A genuinely rank-8 matrix, truncated at every k from 0 to 8. At each k
    // the residual must equal sqrt(sum of the squared discarded sigmas).
    let sigmas = [30.0, 21.0, 15.0, 11.0, 6.0, 4.0, 2.5, 1.0];
    let truth = synthetic::with_spectrum(70, 45, &sigmas, 0.0, 2024);

    for k in 0..=sigmas.len() {
        let got = svd::truncated_svd(&truth.a, k);
        let measured = metrics::frobenius_error(&truth.a, &got.reconstruct());
        let predicted = metrics::tail_energy(&sigmas, k);

        let scale = metrics::frobenius_norm(&truth.a);
        let err = (measured - predicted).abs() / scale;
        assert!(
            err < 1e-9,
            "k = {k}: measured {measured:.12}, Eckart-Young predicts {predicted:.12} \
             (normalized gap {err:.3e})"
        );
    }
}

#[test]
fn frobenius_norm_equals_the_root_sum_of_squared_singular_values() {
    // ||A||_F^2 = sum_i sigma_i^2. Equivalent to Eckart-Young at k = 0, but
    // it checks the *whole* spectrum was found rather than just the tail.
    let truth = synthetic::low_rank(50, 30, 9, 0.0, 555);
    let got = svd::truncated_svd(&truth.a, 9);
    let from_spectrum = got.sigma.iter().map(|s| s * s).sum::<f64>().sqrt();
    let direct = metrics::frobenius_norm(&truth.a);
    assert!(
        (from_spectrum - direct).abs() / direct < 1e-10,
        "spectrum gives {from_spectrum}, direct gives {direct}"
    );
}

// ---------------------------------------------------------------------------
// 5: optimality
// ---------------------------------------------------------------------------

#[test]
fn no_random_rank_k_competitor_beats_the_truncated_svd() {
    // Eckart-Young says A_k is the minimizer over all rank-<=k matrices. We
    // cannot check every competitor, but we can check that a pile of random
    // ones all lose — and that a *perturbed* version of A_k itself loses,
    // which is the local-optimality statement.
    let truth = synthetic::geometric_decay(40, 28, 12, 10.0, 0.7, 0.0, 99);
    let k = 4;
    let best = svd::truncated_svd(&truth.a, k);
    let best_err = metrics::frobenius_error(&truth.a, &best.reconstruct());

    for seed in 0..8u64 {
        let competitor = synthetic::low_rank(40, 28, k, 0.0, 1000 + seed);
        // Scale the competitor to the same Frobenius norm as A so it is not
        // losing merely on magnitude.
        let target = metrics::frobenius_norm(&truth.a);
        let have = metrics::frobenius_norm(&competitor.a);
        let scaled = Matrix::from_fn(40, 28, |i, j| competitor.a[(i, j)] * (target / have));
        let err = metrics::frobenius_error(&truth.a, &scaled);
        assert!(
            err > best_err,
            "random rank-{k} competitor (seed {seed}) beat the SVD"
        );
    }

    // Perturbing the optimal factors must make things worse.
    let mut perturbed = best.clone();
    perturbed.sigma[0] *= 1.01;
    let perturbed_err = metrics::frobenius_error(&truth.a, &perturbed.reconstruct());
    assert!(
        perturbed_err > best_err,
        "perturbing sigma_1 did not increase the error"
    );
}

// ---------------------------------------------------------------------------
// 6: degenerate and adversarial cases
// ---------------------------------------------------------------------------

#[test]
fn asking_for_more_rank_than_exists_returns_only_what_exists() {
    let truth = synthetic::low_rank(35, 22, 3, 0.0, 13);
    let got = svd::truncated_svd(&truth.a, 15);

    assert_eq!(got.rank(), 3, "invented triplets: {:?}", got.sigma);
    assert!(metrics::relative_error(&truth.a, &got.reconstruct()) < 1e-10);
}

#[test]
fn noisy_matrix_recovers_the_leading_spectrum_within_the_weyl_bound() {
    // Weyl: |sigma_i(A + E) - sigma_i(A)| <= ||E||_2, and for an m x n i.i.d.
    // N(0, s^2) matrix ||E||_2 concentrates around s * (sqrt(m) + sqrt(n)).
    let (m, n) = (100usize, 60usize);
    let stddev = 0.02;
    let sigmas = [25.0, 18.0, 11.0, 5.0];
    let truth = synthetic::with_spectrum(m, n, &sigmas, stddev, 314);

    let got = svd::truncated_svd(&truth.a, sigmas.len());
    assert_eq!(got.rank(), sigmas.len());

    let weyl = stddev * ((m as f64).sqrt() + (n as f64).sqrt());
    for (i, (&want, &have)) in sigmas.iter().zip(got.sigma.iter()).enumerate() {
        let gap = (have - want).abs();
        assert!(
            gap <= 1.5 * weyl,
            "sigma[{i}]: got {have}, want {want}, gap {gap} > Weyl {weyl}"
        );
    }

    // The signal subspace should still be recovered accurately: the leading
    // vectors are well separated relative to the noise level.
    for i in 0..sigmas.len() {
        let cu = abs_cosine(&got.left_vector(i), &truth.left_vector(i));
        assert!(cu > 0.999, "noisy u[{i}]: |cos| = {cu}");
    }

    // Against the *clean* matrix, rank-4 should be far better than the noise
    // floor of the observed one — i.e. the SVD denoises.
    let clean = truth.clean();
    let to_clean = metrics::relative_error(&clean, &got.reconstruct());
    let observed_to_clean = metrics::relative_error(&clean, &truth.a);
    assert!(
        to_clean < observed_to_clean,
        "rank-4 approx ({to_clean:.4e}) was no closer to the clean matrix \
         than the noisy observation itself ({observed_to_clean:.4e})"
    );
}

#[test]
fn pure_noise_does_not_compress() {
    // The negative control. An i.i.d. Gaussian matrix has a Marchenko-Pastur
    // spectrum: no sharp decay, so a low-rank truncation keeps only a small
    // fraction of the energy. If this test ever "passes" with a high energy
    // fraction, something is wrong with the generator, not with SVD.
    let (m, n) = (120usize, 120usize);
    let e = synthetic::noise(m, n, 1.0, 8675309);
    let k = 12; // 10% of the rank
    let got = svd::truncated_svd(&e, k);

    let kept = got.sigma.iter().map(|s| s * s).sum::<f64>();
    let total = metrics::frobenius_norm(&e).powi(2);
    let fraction = kept / total;

    assert!(
        fraction < 0.35,
        "rank-{k} captured {:.1}% of a noise matrix's energy — too much",
        100.0 * fraction
    );

    // And the spectrum should be flat-ish: sigma_1 / sigma_k stays small,
    // unlike the structured cases where it blows up.
    let ratio = got.sigma[0] / got.sigma[k - 1];
    assert!(
        ratio < 2.0,
        "noise spectrum decayed by {ratio:.2}x over {k} values — too structured"
    );
}

#[test]
fn structured_data_does_compress() {
    // The contrast case for the writeup: same size, same k, fast decay.
    let truth = synthetic::geometric_decay(120, 120, 120, 100.0, 0.75, 0.0, 4242);
    let k = 12;
    let got = svd::truncated_svd(&truth.a, k);

    let fraction = metrics::energy_fraction(&got.sigma, k)
        * (got.sigma.iter().map(|s| s * s).sum::<f64>()
            / metrics::frobenius_norm(&truth.a).powi(2));
    assert!(
        fraction > 0.99,
        "rank-{k} captured only {:.2}% of a fast-decaying matrix's energy",
        100.0 * fraction
    );
}

#[test]
fn repeated_singular_values_still_give_a_valid_decomposition() {
    // With sigma_1 == sigma_2 the individual singular *vectors* are not
    // unique — any orthonormal basis of the shared eigenspace works, and
    // power iteration's convergence rate degenerates. The singular *values*
    // and the reconstruction are still well defined, so that is all we assert.
    let truth = synthetic::with_spectrum(40, 30, &[9.0, 9.0, 9.0, 2.0], 0.0, 17);
    let cfg = SvdConfig::default().with_max_iters(20_000).with_tol(1e-13);
    let got = svd::truncated_svd_with(&truth.a, 4, &cfg);

    assert_eq!(got.rank(), 4);
    for (i, (&want, &have)) in truth
        .singular_values
        .iter()
        .zip(got.sigma.iter())
        .enumerate()
    {
        assert!(
            (have - want).abs() / want < 1e-6,
            "sigma[{i}]: got {have}, want {want}"
        );
    }
    let rel = metrics::relative_error(&truth.a, &got.reconstruct());
    assert!(rel < 1e-8, "relative reconstruction error was {rel:.3e}");
}

#[test]
fn results_do_not_depend_on_the_random_seed() {
    // Different starting vectors must converge to the same triplets (up to
    // sign, which the fixed sign convention already pins down).
    let truth = synthetic::with_spectrum(45, 33, &[14.0, 8.0, 5.0, 2.0], 0.0, 88);
    let reference = svd::truncated_svd_with(&truth.a, 4, &SvdConfig::default().with_seed(1));

    for seed in [2u64, 3, 4, 999, 123_456_789] {
        let got = svd::truncated_svd_with(&truth.a, 4, &SvdConfig::default().with_seed(seed));
        for i in 0..4 {
            let d = (got.sigma[i] - reference.sigma[i]).abs() / reference.sigma[i];
            assert!(d < 1e-9, "seed {seed}, sigma[{i}] drifted by {d:.3e}");
            let c = abs_cosine(&got.left_vector(i), &reference.left_vector(i));
            assert!(c > 1.0 - 1e-8, "seed {seed}, u[{i}]: |cos| = {c}");
        }
    }
}

#[test]
fn error_decreases_monotonically_as_k_grows() {
    // Adding a term can never make the approximation worse — the residual is
    // a tail sum of non-negative squares.
    let truth = synthetic::geometric_decay(60, 40, 20, 50.0, 0.8, 0.0, 606);
    let mut previous = f64::INFINITY;
    for k in 0..=20 {
        let got = svd::truncated_svd(&truth.a, k);
        let err = metrics::frobenius_error(&truth.a, &got.reconstruct());
        assert!(
            err <= previous + 1e-9,
            "error rose from {previous} to {err} at k = {k}"
        );
        previous = err;
    }
    assert!(
        previous < 1e-8,
        "rank-20 approximation of a rank-20 matrix left error {previous}"
    );
}

#[test]
fn compression_ratio_lines_up_with_the_factor_sizes() {
    let truth = synthetic::low_rank(200, 100, 6, 0.0, 21);
    let k = 6;
    let got = svd::truncated_svd(&truth.a, k);

    let stored = got.u.len() + got.sigma.len() + got.v_t.len();
    assert_eq!(stored, metrics::factored_storage(200, 100, k));

    let ratio = metrics::compression_ratio(200, 100, k);
    assert!((ratio - truth.a.len() as f64 / stored as f64).abs() < 1e-12);
    assert!(ratio > 10.0, "expected real compression, got {ratio:.2}x");
}
