//! A tiny deterministic PRNG.
//!
//! Power iteration needs a random starting vector and the synthetic loader
//! needs Gaussian noise, but the project is deliberately dependency-free, so
//! this is a hand-rolled xorshift64* — a well-known, adequate-quality
//! generator for Monte-Carlo-ish uses like these. It is *not* suitable for
//! anything cryptographic.
//!
//! Everything is seeded explicitly so runs are reproducible: the same seed
//! gives the same starting vector, hence the same singular triplets, hence
//! the same numbers in the writeup.

/// xorshift64* generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    /// Box-Muller produces two independent normals per call; the second is
    /// cached here and handed out on the following call.
    spare_normal: Option<f64>,
}

impl Rng {
    /// Seed the generator. A zero seed is remapped, since xorshift is stuck at
    /// zero.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Rng {
            state,
            spare_normal: None,
        }
    }

    /// Next raw 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    ///
    /// Uses the top 53 bits, which is exactly the mantissa width of `f64`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Standard normal `N(0, 1)`, via the polar form of Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        // Rejection-sample a point inside the unit disc, then map it to a pair
        // of independent normals. The polar form avoids calling sin/cos.
        loop {
            let u = self.next_range(-1.0, 1.0);
            let v = self.next_range(-1.0, 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some(v * factor);
                return u * factor;
            }
        }
    }

    /// Fill `out` with standard normal samples.
    pub fn fill_normal(&mut self, out: &mut [f64]) {
        for x in out.iter_mut() {
            *x = self.next_normal();
        }
    }

    /// A fresh vector of `n` standard normal samples.
    pub fn normal_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_normal()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn zero_seed_is_not_stuck() {
        let mut r = Rng::new(0);
        let first = r.next_u64();
        assert_ne!(first, 0);
        assert_ne!(first, r.next_u64());
    }

    #[test]
    fn uniform_stays_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn normals_have_roughly_the_right_moments() {
        let mut r = Rng::new(2024);
        let n = 200_000;
        let samples: Vec<f64> = (0..n).map(|_| r.next_normal()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        // Standard error of the mean is 1/sqrt(n) ~= 0.0022 here, so these are
        // very loose bounds — this is a smoke test, not a distribution test.
        assert!(mean.abs() < 0.02, "mean was {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance was {var}");
    }
}
