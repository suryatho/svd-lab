//! Dense, row-major matrix backed by a flat `Vec<f64>`, plus the handful of
//! vector primitives the rest of the crate needs.
//!
//! Layout: element `(i, j)` lives at `data[i * cols + j]`. Row-major is chosen
//! deliberately — the two hot kernels here (`matvec` and `matvec_t`) both walk
//! memory in contiguous row order under this layout, which is what makes them
//! straightforward to vectorize later.
//!
//! The kernels are free functions rather than inherent methods so that a
//! `std::simd` implementation can be dropped in behind the same signature
//! without touching any call site:
//!
//! ```text
//! matvec(&a, &v, &mut out)        // scalar, this file
//! matvec_simd(&a, &v, &mut out)   // portable_simd, added in a later milestone
//! ```

use std::fmt;
use std::ops::{Index, IndexMut};

/// A dense `rows x cols` matrix of `f64`, stored row-major in one allocation.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// All-zeros `rows x cols` matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Wrap an existing row-major buffer.
    ///
    /// # Panics
    /// If `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "matrix buffer is {} elements, expected {rows}x{cols} = {}",
            data.len(),
            rows * cols
        );
        Matrix { rows, cols, data }
    }

    /// Build from a slice of equal-length rows.
    ///
    /// # Panics
    /// If `rows` is empty or the rows have differing lengths.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        assert!(!rows.is_empty(), "cannot build a matrix from zero rows");
        let cols = rows[0].len();
        assert!(cols > 0, "cannot build a matrix with zero columns");
        assert!(
            rows.iter().all(|r| r.len() == cols),
            "all rows must have the same length"
        );
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            data.extend_from_slice(r);
        }
        Matrix {
            rows: rows.len(),
            cols,
            data,
        }
    }

    /// Build from a closure over `(row, col)` indices.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Matrix { rows, cols, data }
    }

    /// Build an `n x n` matrix with `values` on the diagonal and zeros elsewhere.
    pub fn from_diagonal(values: &[f64]) -> Self {
        let n = values.len();
        let mut m = Matrix::zeros(n, n);
        for (i, &v) in values.iter().enumerate() {
            m[(i, i)] = v;
        }
        m
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Total element count (`rows * cols`).
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The flat row-major backing buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Row `i` as a contiguous slice of length `cols`.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Iterator over the rows, each a contiguous slice.
    pub fn row_iter(&self) -> impl Iterator<Item = &[f64]> {
        self.data.chunks_exact(self.cols)
    }

    /// Copy column `j` out into a freshly allocated vector.
    ///
    /// This is a strided gather — avoid it in hot loops.
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows)
            .map(|i| self.data[i * self.cols + j])
            .collect()
    }

    /// Overwrite column `j`.
    ///
    /// # Panics
    /// If `values.len() != self.rows()`.
    pub fn set_column(&mut self, j: usize, values: &[f64]) {
        assert_eq!(values.len(), self.rows, "column length mismatch");
        for (i, &v) in values.iter().enumerate() {
            self.data[i * self.cols + j] = v;
        }
    }

    /// Materialize `A^T` as a new matrix.
    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        t
    }

    /// Rank-one downdate, in place: `A <- A - scale * u v^T`.
    ///
    /// This is the deflation step used by the truncated SVD (see
    /// `crate::svd`): after extracting a singular triplet `(sigma, u, v)`,
    /// subtracting `sigma * u v^T` removes that component from the matrix so
    /// the next power iteration converges to the *next* singular triplet.
    ///
    /// # Panics
    /// If `u.len() != rows` or `v.len() != cols`.
    pub fn sub_rank_one(&mut self, scale: f64, u: &[f64], v: &[f64]) {
        assert_eq!(u.len(), self.rows, "u must have length rows");
        assert_eq!(v.len(), self.cols, "v must have length cols");
        for (i, &ui) in u.iter().enumerate() {
            let coeff = scale * ui;
            if coeff == 0.0 {
                continue;
            }
            let row = &mut self.data[i * self.cols..(i + 1) * self.cols];
            for (dst, &vj) in row.iter_mut().zip(v.iter()) {
                *dst -= coeff * vj;
            }
        }
    }

    /// Rank-one update, in place: `A <- A + scale * u v^T`.
    pub fn add_rank_one(&mut self, scale: f64, u: &[f64], v: &[f64]) {
        self.sub_rank_one(-scale, u, v);
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        debug_assert!(
            i < self.rows && j < self.cols,
            "index ({i}, {j}) out of bounds"
        );
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        debug_assert!(
            i < self.rows && j < self.cols,
            "index ({i}, {j}) out of bounds"
        );
        &mut self.data[i * self.cols + j]
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}x{} matrix", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for (j, x) in self.row(i).iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{x:>10.4}")?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Kernels
//
// These are the only two operations power iteration needs, and they are the
// only two worth vectorizing. Both are written against the row-major layout so
// the inner loop is a contiguous walk.
// ---------------------------------------------------------------------------

/// `out <- A * v`. `A` is `m x n`, `v` has length `n`, `out` has length `m`.
///
/// Row-major dot-product form: each output element is one contiguous dot
/// product over a row.
///
/// # Panics
/// If the dimensions do not line up.
pub fn matvec(a: &Matrix, v: &[f64], out: &mut [f64]) {
    assert_eq!(v.len(), a.cols(), "matvec: v must have length A.cols()");
    assert_eq!(out.len(), a.rows(), "matvec: out must have length A.rows()");
    for (dst, row) in out.iter_mut().zip(a.row_iter()) {
        let mut acc = 0.0;
        for (&aij, &vj) in row.iter().zip(v.iter()) {
            acc += aij * vj;
        }
        *dst = acc;
    }
}

/// `out <- A^T * u`. `A` is `m x n`, `u` has length `m`, `out` has length `n`.
///
/// Under row-major storage the transpose product is *not* a dot product over
/// contiguous memory — it is an accumulation of scaled rows (a sequence of
/// axpy operations into `out`). Written this way it still reads `A`
/// contiguously, which matters more than the extra write traffic.
///
/// # Panics
/// If the dimensions do not line up.
pub fn matvec_t(a: &Matrix, u: &[f64], out: &mut [f64]) {
    assert_eq!(u.len(), a.rows(), "matvec_t: u must have length A.rows()");
    assert_eq!(
        out.len(),
        a.cols(),
        "matvec_t: out must have length A.cols()"
    );
    out.fill(0.0);
    for (&ui, row) in u.iter().zip(a.row_iter()) {
        if ui == 0.0 {
            continue;
        }
        for (dst, &aij) in out.iter_mut().zip(row.iter()) {
            *dst += ui * aij;
        }
    }
}

/// Allocating convenience wrapper around [`matvec`].
pub fn matvec_alloc(a: &Matrix, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.rows()];
    matvec(a, v, &mut out);
    out
}

/// Allocating convenience wrapper around [`matvec_t`].
pub fn matvec_t_alloc(a: &Matrix, u: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.cols()];
    matvec_t(a, u, &mut out);
    out
}

/// Dense matrix product `A * B`.
///
/// Naive triple loop, ordered `i-k-j` so the inner loop streams both `B`'s row
/// and the output row contiguously. Only used for test assertions and for
/// reconstructing from factors, never inside power iteration.
///
/// # Panics
/// If `a.cols() != b.rows()`.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols(), b.rows(), "matmul: inner dimensions must agree");
    let mut c = Matrix::zeros(a.rows(), b.cols());
    for i in 0..a.rows() {
        for k in 0..a.cols() {
            let aik = a[(i, k)];
            if aik == 0.0 {
                continue;
            }
            let brow = b.row(k);
            let crow = c.row_mut(i);
            for (dst, &bkj) in crow.iter_mut().zip(brow.iter()) {
                *dst += aik * bkj;
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

/// Euclidean inner product.
///
/// # Panics
/// If the slices have different lengths.
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "dot: length mismatch");
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// Euclidean (2-)norm.
#[inline]
pub fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// `x <- alpha * x`.
#[inline]
pub fn scale(x: &mut [f64], alpha: f64) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

/// Scale `x` to unit length in place and return its original norm.
///
/// Returns `0.0` and leaves `x` untouched if it is the zero vector — callers
/// must treat a zero return as "this direction is degenerate".
#[inline]
pub fn normalize(x: &mut [f64]) -> f64 {
    let n = norm(x);
    if n > 0.0 {
        scale(x, 1.0 / n);
    }
    n
}

/// Euclidean distance `||x - y||`.
///
/// # Panics
/// If the slices have different lengths.
#[inline]
pub fn distance(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "distance: length mismatch");
    x.iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Matrix {
        // 2x3
        Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
    }

    #[test]
    fn indexing_is_row_major() {
        let a = sample();
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a[(1, 2)], 6.0);
        assert_eq!(a.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(a.column(1), vec![2.0, 5.0]);
    }

    #[test]
    fn matvec_matches_hand_computation() {
        let a = sample();
        let v = [1.0, 0.0, -1.0];
        // [1*1 + 2*0 + 3*(-1), 4*1 + 5*0 + 6*(-1)] = [-2, -2]
        assert_eq!(matvec_alloc(&a, &v), vec![-2.0, -2.0]);
    }

    #[test]
    fn matvec_t_matches_transpose_then_matvec() {
        let a = sample();
        let u = [2.0, -1.0];
        let direct = matvec_t_alloc(&a, &u);
        let via_transpose = matvec_alloc(&a.transpose(), &u);
        assert_eq!(direct, via_transpose);
        // A^T u = [2*1 - 1*4, 2*2 - 1*5, 2*3 - 1*6] = [-2, -1, 0]
        assert_eq!(direct, vec![-2.0, -1.0, 0.0]);
    }

    #[test]
    fn transpose_is_an_involution() {
        let a = sample();
        assert_eq!(a.transpose().transpose(), a);
    }

    #[test]
    fn sub_rank_one_zeroes_a_rank_one_matrix() {
        let u = [1.0, 2.0];
        let v = [3.0, 4.0, 5.0];
        // A = 2 * u v^T
        let mut a = Matrix::from_fn(2, 3, |i, j| 2.0 * u[i] * v[j]);
        a.sub_rank_one(2.0, &u, &v);
        assert!(a.as_slice().iter().all(|&x| x.abs() < 1e-15));
    }

    #[test]
    fn matmul_matches_repeated_matvec() {
        let a = sample();
        let b = a.transpose(); // 3x2
        let c = matmul(&a, &b); // 2x2
        for j in 0..b.cols() {
            let col = matvec_alloc(&a, &b.column(j));
            for i in 0..a.rows() {
                assert!((c[(i, j)] - col[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn normalize_reports_original_norm_and_leaves_zero_alone() {
        let mut x = vec![3.0, 4.0];
        assert!((normalize(&mut x) - 5.0).abs() < 1e-15);
        assert!((norm(&x) - 1.0).abs() < 1e-15);

        let mut z = vec![0.0, 0.0];
        assert_eq!(normalize(&mut z), 0.0);
        assert_eq!(z, vec![0.0, 0.0]);
    }
}
