//! Image loader: uncompressed BMP.
//!
//! Milestone 7 — not implemented yet.
//!
//! Planned shape: `load(path) -> io::Result<Matrix>` and
//! `save(path, &Matrix) -> io::Result<()>`, producing a `height x width`
//! matrix of grayscale intensities in `[0, 255]`.
//!
//! BMP is chosen so no DEFLATE decoder is needed. Details to get right when
//! this is written:
//! - 14-byte file header + 40-byte `BITMAPINFOHEADER` = the usual 54-byte
//!   preamble, but read the pixel-data offset from the header rather than
//!   assuming 54.
//! - Rows are stored bottom-to-top for a positive `biHeight` (a negative
//!   `biHeight` means top-down).
//! - Each row is padded to a 4-byte boundary.
//! - Channel order is BGR, not RGB.
