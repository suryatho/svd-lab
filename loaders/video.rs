//! Video loader: a stack of near-static frames -> matrix.
//!
//! Milestone 8, stretch goal — not implemented, stub only.
//!
//! Planned shape: read a directory of BMP frames via [`crate::image`] and
//! flatten each frame into one row, giving a `frames x pixels` matrix. Static
//! background plus small motion is close to a textbook low-rank-plus-sparse
//! decomposition, so the spectrum should decay hard.
