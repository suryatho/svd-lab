//! Data loaders: each module turns one data domain into an `svd_core::Matrix`
//! so the same truncated SVD can be pointed at all of them.
//!
//! Status by milestone:
//! - [`synthetic`] — implemented. Ground-truth matrices with a known spectrum.
//! - [`robot_traj`] — stub. CSV of MuJoCo qpos/qvel, rows = timesteps.
//! - [`image`] — stub. Uncompressed BMP.
//! - [`spectrogram`] — stub, stretch goal. WAV -> STFT.
//! - [`video`] — stub, stretch goal. Stacked frames.
//!
//! Pure `std` throughout; the parsers are written by hand on purpose.

pub mod image;
pub mod robot_traj;
pub mod spectrogram;
pub mod synthetic;
pub mod video;
