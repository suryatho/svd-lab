//! Spectrogram loader: WAV -> STFT magnitude matrix.
//!
//! Milestone 8, stretch goal — not implemented, stub only.
//!
//! Planned shape: parse an uncompressed PCM WAV, window the samples, run a
//! hand-written radix-2 FFT per frame, and stack the magnitude spectra into a
//! `frames x bins` matrix.
