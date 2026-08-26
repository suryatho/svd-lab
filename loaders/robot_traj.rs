//! Robot trajectory loader: header-less CSV, one row per timestep, one column
//! per state variable (qpos/qvel logged from MuJoCo).
//!
//! Milestone 6 — not implemented yet. Deliberately left until after the
//! synthetic correctness test passes.
//!
//! Planned shape: `load(path) -> io::Result<Matrix>` producing a
//! `timesteps x state_dim` matrix, parsed with `str::split(',')` and
//! `f64::from_str`, no CSV crate.
//!
//! Why this domain is interesting: a robot following a smooth policy visits a
//! low-dimensional manifold of its state space, so the trajectory matrix
//! should have a fast-decaying spectrum. Columns have wildly different units
//! and scales (radians vs rad/s), though, so whether to centre and scale the
//! columns before factoring is a real modelling choice — one that changes what
//! "best rank-k" even means.
