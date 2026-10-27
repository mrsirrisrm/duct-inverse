//! Rebuild a connected duct network from what a Revit model actually records.
//!
//! Mechanical models are rarely fully connected: takeoffs record only their
//! branch, flex ducts are not decoded, connectors are left open. This crate
//! takes the elements rvt-rs extracts, keeps every join the file records, and
//! adds the joins the geometry implies, each tagged with how it was found.
//!
//! Pure Rust, no I/O, so it can sit next to `wall-duct-core` in WASM.
//!
//! Conventions (as `wall-duct-core`): lengths in metres, flows in m^3/s.

pub mod graph;
pub mod heal;
pub mod network;
pub mod path;
pub mod rvt_json;

pub use network::{EdgeSource, Network, Node, NodeKind};
