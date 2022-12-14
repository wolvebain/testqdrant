#![allow(deprecated)]

pub mod collection;
pub mod collection_manager;
pub mod collection_state;
mod common;
pub mod config;
pub mod hash_ring;
pub mod operations;
pub mod optimizers_builder;
pub mod recommendations;
pub mod save_on_disk;
pub mod shards;
pub mod telemetry;
mod update_handler;
pub mod wal;

#[cfg(test)]
mod tests;
