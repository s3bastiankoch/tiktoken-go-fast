//! The Rust side of the cross-language benchmark: a long-lived round server
//! speaking the same line protocol as `goarm`.
//!
//!     load <encoding>   -> ok
//!     open <path>       -> ok <bytes>
//!     check             -> ok <ntokens> <fnv1a64 of the ids>
//!     run <iters>       -> ok <elapsed_ns> <sink>
//!     quit
//!
//! The vocabulary is loaded once, before any timing, and the driver alternates
//! single rounds between this process and the Go one so that machine drift
//! cancels in the paired ratio.

use std::hint::black_box;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use tiktoken_rs::CoreBPE;
use tiktoken_rs::{cl100k_base, o200k_base, p50k_base, r50k_base};

/// FNV-1a over the little-endian token ids, byte for byte what the Go arm
/// computes. The driver compares the two before it times anything.
fn digest(ids: &[u32]) -> u64 {
    const OFFSET64: u64 = 0xcbf29ce484222325;
    const PRIME64: u64 = 0x100000001b3;
    let mut h = OFFSET64;
    for id in ids {
        for byte in id.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(PRIME64);
        }
    }
    h
}

fn load(name: &str) -> Option<CoreBPE> {
    match name {
        "cl100k_base" => cl100k_base().ok(),
        "o200k_base" => o200k_base().ok(),
        "p50k_base" => p50k_base().ok(),
        "r50k_base" => r50k_base().ok(),
        _ => None,
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut bpe: Option<CoreBPE> = None;
    let mut text = String::new();

    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let mut fields = line.split_whitespace();
        let Some(cmd) = fields.next() else { continue };
        match cmd {
            "load" => {
                let name = fields.next().unwrap_or("");
                match load(name) {
                    Some(enc) => {
                        bpe = Some(enc);
                        writeln!(stdout, "ok").unwrap();
                    }
                    None => writeln!(stdout, "err unknown encoding {name}").unwrap(),
                }
            }
            "open" => {
                let path = fields.next().unwrap_or("");
                match std::fs::read_to_string(path) {
                    Ok(s) => {
                        let n = s.len();
                        text = s;
                        writeln!(stdout, "ok {n}").unwrap();
                    }
                    Err(e) => writeln!(stdout, "err {e}").unwrap(),
                }
            }
            "check" => {
                let enc = bpe.as_ref().expect("load first");
                let ids = enc.encode_ordinary(&text);
                writeln!(stdout, "ok {} {:016x}", ids.len(), digest(&ids)).unwrap();
            }
            "run" => {
                let n: usize = fields.next().unwrap_or("1").parse().unwrap_or(1);
                let enc = bpe.as_ref().expect("load first");
                let mut sink: usize = 0;
                let start = Instant::now();
                for _ in 0..n {
                    sink += black_box(enc.encode_ordinary(black_box(&text))).len();
                }
                let elapsed = start.elapsed().as_nanos();
                writeln!(stdout, "ok {elapsed} {sink}").unwrap();
            }
            "quit" => {
                stdout.flush().unwrap();
                return;
            }
            other => writeln!(stdout, "err unknown command {other}").unwrap(),
        }
        stdout.flush().unwrap();
    }
}
