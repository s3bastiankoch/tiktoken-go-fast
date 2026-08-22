# tiktoken-go-fast vs openai/tiktoken's Rust core

The other comparison in this repository is Go against Go. This one is against
the implementation everything else is a port of: the Rust `CoreBPE` in
[openai/tiktoken](https://github.com/openai/tiktoken), reached through
[tiktoken-rs](https://crates.io/crates/tiktoken-rs) v0.12.0, which vendors that
core verbatim as `src/vendor_tiktoken.rs` and ships the same `.tiktoken`
vocabularies. It is the reference, not a third-party reimplementation of it.

**On English prose and source code this fork is 1.3-3.2x faster than the Rust
core. On continuous Chinese and Japanese the Rust core is up to 2.4x faster than
this fork.** Both halves of that sentence come from real corpora, and one thing
explains both: this fork replaced the pre-tokeniser and did not touch the
byte-pair merge.

```sh
cd bench && go run ./cmd/dumpcorpora -out /tmp/corpora        # same bytes the other tables use
cd vsrust/rustarm && cargo build --release                    # the Rust arm
cd .. && go build -o /tmp/goarm ./goarm                       # the Go arm
go run ./driver -corpora /tmp/corpora -go-bin /tmp/goarm \
    -rust-bin rustarm/target/release/rustarm                  # ~20s, 24 cells
```

## What is being compared

One entry point on each side, doing the same work and returning the same thing:

| | signature | returns |
|---|---|---|
| `fast.EncodeOrdinary` | `func (*Tiktoken) EncodeOrdinary(string) []int` | token ids |
| `CoreBPE::encode_ordinary` | `fn encode_ordinary(&self, &str) -> Vec<Rank>` | token ids |

No special-token scan on either side, no piece strings, no counting shortcut.
The tokenizer comparison in [`../README.md`](../README.md) needs three columns
because the two libraries expose no identical call; this one does not.

The Rust arm is given every advantage. It is built `--release` with fat LTO,
one codegen unit, `panic = "abort"` and `-C target-cpu=native`. Go has no
equivalent of that last flag, so the asymmetry runs against this fork: whatever
it wins here, it wins against Rust's best local build.

## Correctness gate

Before any cell is timed, both arms encode the corpus once and report FNV-1a
over their token ids. The driver skips the cell and prints `MISMATCH` if the
digests differ. Every cell in every table on this page passed, across four
vocabularies and seventeen corpora, which is also an independent check of this
fork against the reference implementation, not just against other Go ports.

## Method

The two implementations cannot be linked into one binary, and two sequential
sweeps on a loaded machine are not comparable: `../README.md` records a 2x
disagreement between sweeps of identical code. So both arms are long-lived child
processes with their vocabulary already loaded, and the driver alternates single
~5 ms bursts between them, taking the ratio inside the round. Drift lasting
longer than a round cancels. The order of the two arms flips every round, so
neither systematically runs first.

Reported figures are the median of the per-round ratios, with the interquartile
range in brackets. A ratio above 1 means this fork is faster, the same direction
every other table in `bench/` uses. Absolute `ns/op` and MB/s move with the
machine; the ratios are the part that reproduces.

Iteration counts are calibrated per arm, so both are measured over a comparable
slice of wall clock even where their per-call costs differ by 8x.

## Results

Apple M-series, load average ~5, Go 1.25, rustc 1.92. Median of 31 interleaved
rounds.

| encoding | workload | KiB | rust ns/op | fork ns/op | vs rust | rust MB/s | fork MB/s |
|---|---|---|---|---|---|---|---|
| cl100k_base | ascii_4k | 4 | 340 352 | 41 562 | **8.10x** [7.90-8.27] | 12 | 99 |
| cl100k_base | code_4k | 4 | 509 646 | 72 199 | **7.06x** [6.53-7.07] | 8 | 57 |
| cl100k_base | gosrc_256k | 256 | 30 929 958 | 9 730 417 | **3.21x** [3.05-3.40] | 8 | 27 |
| cl100k_base | mobydick_256k | 256 | 25 127 416 | 8 528 709 | **2.89x** [2.76-3.19] | 10 | 31 |
| cl100k_base | prose_64k | 64 | 6 386 500 | 2 737 020 | **2.31x** [2.23-2.35] | 10 | 24 |
| cl100k_base | prose_4k | 4 | 417 195 | 184 592 | **2.24x** [2.09-2.32] | 10 | 22 |
| cl100k_base | udhr_zho | 10 | 1 350 594 | 746 953 | **1.82x** [1.73-1.85] | 8 | 15 |
| cl100k_base | udhr_kor | 11 | 1 044 990 | 997 344 | 1.04x [0.92-1.12] | 11 | 12 |
| cl100k_base | hongloumeng_256k | 256 | 24 967 583 | 32 090 250 | 0.78x [0.73-0.80] | 10 | 8 |
| cl100k_base | longwords_4k | 4 | 210 699 | 333 870 | 0.63x [0.60-0.65] | 20 | 12 |
| cl100k_base | cjk_4k | 4 | 188 189 | 320 570 | 0.58x [0.53-0.62] | 22 | 13 |
| cl100k_base | udhr_jpn | 12 | 922 604 | 1 591 073 | 0.57x [0.55-0.60] | 14 | 8 |
| o200k_base | code_4k | 4 | 304 950 | 66 039 | **4.65x** [4.51-4.76] | 13 | 62 |
| o200k_base | ascii_4k | 4 | 188 738 | 47 079 | **3.97x** [3.92-4.06] | 22 | 87 |
| o200k_base | gosrc_256k | 256 | 20 289 042 | 7 370 875 | **2.70x** [2.54-3.15] | 13 | 36 |
| o200k_base | mobydick_256k | 256 | 18 141 583 | 10 121 917 | **1.77x** [1.47-2.02] | 14 | 26 |
| o200k_base | prose_64k | 64 | 4 472 875 | 3 346 229 | **1.34x** [1.32-1.37] | 15 | 20 |
| o200k_base | prose_4k | 4 | 283 003 | 223 783 | **1.27x** [1.24-1.30] | 15 | 18 |
| o200k_base | udhr_zho | 10 | 746 464 | 856 542 | 0.88x [0.85-0.90] | 15 | 13 |
| o200k_base | udhr_kor | 11 | 786 401 | 1 064 167 | 0.74x [0.70-0.76] | 15 | 11 |
| o200k_base | longwords_4k | 4 | 198 352 | 332 880 | 0.60x [0.56-0.61] | 21 | 12 |
| o200k_base | hongloumeng_256k | 256 | 24 562 458 | 47 744 583 | 0.51x [0.49-0.53] | 11 | 5 |
| o200k_base | cjk_4k | 4 | 180 284 | 432 802 | 0.41x [0.40-0.42] | 23 | 9 |
| o200k_base | udhr_jpn | 12 | 918 068 | 2 139 521 | 0.42x [0.40-0.45] | 14 | 6 |

Read the real corpora, not the generated ones. `ascii_4k` draws from a 60-word
list separated by single spaces and `code_4k` cycles eight punctuation-dense
lines: both are close to the best case for a scanner and the worst case for a
backtracking regex, which is why they read 4-8x where real English and real Go
source read 1.8-3.2x. The same caveat applies to every generated row in
[`../README.md`](../README.md) and is discussed there at length.

### Where each implementation wins, and why

The split is not random, and it is not a Go-versus-Rust result. Encoding is two
phases, and this fork changed exactly one of them:

Pre-tokenisation is the phase this fork changed. The reference splits text with
`fancy-regex`, a backtracking engine, on the same pattern `regexp2` runs in
upstream tiktoken-go; this fork replaced that with a hand-written scanner
compiled from the pattern string. Every workload where splitting dominates
(English prose, ASCII, source code, anything with punctuation and short words)
goes to this fork, by 1.3x to 3.2x on real text.

The byte-pair merge is the phase it did not change, and Rust's is better: `u32`
ranks, an `FxHashMap` keyed on byte slices, and a merge loop with fewer
allocations than the Go one. Every workload where the merge dominates goes to
Rust. Dense CJK is the clearest case, since ideographs miss the vocabulary and
land in the merge. `longwords_4k`, 24-64 random lowercase letters per word, is
the synthetic version of the same thing.

CJK is not one answer. Across the four real corpora it runs from 1.82x in this
fork's favour (`udhr_zho` under `cl100k_base`, a file that spaces out many of
its ideographs and so shifts work back into the scanner) to 0.42x against it
(`udhr_jpn` under `o200k_base`). `hongloumeng_256k` under `o200k_base` is 0.51x,
and at 256 KiB of continuous 18th-century Chinese prose it is the most
representative of the four. If your input is mostly CJK, the reference implementation is the
faster choice today, and the merge loop is where this fork would have to improve
to change that.

## Long inputs

The 4 KiB rows are where a per-call fixed cost would hide. It does not survive
contact with 4 MiB: throughput is flat and the ratio holds within a few percent
from 256 KiB up. Median of 21 interleaved rounds.

| encoding | workload | rust ns/op | fork ns/op | vs rust | rust MB/s | fork MB/s |
|---|---|---|---|---|---|---|
| cl100k_base | code_1m | 140 799 084 | 20 088 167 | **6.99x** [6.65-7.36] | 7 | 52 |
| cl100k_base | prose_256k | 25 709 042 | 11 303 708 | **2.29x** [2.15-2.33] | 10 | 23 |
| cl100k_base | prose_1m | 102 473 958 | 45 357 542 | **2.26x** [2.24-2.30] | 10 | 23 |
| cl100k_base | prose_4m | 410 235 042 | 181 640 167 | **2.25x** [2.23-2.30] | 10 | 23 |
| cl100k_base | cjk_1m | 67 576 333 | 107 799 625 | 0.63x [0.61-0.64] | 16 | 10 |
| o200k_base | code_1m | 82 443 125 | 17 717 667 | **4.67x** [4.59-4.75] | 13 | 59 |
| o200k_base | prose_1m | 70 784 833 | 53 015 542 | **1.34x** [1.32-1.36] | 15 | 20 |
| o200k_base | prose_4m | 284 687 291 | 213 182 250 | **1.34x** [1.32-1.35] | 15 | 20 |
| o200k_base | prose_256k | 17 720 916 | 13 296 209 | **1.33x** [1.30-1.34] | 15 | 20 |
| o200k_base | cjk_1m | 67 937 250 | 157 057 958 | 0.45x [0.41-0.47] | 15 | 7 |

Nothing here is a fixed per-call cost being amortised, in either direction:
`prose` sits at 2.25-2.29x on cl100k and 1.33-1.34x on o200k across a 16x range
of input size, and `cjk_1m` stays where `cjk_4k` was.

## Legacy encodings

Neither `p50k_base` nor `r50k_base` backs a current model, but each is a
distinct pre-tokenisation pattern the scanner is compiled from, so each is a
distinct set of branches to get wrong. Median of 31 interleaved rounds.

| encoding | workload | rust ns/op | fork ns/op | vs rust |
|---|---|---|---|---|
| p50k_base | code_4k | 608 036 | 101 706 | **6.01x** [5.68-6.22] |
| p50k_base | gosrc_256k | 36 270 667 | 10 857 875 | **3.34x** [3.17-3.58] |
| p50k_base | mobydick_256k | 23 898 834 | 7 866 042 | **3.05x** [2.85-3.19] |
| p50k_base | prose_4k | 386 794 | 183 159 | **2.14x** [2.06-2.20] |
| p50k_base | hongloumeng_256k | 24 702 166 | 27 305 125 | 0.90x [0.87-0.95] |
| r50k_base | code_4k | 633 745 | 105 578 | **6.06x** [5.78-6.26] |
| r50k_base | gosrc_256k | 36 313 417 | 10 794 167 | **3.33x** [3.18-3.48] |
| r50k_base | mobydick_256k | 26 635 500 | 9 127 917 | **2.96x** [2.80-3.16] |
| r50k_base | prose_4k | 393 406 | 179 202 | **2.20x** [2.03-2.28] |
| r50k_base | hongloumeng_256k | 24 663 292 | 26 298 875 | 0.94x [0.92-0.95] |

The two legacy vocabularies are 50k entries against cl100k's 100k and o200k's
200k, which shortens the merge and lifts even Chinese to 0.90-0.94x, close to
parity where o200k_base is 0.51x on the same text. That is the merge cost
showing up as a function of vocabulary size rather than of language.

## What this does not measure

- Allocation. Go's `allocs/op` has no counterpart on the Rust side, so the
  allocation tables in [`../README.md`](../README.md) stay a Go-to-Go
  comparison. The Rust arm returns `Vec<u32>`, this fork returns `[]int`, which
  is twice the bytes per token before anything else is counted.
- `Encode` with special tokens: both arms run `encode_ordinary` only. This
  fork's special-token path still uses regexp2 and is unchanged from upstream.
- Vocabulary load time, decoding, and anything concurrent. One thread, one
  resident vocabulary, ids out.
- Any machine but this one. Apple M-series, load average 5-10 throughout,
  Go 1.25.0, rustc 1.92.0. The absolute ns/op and MB/s are worth nothing
  elsewhere; the paired ratios are the part that should reproduce.

## Files

| | |
|---|---|
| [`goarm/`](goarm/) | Go round server: `load`, `open`, `check`, `run`, `quit` |
| [`rustarm/`](rustarm/) | the same protocol over `tiktoken-rs` 0.12.0 |
| [`driver/`](driver/) | interleaves the two, gates on the digest, prints the table |
| [`../cmd/dumpcorpora/`](../cmd/dumpcorpora/) | writes the shared corpora to disk |

