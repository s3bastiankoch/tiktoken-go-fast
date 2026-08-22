# tiktoken-go-fast vs tiktoken-go/tokenizer

Two comparisons, both in-process and interleaved:

- against the fork point, `pkoukk/tiktoken-go` at `200cf60`, vendored
  verbatim from upstream into [`internal/upstream`](internal/upstream) so both
  implementations link into one binary and share one vocabulary;
- against [tiktoken-go/tokenizer](https://github.com/tiktoken-go/tokenizer)
  v0.8.1, an independent port with its own embedded vocabularies and its own
  pre-tokeniser, a second opinion that shares no code with either arm.

A third comparison, against openai/tiktoken's own Rust core, lives in
[`vsrust/`](vsrust/). It cannot be in-process, because the arms are
separate processes, so it is interleaved round by round instead, and it is where to
look for how this fork stands against the reference implementation rather than
against other Go ports.

Nested module, so the library itself keeps its dependency set. It requires
Go 1.26 because `tiktoken-go/tokenizer` v0.8.1 does; the library itself still
builds on Go 1.19.

```sh
go test -run TestSameTokens .          # gate: identical token IDs, both encodings
go test -run TestSameTokensLong .      # same gate at 256 KiB - 4 MiB
go test -run TestPairedRatio -v .      # ratios at 4 KiB, interleaved (~15s)
go test -run TestPairedRatioAllEncodings -v .  # p50k_base and r50k_base too
go test -run TestPairedRatioLong -v .  # ratios at 256 KiB - 4 MiB (~1min)
go test -run TestPairedRatioReal -v .  # ratios on real CJK text (~15s)
go test -run TestCorpusStats -v .      # bytes/token per corpus
go test -run xxx -bench . -count=9 .   # raw ns/op, B/op, allocs/op (~3min)
```

## Result

**Against the fork point: 4.1-5.3x on prose across all four encodings, 10.3x on
real source code, 1.3-2.2x on real CJK. Against tokenizer: 2.8-4.8x on English
prose and source code, 1.0-1.8x on CJK.**
The generated workloads below range from 1.0x to 9.1x; they isolate individual
cost terms, and their extremes do not survive contact with real corpora. See
[Real text](#real-text) before quoting a number.

Every table on this page is the vs-tokenizer arm. The fork-point figures in
the summary above come from the `vs upstr` column that `TestPairedRatio` and
`TestPairedRatioAllEncodings` print alongside it; run them to see both arms at
once, and see the root [`README.md`](../README.md) for the fork-point table.

Median of 31 interleaved rounds; a ratio above 1 means this fork is faster.
Apple M-series, Go 1.26. The machine carried unrelated load throughout (load
average ~17), which is why the ratios are paired rather than taken from separate
sweeps. Absolute ns/op moves with the machine; the paired ratios are the part
that reproduces.

| encoding | workload | vs `Encode` | vs `Count` |
|---|---|---|---|
| o200k_base | code_4k | **9.13x** | 7.38x |
| o200k_base | ascii_4k | **7.92x** | 7.71x |
| o200k_base | prose_64k | 2.45x | 2.32x |
| o200k_base | prose_4k | 2.35x | 2.29x |
| o200k_base | longwords_4k | 1.17x | 0.77x |
| o200k_base | cjk_4k | 0.98x | 0.87x |
| cl100k_base | ascii_4k | **7.64x** | 7.08x |
| cl100k_base | code_4k | **5.62x** | 5.34x |
| cl100k_base | prose_64k | 2.27x | 2.09x |
| cl100k_base | prose_4k | 2.20x | 2.21x |
| cl100k_base | longwords_4k | 1.26x | 1.15x |
| cl100k_base | cjk_4k | 1.17x | 1.00x |

The four rows near parity are the ones where the pre-tokeniser is not the cost:
`cjk_4k` and `longwords_4k` under both encodings have interquartile ranges
touching or spanning 1.0. `o200k_base/longwords_4k` takes ~3.2 ms in this fork
and ~3.2 ms in tokenizer, an order of magnitude more than the same text under
`cl100k_base` in either library. That is the byte-pair merge on a 200k-entry
vocabulary, and no scanner touches it.

### What this table looked like before

The first run of this comparison, against the revision that hand-wrote one
encoding's scanner and fell back to regexp2 for the rest:

| encoding | workload | vs `Encode` then | vs `Encode` now |
|---|---|---|---|
| o200k_base | code_4k | 0.39x | **9.13x** |
| o200k_base | ascii_4k | 0.40x | **7.92x** |
| o200k_base | prose_4k | 0.45x | **2.35x** |
| o200k_base | prose_64k | 0.58x | **2.45x** |
| o200k_base | cjk_4k | 0.67x | 0.98x |
| o200k_base | longwords_4k | 0.89x | 1.17x |

`cl100k_base` is unchanged within noise across the two revisions, so the fast
path was extended rather than traded away.

## Long inputs

The 4 KiB table is where a per-call fixed cost would hide. It does not survive
contact with 4 MiB, and neither does anything super-linear: throughput is flat
from 256 KiB to 4 MiB (23.1, 21.6, 22.7 MB/s on cl100k prose) and the ratio
holds or improves with size. Median of 11 interleaved rounds, IQRs at these
sizes are within a few percent.

| encoding | workload | vs `Encode` | vs `Count` |
|---|---|---|---|
| o200k_base | code_1m | **8.57x** | 8.09x |
| o200k_base | prose_4m | 2.64x | 2.47x |
| o200k_base | prose_1m | 2.62x | 2.42x |
| o200k_base | prose_256k | 2.39x | 2.32x |
| o200k_base | cjk_1m | 1.00x | 0.85x |
| cl100k_base | code_1m | **6.42x** | 5.83x |
| cl100k_base | prose_4m | 2.52x | 2.30x |
| cl100k_base | prose_1m | 2.45x | 2.27x |
| cl100k_base | prose_256k | 2.32x | 2.25x |
| cl100k_base | cjk_1m | 1.26x | 1.02x |

Allocation is where the gap is widest, and it is load-independent, so these
numbers need no pairing:

| workload | this fork | tokenizer `Encode` | |
|---|---|---|---|
| cl100k prose_4m | 73 MB, 293 219 allocs | 451 MB, 4 999 254 allocs | 17x fewer |
| o200k prose_4m | 55 MB, 251 682 allocs | 436 MB, 5 009 433 allocs | 20x fewer |
| o200k code_1m | 8 MB, 7 773 allocs | 155 MB, 2 168 429 allocs | 279x fewer |

Allocations grow linearly with input in both libraries (16x the bytes, 15.6x the
allocations), so this is a constant-factor difference that does not close.

The exception is CJK, but the synthetic corpus overstates it. `cjk_1m`
under `o200k_base` is 1.00x against `Encode` and 0.85x against `Count`, because
uniformly sampled ideographs are mostly rare characters that miss the vocabulary
and land in the merge loop. Real text does better; see below.

## Real text

The generated workloads are not neutral, in either direction: `cjk_*` samples
ideographs uniformly and lands on rare characters that miss the vocabulary,
while `ascii_4k` and `code_4k` draw from a short fragment list and repeat far
more than real text does. Both distortions show up in the ratio. Measured
density, bytes per token:

| corpus | cl100k_base | o200k_base |
|---|---|---|
| `cjk_4k` / `cjk_1m` (generated) | 1.45 | 1.70 |
| `udhr_zho` (real Chinese) | 2.02 | 2.82 |
| `hongloumeng_256k` (real Chinese) | 2.01 | 2.89 |
| `udhr_jpn` (real Japanese) | 2.48 | 3.33 |
| `udhr_kor` (real Korean) | 2.45 | 4.19 |
| `prose_4k` (generated) | 3.01 | 3.58 |
| `code_4k` (generated) | 3.09 | 3.05 |
| `gosrc_256k` (real Go source) | 3.54 | 3.54 |
| `mobydick_256k` (real English) | 4.03 | 4.07 |

Real CJK packs 1.4-2.5x more densely, which moves time out of the byte-pair
merge and back into pre-tokenisation. The ratio moves with it, from a tie to a
modest win. Median of 31 interleaved rounds:

| encoding | corpus | vs `Encode` | vs `Count` | generated equivalent |
|---|---|---|---|---|
| o200k_base | gosrc_256k | **4.76x** | 4.45x | `code_4k` 9.13x |
| o200k_base | mobydick_256k | 2.89x | 2.73x | `prose_4k` 2.35x |
| o200k_base | udhr_zho | 1.83x | 1.70x | |
| o200k_base | udhr_kor | 1.28x | 1.15x | |
| o200k_base | hongloumeng_256k | 1.11x | 1.05x | `cjk_1m` 1.00x |
| o200k_base | udhr_jpn | 1.02x | 0.96x | |
| cl100k_base | gosrc_256k | **3.10x** | 2.78x | `code_4k` 5.62x |
| cl100k_base | mobydick_256k | 2.79x | 2.63x | `prose_4k` 2.20x |
| cl100k_base | udhr_zho | 1.53x | 1.34x | |
| cl100k_base | udhr_kor | 1.39x | 1.24x | |
| cl100k_base | hongloumeng_256k | 1.25x | 1.15x | `cjk_1m` 1.26x |
| cl100k_base | udhr_jpn | 1.05x | 1.02x | |

Read the last column before quoting anything from the generated tables:

Real prose beats its generated stand-in: 2.89x against `prose_4k`'s 2.35x.

Real source code is roughly half the generated figure. `gosrc_256k` is 4.76x
where `code_4k` claimed 9.13x, and 3.10x where it claimed 5.62x. The generated
corpus cycles eight punctuation-dense lines, which is close to the best case for
a scanner and close to the worst case for a backtracking regex. `ascii_4k`'s
7.6-7.9x is the same artefact: a 60-word vocabulary separated by single spaces
is not English.

CJK lands at 1.0-1.8x, which is neither the wash the generator suggested nor the
general result. Japanese under `o200k_base` is the weakest case measured
anywhere here, 1.02x with an IQR touching parity.

`udhr_zho` reads high for a reason worth knowing: part of that file spaces out
its ideographs individually, which shortens pieces and shifts work into the
pre-tokeniser. `hongloumeng_256k`, 256 KiB of continuous 18th-century Chinese
prose, is the more representative Chinese number, 1.11x under `o200k_base`.

**The defensible summary on real text: 2.8-4.8x on English prose and source
code, 1.0-1.8x on CJK.** The generated tables above are kept because they
isolate cost terms the real corpora mix together, not because their headline
numbers generalise.

Provenance and licences for the vendored corpora are in
[`testdata/README.md`](testdata/README.md).

## Why three columns and not two

The libraries expose no identical entry point:

| | returns | work |
|---|---|---|
| `fast.EncodeOrdinary` | `[]int` | ids only |
| `tokenizer.Encode` | `[]uint, []string` | ids **and** the piece strings |
| `tokenizer.Count` | `int` | neither, just counts |

`Encode` does strictly more than `EncodeOrdinary`; `Count` does strictly less. A
like-for-like ratio sits between the two columns, and both are reported so the
bracket is visible rather than chosen.

## Correctness first

`TestSameTokens` asserts the two libraries emit identical token IDs on all six
workloads under both encodings before any timing is read. They do. A speed
number against a library that tokenises differently would mean nothing.

## Method

Sequential `go test -bench` arms are not comparable on a loaded machine: two full
sweeps of the same benchmarks disagreed by 2x on individual cells while
`allocs/op` stayed identical to the byte: drift, not code. `TestPairedRatio`
measures every arm once per round inside the same ~5 ms window and takes the
ratio within the round, so drift lasting longer than a round cancels. The
reported figure is the median of 31 per-round ratios; the bracket printed after
each row is the interquartile range.

Absolute `ns/op` from `BenchmarkCompare` is still worth reading for allocation
counts, which are load-independent, and those are where the cl100k gap comes
from: 288 allocs/op against 4 855 on `prose_4k`, and 1 against 6 332 on
`ascii_4k`.

## Workloads

Regenerated from `bench_test.go` in the root module with the same seeds,
so a cross-library number lines up with an in-repo one. The one change: `fill`
does not slice back to exactly n bytes, since that cut can land inside a
multi-byte rune and two libraries may treat invalid UTF-8 differently without
either being wrong.
