# tiktoken-go-fast

A fork of [pkoukk/tiktoken-go](https://github.com/pkoukk/tiktoken-go) with a
hand-written pre-tokeniser in place of the backtracking regex engine.
**4.6-5.2x faster encoding, 17x fewer allocations**, byte-identical output.

Not affiliated with or endorsed by the upstream project. Forked at
[`200cf60`](https://github.com/pkoukk/tiktoken-go/commit/200cf60), MIT licence
retained.

## Numbers

`prose_4k`, Apple M-series. Not a single run: the fork-point implementation is
linked into the same binary and both arms are measured round by round, so
machine drift cancels instead of landing on whichever arm ran while the machine
was busy. Median of 31 rounds. See [`bench/`](bench/).

Speed, as paired median ratios against the fork-point implementation:

| | cl100k_base | o200k_base |
|---|---|---|
| `EncodeOrdinary` | **4.61x** | **5.20x** |
| exported `Encode()` | 3.51x* | not measured |

\* sequential `go test -bench`, medians of 3, not paired.

Allocation, as absolute `go test -bench` counts, which are load-independent and
need no pairing (cl100k_base, `prose_4k`):

| | fork point | this fork | |
|---|---|---|---|
| bytes allocated | 427 120 B/op | 63 832 B/op | **6.7x less** |
| allocations | 5 022 /op | 286 /op | **17.6x fewer** |

`Encode()` gains less than `EncodeOrdinary` because it also runs the
special-token regex, which this fork does not touch. That path is unchanged from
upstream and is the obvious next thing to look at.

### Per encoding

All five shipped encodings take the fast path. Between them they carry four
vocabularies (`p50k_edit` is `p50k_base`'s vocabulary plus three special tokens)
and three distinct pre-tokenisation patterns. The pattern is what the scanner is
keyed on: it is compiled from the pattern string at construction time rather
than hard-coded for one encoding.

Every figure below is measured in this repository by `bench/`, which links the
fork-point implementation (`200cf60`, vendored verbatim from upstream under
[`bench/internal/upstream`](bench/internal/upstream)) into the same binary and
interleaves the two arms round by round. Median of 31 rounds:

| encoding | models | generated `prose_4k` | 256 KiB of *Moby-Dick* |
|---|---|---|---|
| `o200k_base` | **gpt-4o, gpt-4.1, gpt-5.x, o-series** | **5.20x** | 4.92x |
| `cl100k_base` | gpt-4, gpt-3.5-turbo, embeddings | 4.61x | 5.00x |
| `p50k_base` | text-davinci-002/003, codex | 4.09x | 4.57x |
| `r50k_base` | gpt-3, davinci | 4.11x | 4.45x |

Real source code goes further, 10.29x on 256 KiB of Go standard library under
`o200k_base`. CJK goes less far, 1.26-2.22x on real Chinese, Japanese and
Korean. The generated `ascii_4k` and `code_4k` workloads reach 18-23x, which is
an artefact of their fragment lists rather than a number to quote.

An earlier revision of this fork covered only `cl100k_base` and `p50k_base` and
fell back to regexp2 for `o200k_base`, which left every current model at 1.03x.
That is what [`bench/`](bench/) was built to catch, and it caught it.

The commit that landed the current scanner reported 5.8x on `o200k_base` from
its own harness. Measured here it is 5.20x, interquartile range 5.14-5.31.

### Against the other Go port

Measured against [tiktoken-go/tokenizer](https://github.com/tiktoken-go/tokenizer)
v0.8.1, medians of 31 interleaved rounds, identical token IDs gated before any
timing is read:

| corpus | `o200k_base` | `cl100k_base` |
|---|---|---|
| 256 KiB of Go stdlib source | **4.76x** | 3.10x |
| 256 KiB of *Moby-Dick* | **2.89x** | 2.79x |
| 256 KiB of 紅樓夢 | 1.11x | 1.25x |
| UDHR, Japanese | 1.02x | 1.05x |

These are real corpora, vendored under `bench/testdata/`. The generated
workloads in the same harness span 1.0-9.1x, and their extremes are artefacts:
`code_4k` cycles eight punctuation-dense lines and reads 9.13x where real Go
source reads 4.76x, while `cjk_4k` samples ideographs uniformly and reads 0.98x
where real Chinese reads 1.11x. They are kept because they separate cost terms
that real text mixes together, not because they generalise.

The ratios hold at scale. Repeated from 256 KiB to 4 MiB, prose goes 2.39x,
2.62x, 2.64x on `o200k_base`, and throughput stays flat, so nothing here is a
fixed per-call cost being amortised. Allocation is the widest gap and does not
close with size: 4 MiB of prose costs 252 k allocations here against 5.0 M in
tokenizer.

CJK is the weak case. Real Chinese, Japanese and Korean run 1.0-1.8x. Real text
packs 1.4-2.5x more densely into the vocabulary than the generated CJK corpus,
which moves work out of the byte-pair merge, but not far enough to reach the
prose figure. Japanese under `o200k_base` is the weakest number measured
anywhere here, 1.02x, with an interquartile range touching parity. Method,
caveats and the full tables are in [`bench/`](bench/).

### Against the Rust reference

openai/tiktoken's own Rust `CoreBPE`, reached through
[tiktoken-rs](https://crates.io/crates/tiktoken-rs) v0.12.0, which vendors that
core verbatim and ships the same vocabularies. Both sides run the same entry
point, `EncodeOrdinary` against `encode_ordinary`, ids in and ids out, with
identical token IDs gated before any timing. Medians of 31 interleaved rounds.
The Rust arm is built `--release` with fat LTO, one codegen unit and
`-C target-cpu=native`, which Go has no equivalent of, so the build asymmetry
runs against this fork:

| corpus | `o200k_base` | `cl100k_base` |
|---|---|---|
| 256 KiB of Go stdlib source | **2.70x** | **3.21x** |
| 256 KiB of *Moby-Dick* | 1.77x | 2.89x |
| 4 KiB of generated prose | 1.27x | 2.24x |
| 256 KiB of 紅樓夢 | 0.51x | 0.78x |
| UDHR, Japanese | 0.42x | 0.57x |

**English prose and source code run 1.3-3.2x faster than the reference
implementation; continuous Chinese and Japanese run up to 2.4x slower.** One
cause explains both halves: this fork replaced the pre-tokeniser and left the
byte-pair merge alone. Splitting is a hand-written scanner here against
`fancy-regex` there, which is why prose and code go this way. Merging is
unchanged Go against Rust's `u32` ranks and `FxHashMap`, and dense CJK misses
the vocabulary and lands in the merge, which is why it goes the other way. The
ratios hold from 256 KiB to 4 MiB and across all four vocabularies; tables,
method and the harness are in [`bench/vsrust/`](bench/vsrust/).

## Correctness

The upstream implementation is the oracle. `equiv_test.go` keeps a verbatim copy
of the original `bytePairMerge` and `encodeOrdinaryNative` from the fork point;
`allenc_test.go` runs that comparison on every shipped encoding, and `bench/`
adds a second, independent implementation as a witness:

- 1 683 810 fuzz executions (`FuzzEncodeOrdinaryAllEncodings`, native
  `go test -fuzz`), each input compared against the frozen implementation under
  all four vocabularies. 6.7 M comparisons, zero divergences.
- All four vocabularies on 322 texts each: CJK, emoji and ZWJ sequences, mixed
  scripts, code, contractions in both cases, CRLF, full-width forms, and
  adversarial rune boundaries. `p50k_base` and `r50k_base` share the GPT-2
  pattern, so three scanners cover the four.
- Invalid UTF-8 and special-token segmentation, the two input classes the fuzz
  target skips: encoding around `<|endoftext|>` must equal encoding each segment
  ordinarily with the special id spliced in.
- An independent port: `bench/` asserts byte-identical token IDs against
  [tiktoken-go/tokenizer](https://github.com/tiktoken-go/tokenizer) on 17 corpora
  under four encodings before it reads a single timing number.
- Decode round-trip invariant, plus the upstream test suite unchanged.

A custom pattern passed to `NewCoreBPE` is matched exactly against the three
known ones and otherwise takes the regexp2 path, so an encoding this fork has
never seen is slow rather than wrong. That fallback is itself tested.

```sh
go test ./...                                              # gate
go test -run xxx -fuzz FuzzEncodeOrdinaryAllEncodings      # differential fuzzing
cd bench && go test .                                      # vs an independent port
```

## Benchmarks you can trust

Upstream ships `BenchmarkEncoding` in
[`benchmark_test.go`](https://github.com/pkoukk/tiktoken-go/blob/200cf60/benchmark_test.go),
which never reads `b.N` and never resets the timer: it encodes a fixed text four
times, at 1x, 10x, 100x and 1000x its length, and `fmt.Printf`s each size to
stdout. `go test -bench` then divides whatever that cost by a `b.N` the function
ignored. Every published figure for this library traces back to it.

That file is not carried over here. `bench_test.go` replaces it. (The one
upstream benchmark this fork does keep, `test/benchmark_test.go`, is a different
function and does drive `b.N` correctly.)

`bench_test.go` provides `b.N`-driven benchmarks over held-out workloads chosen
to separate the cost terms: input length, script (ASCII vs 3-byte CJK, where
rune-indexing pays triple), punctuation density (regex backtracking), and
unbroken word length (the quadratic merge).

```sh
go test -run xxx -bench . -benchtime 2s
```

Vocabulary comes from [tiktoken-go-loader](https://github.com/pkoukk/tiktoken-go-loader)'s
embedded assets, so benchmarks and tests need no network and see identical ranks
on every run.

## Models

`EncodingForModel` follows upstream openai/tiktoken's `tiktoken/model.py`
(checked 2026-08-22), including the prefix rules that make new releases work
without a library update:

```go
enc, _ := tiktoken.EncodingForModel("gpt-5.5")   // o200k_base, no table entry needed
```

The `"gpt-5"` prefix carries no trailing dash, so every gpt-5 point release and
size variant resolves. `o1-`, `o3-`, `o4-mini-`, `chatgpt-4o-`, the Azure
`gpt-35-turbo-` names and the `ft:` fine-tune prefixes are covered too.

Two things differ from a naive port of upstream's table. The first is that the
longest prefix wins. Upstream relies on Python dict insertion order to put
`ft:gpt-4o` ahead of `ft:gpt-4`; Go randomises map iteration, so the prefixes
are sorted by length at init instead. Without that, a fine-tuned gpt-4o model
resolves to cl100k_base on roughly half of all runs.

The second is that `gpt-oss-` is deliberately absent. It needs `o200k_harmony`,
which this library does not ship. Falling through to a "no encoding for model"
error beats returning confident, wrong token counts.

## Using it

Either as a normal dependency:

```sh
go get github.com/s3bastiankoch/tiktoken-go-fast
```

```go
import tiktoken "github.com/s3bastiankoch/tiktoken-go-fast"
```

The package is named `tiktoken`, not `tiktokengofast`, so the import above is
spelled explicitly. goimports will do it for you.

Or as a drop-in for an existing `pkoukk/tiktoken-go` dependency, without touching
a single import statement:

```
// go.mod
require github.com/pkoukk/tiktoken-go v0.1.7
replace github.com/pkoukk/tiktoken-go => github.com/s3bastiankoch/tiktoken-go-fast v0.1.0
```

Go resolves imports of the replaced path to this module, so existing
`import "github.com/pkoukk/tiktoken-go"` lines keep compiling and pick up the
scanner. Note that `replace` applies only to the main module: it is ignored in
anything that depends on *you*, so a library should import this module directly
rather than relying on a replace its own consumers will not inherit.

One API note carried over unchanged: `Encode` panics when the text contains a
special token listed in its `disallowedSpecial` argument. Passing `nil` for both
lists, or calling `EncodeOrdinary`, never panics. That is upstream's contract and
this fork keeps it, so a drop-in replacement stays drop-in. Wrap the call if a
panic on untrusted input is not acceptable in your caller.

## Licence

MIT, as upstream. `LICENSE` is unmodified, so copyright remains with the
original authors. The scanner, harness and benchmarks added here are offered
under the same terms. Upstream's own README is kept as `README-upstream.md`.
