package bench

import (
	"fmt"
	"testing"
	"unicode/utf8"

	loader "github.com/pkoukk/tiktoken-go-loader"
	fast "github.com/s3bastiankoch/tiktoken-go-fast"
	"github.com/s3bastiankoch/tiktoken-go-fast/bench/internal/upstream"
	"github.com/tiktoken-go/tokenizer"
)

// This module compares tiktoken-go-fast against tiktoken-go/tokenizer, an
// independent Go port with its own embedded vocabularies. It lives in a nested
// module so the library itself keeps its dependency set.
//
// Both encodings are measured, and the split matters: the hand-written scanner
// in this fork covers cl100k_base but not o200k_base, and o200k_base is what
// every gpt-4o, gpt-4.1 and gpt-5.x model uses.
//
// The two libraries do not offer one identical entry point, so three arms are
// reported rather than one:
//
//	fast/EncodeOrdinary   -> []int
//	tokenizer/Encode      -> []uint and []string (strictly more work)
//	tokenizer/Count       -> int only (strictly less work)
//
// Count and Encode bracket what a like-for-like API would cost, so the true
// ratio for equivalent work sits between the two.

type arm struct {
	name string
	fn   func(string)
}

type namedPair struct {
	name string
	fast *fast.Tiktoken
	up   *upstream.Tiktoken
	tok  tokenizer.Codec
}

// encodings returns the two encodings worth timing: everything current is
// o200k_base, and cl100k_base still carries the embeddings models.
func encodings() []namedPair { return load(tokenizer.Cl100kBase, tokenizer.O200kBase) }

// gateEncodings adds the two legacy encodings. They are not benchmarked, since
// no current model uses them, but the scanner is compiled from the pattern
// string, so each one is a distinct set of branches and belongs in the
// correctness gate.
func gateEncodings() []namedPair {
	return load(tokenizer.Cl100kBase, tokenizer.O200kBase, tokenizer.P50kBase, tokenizer.R50kBase)
}

func load(encs ...tokenizer.Encoding) []namedPair {
	fast.SetBpeLoader(loader.NewOfflineLoader())
	upstream.SetBpeLoader(loader.NewOfflineLoader())
	out := make([]namedPair, 0, len(encs))
	for _, e := range encs {
		f, err := fast.GetEncoding(string(e))
		if err != nil {
			panic(fmt.Sprintf("fast %s: %v", e, err))
		}
		u, err := upstream.GetEncoding(string(e))
		if err != nil {
			panic(fmt.Sprintf("upstream %s: %v", e, err))
		}
		c, err := tokenizer.Get(e)
		if err != nil {
			panic(fmt.Sprintf("tokenizer %s: %v", e, err))
		}
		out = append(out, namedPair{string(e), f, u, c})
	}
	return out
}

// TestSameTokens is the gate on the comparison: a speed number against a
// library that produces different tokens would not mean anything.
func TestSameTokens(t *testing.T) { sameTokens(t, Workloads()) }

// TestSameTokensAllEncodings runs the gate on p50k_base and r50k_base too.
func TestSameTokensAllEncodings(t *testing.T) { sameTokensWith(t, gateEncodings(), Workloads()) }

func sameTokens(t *testing.T, workloads []Workload) {
	sameTokensWith(t, encodings(), workloads)
}

func sameTokensWith(t *testing.T, encs []namedPair, workloads []Workload) {
	t.Helper()
	for _, e := range encs {
		for _, w := range workloads {
			t.Run(e.name+"/"+w.Name, func(t *testing.T) {
				got := e.fast.EncodeOrdinary(w.Text)
				want, _, err := e.tok.Encode(w.Text)
				if err != nil {
					t.Fatalf("tokenizer.Encode: %v", err)
				}
				if len(got) != len(want) {
					t.Fatalf("token count: fast %d, tokenizer %d", len(got), len(want))
				}
				for i := range got {
					if uint(got[i]) != want[i] {
						t.Fatalf("token %d: fast %d, tokenizer %d", i, got[i], want[i])
					}
				}
				up := e.up.EncodeOrdinary(w.Text)
				if len(up) != len(got) {
					t.Fatalf("token count: fast %d, upstream %d", len(got), len(up))
				}
				for i := range got {
					if up[i] != got[i] {
						t.Fatalf("token %d: fast %d, upstream %d", i, got[i], up[i])
					}
				}
			})
		}
	}
}

// TestCorpusStats prints how densely each corpus packs into the vocabulary.
// It is the number that decides whether a CJK result generalises: a corpus at
// ~1.5 bytes/token is spending its time in the byte-pair merge, one at 3+ is
// spending it in the pre-tokeniser.
func TestCorpusStats(t *testing.T) {
	fmt.Printf("%-18s %10s %10s %13s %13s\n", "workload", "bytes", "runes", "b/tok cl100k", "b/tok o200k")
	var all []Workload
	for _, i := range []int{0, 3, 4} { // prose_4k, cjk_4k, code_4k
		all = append(all, Workloads()[i])
	}
	all = append(all, Workload{"cjk_1m", LongWorkloads()[3].Text})
	all = append(all, RealWorkloads()...)
	encs := encodings()
	for _, w := range all {
		cl := float64(len(w.Text)) / float64(len(encs[0].fast.EncodeOrdinary(w.Text)))
		o2 := float64(len(w.Text)) / float64(len(encs[1].fast.EncodeOrdinary(w.Text)))
		fmt.Printf("%-18s %10d %10d %13.2f %13.2f\n", w.Name, len(w.Text), utf8.RuneCountInString(w.Text), cl, o2)
	}
}

var (
	sinkInts   []int
	sinkUints  []uint
	sinkPieces []string
	sinkCount  int
)

func BenchmarkCompare(b *testing.B) {
	for _, e := range encodings() {
		arms := []arm{
			{"fast_EncodeOrdinary", func(s string) { sinkInts = e.fast.EncodeOrdinary(s) }},
			{"upstream_EncodeOrdinary", func(s string) { sinkInts = e.up.EncodeOrdinary(s) }},
			{"fast_Encode", func(s string) { sinkInts = e.fast.Encode(s, nil, nil) }},
			{"upstream_Encode", func(s string) { sinkInts = e.up.Encode(s, nil, nil) }},
			{"tokenizer_Encode", func(s string) { sinkUints, sinkPieces, _ = e.tok.Encode(s) }},
			{"tokenizer_Count", func(s string) { sinkCount, _ = e.tok.Count(s) }},
		}
		for _, w := range Workloads() {
			for _, a := range arms {
				b.Run(e.name+"/"+w.Name+"/"+a.name, func(b *testing.B) {
					b.SetBytes(int64(len(w.Text)))
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						a.fn(w.Text)
					}
				})
			}
		}
	}
}

// TestSameTokensLong is the same gate at 256 KiB to 4 MiB.
func TestSameTokensLong(t *testing.T) { sameTokens(t, LongWorkloads()) }

// TestSameTokensReal is the same gate on the vendored real-text corpora.
func TestSameTokensReal(t *testing.T) { sameTokens(t, RealWorkloads()) }

// BenchmarkLong reports allocation, which is load-independent and is where a
// super-linear term would show first.
func BenchmarkLong(b *testing.B) {
	for _, e := range encodings() {
		for _, w := range LongWorkloads() {
			for _, a := range []arm{
				{"fast_EncodeOrdinary", func(s string) { sinkInts = e.fast.EncodeOrdinary(s) }},
				{"tokenizer_Encode", func(s string) { sinkUints, sinkPieces, _ = e.tok.Encode(s) }},
			} {
				b.Run(e.name+"/"+w.Name+"/"+a.name, func(b *testing.B) {
					b.SetBytes(int64(len(w.Text)))
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						a.fn(w.Text)
					}
				})
			}
		}
	}
}
