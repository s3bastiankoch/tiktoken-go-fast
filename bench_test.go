package tiktoken

import (
	"math/rand"
	"strings"
	"sync"
	"testing"
)

// Benchmarks for the tiktoken hot path.
//
// Upstream ships BenchmarkEncoding in benchmark_test.go, which never reads b.N
// and never resets the timer: it encodes one text at four fixed sizes and
// fmt.Printfs each, so `go test -bench` divides that fixed cost by a b.N the
// function ignored. Nothing downstream of it means anything. That file is not
// carried over; these replace it.
//
//	https://github.com/pkoukk/tiktoken-go/blob/200cf60/benchmark_test.go
//
// Workloads are held out from the equivalence gate's inputs and span what the
// cost actually depends on: input length (regex walk and rune conversion are
// linear in it, byte-pair merging is quadratic in piece length), script (ASCII
// vs multi-byte, where a rune-index implementation pays twice), and how much of
// the text hits the vocab directly instead of falling through to the merge loop.

type benchWorkload struct {
	name string
	text string
}

var (
	benchOnce sync.Once
	benchWork []benchWorkload
)

func workloads() []benchWorkload {
	benchOnce.Do(func() {
		rng := rand.New(rand.NewSource(7919))
		benchWork = []benchWorkload{
			{"short_prose", corpusText(rng, 128)},
			{"prose_4k", corpusText(rng, 4<<10)},
			{"prose_64k", corpusText(rng, 64<<10)},
			{"ascii_4k", asciiText(rand.New(rand.NewSource(11)), 4<<10)},
			{"cjk_4k", cjkText(rand.New(rand.NewSource(13)), 4<<10)},
			{"code_4k", codeText(rand.New(rand.NewSource(17)), 4<<10)},
			{"longwords_4k", longWordText(rand.New(rand.NewSource(19)), 4<<10)},
		}
	})
	return benchWork
}

var sinkTokens []int

// BenchmarkEncodeOrdinary is the end-to-end path: pre-tokenisation regex, rune
// slicing, vocab lookup, byte-pair merge. b.SetBytes makes the sub-benchmarks
// comparable across input sizes.
func BenchmarkEncodeOrdinary(b *testing.B) {
	enc := testEncoding(b)
	for _, w := range workloads() {
		b.Run(w.name, func(b *testing.B) {
			b.SetBytes(int64(len(w.text)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkTokens = enc.bpe.encodeOrdinaryNative(w.text)
			}
		})
	}
}

// BenchmarkBytePairEncode isolates the merge loop from the regex walk: the
// pieces are the ones the pre-tokeniser actually produces on the prose corpus,
// keeping only those that miss the vocab and therefore reach the merge.
func BenchmarkBytePairEncode(b *testing.B) {
	enc := testEncoding(b)
	ranks := enc.bpe.encoder
	pieces := mergePieces(enc, corpusText(rand.New(rand.NewSource(23)), 64<<10))
	if len(pieces) == 0 {
		b.Skip("no piece misses the vocab; nothing reaches the merge loop")
	}
	var total int
	for _, p := range pieces {
		total += len(p)
	}
	b.SetBytes(int64(total))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range pieces {
			sinkTokens = bytePairEncode(p, ranks)
		}
	}
}

// mergePieces returns the pre-tokenised pieces that are not single vocab hits.
func mergePieces(enc *Tiktoken, text string) [][]byte {
	var out [][]byte
	runes := []rune(text)
	for _, mat := range findRegex2AllStringMatchIndex(text, enc.bpe.tlRegex) {
		piece := refCutRunes(runes, mat[0], mat[1])
		if _, ok := enc.bpe.encoder[piece]; ok {
			continue
		}
		out = append(out, []byte(piece))
		if len(out) >= 4096 {
			break
		}
	}
	return out
}

// BenchmarkEncode is the exported API, allowed-special handling included: what a
// gateway actually calls per request.
func BenchmarkEncode(b *testing.B) {
	enc := testEncoding(b)
	for _, w := range workloads() {
		b.Run(w.name, func(b *testing.B) {
			b.SetBytes(int64(len(w.text)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkTokens = enc.Encode(w.text, nil, nil)
			}
		})
	}
}

// --- workload generators -----------------------------------------------------
// Held out from the gate's inputs: the gate pushes on correctness edges, these
// push on the cost model.

func fill(n int, next func(*strings.Builder)) string {
	var b strings.Builder
	for b.Len() < n {
		next(&b)
	}
	return b.String()[:n]
}

// asciiText: the cheap case, one byte per rune, most pieces hit the vocab.
func asciiText(rng *rand.Rand, n int) string {
	words := strings.Fields("the of and to in a is that for it as was with be by on not he i this " +
		"but had are they from or an she which you one we all were her would there their")
	return fill(n, func(b *strings.Builder) {
		b.WriteString(words[rng.Intn(len(words))])
		b.WriteByte(' ')
	})
}

// cjkText: three bytes per rune, so every rune-index conversion costs 3x what a
// byte-index one would, and few pieces hit the vocab whole.
func cjkText(rng *rand.Rand, n int) string {
	return fill(n, func(b *strings.Builder) {
		for i := 0; i < 8; i++ {
			b.WriteRune(rune(0x4E00 + rng.Intn(0x2000)))
		}
		b.WriteString("，")
	})
}

// codeText: punctuation-dense, which is where the pre-tokenisation regex does
// the most backtracking.
func codeText(rng *rand.Rand, n int) string {
	lines := []string{
		"func (s *Server) handle(w http.ResponseWriter, r *http.Request) error {",
		"\tif err := s.db.QueryRowContext(ctx, q, id).Scan(&out); err != nil {",
		"\t\treturn fmt.Errorf(\"scan %d: %w\", id, err)",
		"\t}",
		"\tm := map[string][]int{\"a\": {1, 2, 3}, \"b\": {4, 5}}",
		"\treturn json.NewEncoder(w).Encode(&resp{OK: true, N: len(m)})",
		"}",
		"",
	}
	return fill(n, func(b *strings.Builder) {
		b.WriteString(lines[rng.Intn(len(lines))])
		b.WriteByte('\n')
	})
}

// longWordText: unbroken runs, so the merge loop sees long pieces, the
// quadratic term dominates here and nowhere else.
func longWordText(rng *rand.Rand, n int) string {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"
	return fill(n, func(b *strings.Builder) {
		for i := 0; i < 24+rng.Intn(40); i++ {
			b.WriteByte(alphabet[rng.Intn(len(alphabet))])
		}
		b.WriteByte(' ')
	})
}
