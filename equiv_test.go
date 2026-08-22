package tiktoken

import (
	"math"
	"math/rand"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

// Differential gate for tiktoken variants.
//
// The oracle is the library itself at the pin: refBytePairMerge and
// refEncodeOrdinary below are verbatim copies of bpe.go and core_bpe.go as of
// 200cf60. A variant rewrites the live implementation; these copies do not
// move, so `live == ref` is a real equivalence claim rather than a tautology.
// Editing anything in the "frozen reference" block invalidates the gate.
//
// Vocab comes from the offline loader, so the gate needs no network and every
// run sees the same ranks.

// --- frozen reference: do not edit ------------------------------------------

func refBytePairMerge[T any](piece []byte, ranks map[string]int, f func(start, end int) T) []T {
	parts := make([][2]int, len(piece)+1)
	for i := 0; i < len(parts); i++ {
		parts[i][0], parts[i][1] = i, math.MaxInt
	}
	getRank := func(startIdx, skip int) int {
		if startIdx+skip+2 < len(parts) {
			b := piece[parts[startIdx][0]:parts[startIdx+skip+2][0]]
			if rank, ok := ranks[string(b)]; ok {
				return rank
			}
		}
		return -1
	}
	for i := 0; i < len(parts)-2; i++ {
		if rank := getRank(i, 0); rank >= 0 {
			parts[i][1] = rank
		}
	}
	for len(parts) > 1 {
		minRank, minIdx := math.MaxInt, -1
		for i := 0; i < len(parts)-1; i++ {
			if parts[i][1] < minRank {
				minRank, minIdx = parts[i][1], i
			}
		}
		if minRank < math.MaxInt {
			i := minIdx
			if rank := getRank(i, 1); rank >= 0 {
				parts[i][1] = rank
			} else {
				parts[i][1] = math.MaxInt
			}
			if i > 0 {
				if rk := getRank(i-1, 1); rk >= 0 {
					parts[i-1][1] = rk
				} else {
					parts[i-1][1] = math.MaxInt
				}
			}
			parts = append(parts[:i+1], parts[i+2:]...)
		} else {
			break
		}
	}
	out := make([]T, len(parts)-1)
	for i := 0; i < len(out); i++ {
		out[i] = f(parts[i][0], parts[i+1][0])
	}
	return out
}

func refBytePairEncode(piece []byte, ranks map[string]int) []int {
	if len(piece) == 1 {
		return []int{ranks[string(piece)]}
	}
	return refBytePairMerge(piece, ranks, func(start, end int) int {
		return ranks[string(piece[start:end])]
	})
}

func refCutRunes(runes []rune, start, end int) string {
	if start < 0 {
		start = 0
	}
	if end > len(runes) {
		end = len(runes)
	}
	return string(runes[start:end])
}

func (bp *CoreBPE) refEncodeOrdinary(text string) []int {
	ret := []int{}
	textRunes := []rune(text)
	for _, mat := range findRegex2AllStringMatchIndex(text, bp.tlRegex) {
		piece := refCutRunes(textRunes, mat[0], mat[1])
		if token, ok := bp.encoder[piece]; ok {
			ret = append(ret, token)
			continue
		}
		ret = append(ret, refBytePairEncode([]byte(piece), bp.encoder)...)
	}
	return ret
}

// --- vocab -------------------------------------------------------------------

var (
	encOnce sync.Once
	encVal  *Tiktoken
	encErr  error
)

// testEncoding returns cl100k_base from the embedded vocab: no network, same
// ranks every run.
func testEncoding(tb testing.TB) *Tiktoken {
	tb.Helper()
	encOnce.Do(func() {
		SetBpeLoader(tiktoken_loader.NewOfflineLoader())
		encVal, encErr = GetEncoding(MODEL_CL100K_BASE)
	})
	if encErr != nil {
		tb.Fatalf("load cl100k_base: %v", encErr)
	}
	return encVal
}

// --- inputs ------------------------------------------------------------------

// corpusText is deterministic and covers what a tokeniser actually meets: prose,
// CJK and other multi-byte scripts, source code, digits, whitespace runs, and
// long unbroken words that make byte-pair merging do work.
func corpusText(rng *rand.Rand, n int) string {
	fragments := []string{
		"the quick brown fox jumps over the lazy dog ",
		"Tokenisation is the first step of every language model pipeline. ",
		"你好世界，这是一个测试。", "こんにちは世界！これはテストです。", "안녕하세요 세계! 이것은 테스트입니다.",
		"Привет мир! Это тест. ", "¡Hola mundo! ¿Qué tal? ",
		"func main() { fmt.Println(\"hello\") }\n",
		"SELECT id, name FROM users WHERE created_at > '2024-01-01' ORDER BY id;\n",
		"{\"key\": \"value\", \"n\": 12345, \"nested\": {\"a\": [1, 2, 3]}}",
		"0123456789 3.14159265358979 1e-9 0xDEADBEEF ",
		"        \t\t   \n\n\n   ",
		"supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis ",
		"https://example.com/path?query=value&other=thing#fragment ",
		"e-mail: user.name+tag@example.co.uk ",
		"🙂🎉🚀 emoji sequences ❤️‍🔥 and ZWJ families 👨‍👩‍👧‍👦 ",
	}
	var b strings.Builder
	for b.Len() < n {
		b.WriteString(fragments[rng.Intn(len(fragments))])
	}
	return b.String()
}

// randomText is adversarial rather than realistic: arbitrary runes, including
// the boundaries where a rune-index and a byte-index implementation diverge.
func randomText(rng *rand.Rand, maxRunes int) string {
	var b strings.Builder
	for i := 0; i < rng.Intn(maxRunes+1); i++ {
		switch rng.Intn(10) {
		case 0, 1, 2, 3:
			b.WriteByte(byte(' ' + rng.Intn(95))) // ASCII
		case 4:
			b.WriteRune(rune(0x80 + rng.Intn(0x700))) // 2-byte
		case 5:
			b.WriteRune(rune(0x4E00 + rng.Intn(0x1000))) // CJK, 3-byte
		case 6:
			b.WriteRune(rune(0x1F300 + rng.Intn(0x300))) // emoji, 4-byte
		case 7:
			b.WriteString("\n\t ")
		case 8:
			b.WriteString(strings.Repeat("a", rng.Intn(40)))
		default:
			b.WriteRune(rune(rng.Intn(0x2000)))
		}
	}
	return b.String()
}

// --- gates -------------------------------------------------------------------

func TestBytePairEncodeMatchesReference(t *testing.T) {
	ranks := testEncoding(t).bpe.encoder
	rng := rand.New(rand.NewSource(1))
	pieces := []string{"", "a", "the", "  ", "hello", "tokenisation", "ЗдравствуйМир",
		"supercalifragilisticexpialidocious", "0123456789", "\n\n\t", "🙂🎉"}
	for i := 0; i < 2000; i++ {
		pieces = append(pieces, randomText(rng, 24))
	}
	for _, p := range pieces {
		if len(p) == 0 {
			continue // bytePairEncode is never called with an empty piece
		}
		got, want := bytePairEncode([]byte(p), ranks), refBytePairEncode([]byte(p), ranks)
		if !equalInts(got, want) {
			t.Fatalf("bytePairEncode(%q):\n got %v\nwant %v", p, got, want)
		}
	}
}

func TestEncodeOrdinaryMatchesReference(t *testing.T) {
	enc := testEncoding(t)
	rng := rand.New(rand.NewSource(2))
	texts := []string{"", " ", "hello world", "你好世界", corpusText(rng, 1<<14)}
	for i := 0; i < 300; i++ {
		texts = append(texts, randomText(rng, 200))
	}
	for _, text := range texts {
		got, want := enc.bpe.encodeOrdinaryNative(text), enc.bpe.refEncodeOrdinary(text)
		if !equalInts(got, want) {
			t.Fatalf("encodeOrdinaryNative(%q):\n got %v\nwant %v", trunc(text), got, want)
		}
	}
}

// TestEncodeRoundTrips is the end-to-end invariant: whatever the tokens are,
// decoding them must give the input back.
func TestEncodeRoundTrips(t *testing.T) {
	enc := testEncoding(t)
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 200; i++ {
		text := randomText(rng, 300)
		if !utf8.ValidString(text) {
			continue
		}
		if got := enc.Decode(enc.Encode(text, nil, nil)); got != text {
			t.Fatalf("round trip: %q -> %q", trunc(text), trunc(got))
		}
	}
}

func FuzzEncodeOrdinary(f *testing.F) {
	for _, s := range []string{"", " ", "hello world", "你好世界", "🙂", "a\n\tb",
		"supercalifragilisticexpialidocious", "0x1p-3 1e10"} {
		f.Add(s)
	}
	enc := testEncoding(f)
	f.Fuzz(func(t *testing.T, text string) {
		if !utf8.ValidString(text) {
			t.Skip("tiktoken is defined over valid UTF-8")
		}
		got, want := enc.bpe.encodeOrdinaryNative(text), enc.bpe.refEncodeOrdinary(text)
		if !equalInts(got, want) {
			t.Fatalf("encodeOrdinaryNative(%q):\n got %v\nwant %v", trunc(text), got, want)
		}
	})
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func trunc(s string) string {
	if len(s) <= 120 {
		return s
	}
	return s[:120] + "…"
}
