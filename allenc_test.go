package tiktoken

import (
	"math/rand"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

// The differential gate in equiv_test.go runs on cl100k_base only. That
// was adequate while the scanner implemented one encoding by hand and fell back
// to regexp2 for the rest: the fallback path *was* the reference. It is not
// adequate now that the scanner is compiled from the pattern string and every
// shipped encoding takes the fast path: p50k_base, r50k_base and o200k_base
// each exercise scanner branches nothing else covers.
//
// These run the same comparison against the frozen reference implementation on
// all four.

var (
	allEncOnce sync.Once
	allEncVal  []namedEncoding
	allEncErr  error
)

type namedEncoding struct {
	name string
	tk   *Tiktoken
}

// allEncodings loads the four shipped encodings from the embedded vocab. It
// deliberately bypasses GetEncoding's package-global cache:
// TestGetEncoding_ErrorResponseNotCached asserts that o200k_base is absent from
// it, and a test that quietly populates it would make that assertion pass or
// fail depending on run order.
func allEncodings(tb testing.TB) []namedEncoding {
	tb.Helper()
	allEncOnce.Do(func() {
		SetBpeLoader(tiktoken_loader.NewOfflineLoader())
		for _, name := range []string{MODEL_CL100K_BASE, MODEL_O200K_BASE, MODEL_P50K_BASE, MODEL_R50K_BASE} {
			enc, err := initEncoding(name)
			if err != nil {
				allEncErr = err
				return
			}
			bpe, err := NewCoreBPE(enc.MergeableRanks, enc.SpecialTokens, enc.PatStr)
			if err != nil {
				allEncErr = err
				return
			}
			specialTokensSet := map[string]any{}
			for k := range enc.SpecialTokens {
				specialTokensSet[k] = true
			}
			allEncVal = append(allEncVal, namedEncoding{name, NewTiktoken(bpe, enc, specialTokensSet)})
		}
	})
	if allEncErr != nil {
		tb.Fatalf("load encodings: %v", allEncErr)
	}
	return allEncVal
}

func TestEncodeOrdinaryMatchesReferenceAllEncodings(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	texts := []string{"", " ", "  ", "\n", "\r\n", "hello world", "你好世界",
		"🙂🎉", "👨‍👩‍👧‍👦", "ЗдравствуйМир", "don't can't I'll we've",
		"DON'T CAN'T", "MixedCaseWord", "ABCdef", "123456789", "1.5e-9",
		"  \t\n  x", "a\r\nb", "x\n\n\n\ny", " nbsp", "ｆｕｌｌｗｉｄｔｈ",
		corpusText(rng, 1<<14),
	}
	for i := 0; i < 300; i++ {
		texts = append(texts, randomText(rng, 200))
	}
	for _, e := range allEncodings(t) {
		for _, text := range texts {
			got, want := e.tk.bpe.encodeOrdinaryNative(text), e.tk.bpe.refEncodeOrdinary(text)
			if !equalInts(got, want) {
				t.Fatalf("%s: encodeOrdinaryNative(%q):\n got %v\nwant %v", e.name, trunc(text), got, want)
			}
		}
	}
}

func TestEncodeRoundTripsAllEncodings(t *testing.T) {
	rng := rand.New(rand.NewSource(103))
	for _, e := range allEncodings(t) {
		for i := 0; i < 200; i++ {
			text := randomText(rng, 300)
			if !utf8.ValidString(text) {
				continue
			}
			if got := e.tk.Decode(e.tk.Encode(text, nil, nil)); got != text {
				t.Fatalf("%s: round trip: %q -> %q", e.name, trunc(text), trunc(got))
			}
		}
	}
}

// FuzzEncodeOrdinaryAllEncodings is FuzzEncodeOrdinary widened: every input is
// checked against the reference under all four encodings, so a divergence in a
// scanner branch that only o200k_base or r50k_base reaches is a failure here
// rather than something the corpus never visits.
//
//	go test -run xxx -fuzz FuzzEncodeOrdinaryAllEncodings -fuzztime 5m
func FuzzEncodeOrdinaryAllEncodings(f *testing.F) {
	for _, s := range []string{"", " ", "hello world", "你好世界", "🙂", "a\n\tb",
		"supercalifragilisticexpialidocious", "0x1p-3 1e10", "don't", "DON'T",
		"A1b2", "\r\n\r\n", "  \t  ", "Ａ１", "ẞß", "ᾈᾯ", "０１２３"} {
		f.Add(s)
	}
	encs := allEncodings(f)
	f.Fuzz(func(t *testing.T, text string) {
		if !utf8.ValidString(text) {
			t.Skip("tiktoken is defined over valid UTF-8")
		}
		for _, e := range encs {
			got, want := e.tk.bpe.encodeOrdinaryNative(text), e.tk.bpe.refEncodeOrdinary(text)
			if !equalInts(got, want) {
				t.Fatalf("%s: encodeOrdinaryNative(%q):\n got %v\nwant %v", e.name, trunc(text), got, want)
			}
		}
	})
}

// TestEncodeOrdinaryInvalidUTF8 covers the one input class the fuzz target
// skips. tiktoken is defined over valid UTF-8, so there is no upstream-correct
// answer here. But the scanner works on bytes while the reference works on
// runes, and Go's []rune conversion replaces every invalid byte with U+FFFD.
// The two must still agree, or a malformed byte in the middle of a request
// changes the token count depending on which path ran.
func TestEncodeOrdinaryInvalidUTF8(t *testing.T) {
	texts := []string{
		"\xff", "a\xffb", "\x80\x80", "hello \xc3\x28 world",
		"\xe2\x28\xa1", "\xf0\x28\x8c\x28", "abc\xed\xa0\x80def",
		"\xc0\x80", "text\xff\xfe\xfdmore", "\xf4\x90\x80\x80",
		"日本\xffご", "\xff\n\t ", strings.Repeat("\xff", 64),
	}
	for _, e := range allEncodings(t) {
		for _, text := range texts {
			got, want := e.tk.bpe.encodeOrdinaryNative(text), e.tk.bpe.refEncodeOrdinary(text)
			if !equalInts(got, want) {
				t.Errorf("%s: encodeOrdinaryNative(%q):\n got %v\nwant %v", e.name, text, got, want)
			}
		}
	}
}

// TestEncodeSpecialSegments covers the path the ordinary-encode gate does not:
// encodeNative splits the input at special tokens and calls the scanner once per
// segment, so segment boundaries (especially whitespace pressed against them,
// where lastPieceTokenLen decides) are where a rewritten encodePieces would
// diverge. The invariant is exact: encoding around a special token must equal
// encoding each segment ordinarily with the special id spliced in.
func TestEncodeSpecialSegments(t *testing.T) {
	rng := rand.New(rand.NewSource(107))
	segments := []string{"", " ", "  ", "\n", "hello", "hello ", " world",
		"你好", "a\n", "\ttab", "end.", "123", "don't"}
	for _, e := range allEncodings(t) {
		special := ENDOFTEXT
		id, ok := e.tk.bpe.specialTokensEncoder[special]
		if !ok {
			t.Fatalf("%s: no %s in special tokens", e.name, special)
		}
		for i := 0; i < 200; i++ {
			a := segments[rng.Intn(len(segments))]
			b := segments[rng.Intn(len(segments))]
			text := a + special + b
			got := e.tk.Encode(text, []string{"all"}, nil)
			want := append([]int{}, e.tk.EncodeOrdinary(a)...)
			want = append(want, id)
			want = append(want, e.tk.EncodeOrdinary(b)...)
			if !equalInts(got, want) {
				t.Fatalf("%s: Encode(%q, all):\n got %v\nwant %v", e.name, text, got, want)
			}
			if round := e.tk.Decode(got); round != text {
				t.Fatalf("%s: round trip %q -> %q", e.name, text, round)
			}
		}
	}
}

// TestCustomPatternFallsBack guards the seam that makes the scanner safe:
// splitKindFor matches the pattern string exactly, so anything else (a custom
// encoding built through NewCoreBPE, or a future upstream pattern) must take
// the regexp2 path rather than being scanned by rules that do not describe it.
func TestCustomPatternFallsBack(t *testing.T) {
	base := allEncodings(t)[0] // cl100k_base ranks, reused with other patterns
	patterns := []string{
		`\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+`,
		`.`,
		`\S+|\s+`,
		patCl100k + `|x`, // a near-miss: one alternative longer than the real thing
	}
	rng := rand.New(rand.NewSource(109))
	texts := []string{"hello world", "don't stop", "你好 world 123", "  \n\tx"}
	for i := 0; i < 50; i++ {
		texts = append(texts, randomText(rng, 120))
	}
	for _, pattern := range patterns {
		bpe, err := NewCoreBPE(base.tk.bpe.encoder, base.tk.bpe.specialTokensEncoder, pattern)
		if err != nil {
			t.Fatalf("NewCoreBPE(%q): %v", pattern, err)
		}
		if bpe.splitKind != splitRegexp2 {
			t.Errorf("pattern %q: splitKind = %d, want splitRegexp2", pattern, bpe.splitKind)
		}
		for _, text := range texts {
			got, want := bpe.encodeOrdinaryNative(text), bpe.refEncodeOrdinary(text)
			if !equalInts(got, want) {
				t.Fatalf("pattern %q on %q:\n got %v\nwant %v", pattern, trunc(text), got, want)
			}
		}
	}
}
