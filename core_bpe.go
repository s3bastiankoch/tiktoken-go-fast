package tiktoken

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2"
)

type CoreBPE struct {
	encoder              map[string]int
	decoder              map[int]string
	specialTokensEncoder map[string]int
	specialTokensDecoder map[int]string
	tlRegex              *regexp2.Regexp
	tlSpecialRegex       *regexp2.Regexp
	sortedTokenBytes     [][]byte
	splitKind            splitKind
}

func NewCoreBPE(encoder map[string]int, specialTokensEncoder map[string]int, pattern string) (*CoreBPE, error) {
	regex, err := regexp2.Compile(pattern, regexp2.None)
	if err != nil {
		return nil, fmt.Errorf("error compiling regex: %s", err)
	}

	specialRegexStrs := make([]string, 0, len(specialTokensEncoder))
	for k := range specialTokensEncoder {
		specialRegexStrs = append(specialRegexStrs, regexp.QuoteMeta(k))
	}
	specialRegex, err := regexp2.Compile(strings.Join(specialRegexStrs, "|"), regexp2.None)
	if err != nil {
		return nil, fmt.Errorf("error compiling special regex: %s", err)
	}

	decoder := make(map[int]string, len(encoder))
	for k, v := range encoder {
		decoder[v] = k
	}

	if len(encoder) != len(decoder) {
		return nil, errors.New("encoder and decoder map sizes are different")
	}

	specialTokensDecoder := make(map[int]string, len(specialTokensEncoder))
	for k, v := range specialTokensEncoder {
		specialTokensDecoder[v] = k
	}

	sortedTokenBytes := make([][]byte, 0, len(encoder))
	for k := range encoder {
		sortedTokenBytes = append(sortedTokenBytes, []byte(k))
	}
	sort.Slice(sortedTokenBytes, func(i, j int) bool {
		return bytes.Compare(sortedTokenBytes[i], sortedTokenBytes[j]) < 0
	})

	return &CoreBPE{
		encoder:              encoder,
		specialTokensEncoder: specialTokensEncoder,
		decoder:              decoder,
		specialTokensDecoder: specialTokensDecoder,
		tlRegex:              regex,
		tlSpecialRegex:       specialRegex,
		sortedTokenBytes:     sortedTokenBytes,
		splitKind:            splitKindFor(pattern),
	}, nil
}

func (bp *CoreBPE) encodeNative(text string, allowedSpecial map[string]any) ([]int, int) {
	specialRegex := bp.tlSpecialRegex
	ret := make([]int, 0, len(text)/4+16)
	lastPieceTokenLen := 0
	textRunes := []rune(text)

	start := 0
	for {
		var nextSpecial []int
		startFind := start
		for {
			// Find the next allowed special token, if any
			temp := cutRunes(textRunes, startFind, len(textRunes))
			nextSpecial = findRegex2StringIndex(temp, specialRegex)
			if nextSpecial != nil {
				token := cutRunes(textRunes, startFind+nextSpecial[0], startFind+nextSpecial[1])
				if _, ok := allowedSpecial[token]; ok {
					break
				}
				startFind += nextSpecial[1]
			} else {
				break
			}
		}

		end := len(textRunes)
		if nextSpecial != nil {
			end = start + nextSpecial[0]
		}

		// Okay, here we go, compare this logic to _encode_ordinary_native
		ret, lastPieceTokenLen = bp.encodePieces(ret, cutRunes(textRunes, start, end), lastPieceTokenLen)

		if nextSpecial != nil {
			temp := cutRunes(textRunes, start+nextSpecial[0], start+nextSpecial[1])
			token := bp.specialTokensEncoder[temp]
			ret = append(ret, token)
			start = start + nextSpecial[1]
			lastPieceTokenLen = 0
		} else {
			break
		}
	}

	return ret, lastPieceTokenLen
}

func (bp *CoreBPE) encodeOrdinaryNative(text string) []int {
	ret, _ := bp.encodePieces(make([]int, 0, len(text)/4+16), text, 0)
	return ret
}

// encodePieces appends the tokens of every piece the pre-tokenisation pattern
// carves out of s, and returns the token count of the last piece, or
// lastPieceTokenLen untouched when s yields no piece at all.
func (bp *CoreBPE) encodePieces(ret []int, s string, lastPieceTokenLen int) ([]int, int) {
	if bp.splitKind != splitRegexp2 {
		if !utf8.ValidString(s) {
			// The regex engine matches over []rune(s) and cutRunes cuts the same
			// slice, so an invalid byte is already a U+FFFD by the time a piece
			// exists. Normalise first and byte offsets line up with rune offsets.
			s = string([]rune(s))
		}
		if out, last, ok := bp.encodePiecesScan(ret, s, lastPieceTokenLen); ok {
			return out, last
		}
	}

	runes := []rune(s)
	for _, mat := range findRegex2AllStringMatchIndex(s, bp.tlRegex) {
		piece := cutRunes(runes, mat[0], mat[1])
		if token, ok := bp.encoder[piece]; ok {
			lastPieceTokenLen = 1
			ret = append(ret, token)
			continue
		}
		tokens := bytePairEncode([]byte(piece), bp.encoder)
		lastPieceTokenLen = len(tokens)
		ret = append(ret, tokens...)
	}
	return ret, lastPieceTokenLen
}

// encodePiecesScan walks s with the hand-written scanner for bp's pattern. It
// reports false without having grown ret if the scanner ever fails to advance,
// which leaves the caller free to fall back to the regex engine.
func (bp *CoreBPE) encodePiecesScan(ret []int, s string, lastPieceTokenLen int) ([]int, int, bool) {
	kind := bp.splitKind
	out := ret
	for i := 0; i < len(s); {
		var end int
		switch kind {
		case splitCl100k:
			end = scanCl100k(s, i)
		case splitO200k:
			end = scanO200k(s, i)
		default:
			end = scanP50k(s, i)
		}
		if end <= i {
			return ret, lastPieceTokenLen, false
		}
		piece := s[i:end]
		if token, ok := bp.encoder[piece]; ok {
			lastPieceTokenLen = 1
			out = append(out, token)
		} else {
			tokens := bytePairEncode([]byte(piece), bp.encoder)
			lastPieceTokenLen = len(tokens)
			out = append(out, tokens...)
		}
		i = end
	}
	return out, lastPieceTokenLen, true
}

func (bpe *CoreBPE) decodeNative(tokens []int) []byte {
	ret := make([]byte, 0, len(tokens)*2)
	for _, token := range tokens {
		tokenBytes, ok := bpe.decoder[token]
		if !ok {
			tokenBytes = bpe.specialTokensDecoder[token]
		}
		if len(tokenBytes) > 0 {
			ret = append(ret, tokenBytes...)
		}
	}
	return ret
}

func findRegex2StringIndex(text string, reg *regexp2.Regexp) []int {
	m, _ := reg.FindStringMatch(text)
	if m == nil {
		return nil
	}
	result := make([]int, 2)
	result[0] = m.Index
	result[1] = m.Index + m.Length
	return result
}

func findRegex2AllStringMatchIndex(text string, reg *regexp2.Regexp) [][]int {
	var matches [][]int
	m, _ := reg.FindStringMatch(text)
	for m != nil {
		result := make([]int, 2)
		result[0] = m.Index
		result[1] = m.Index + m.Length
		matches = append(matches, result)
		m, _ = reg.FindNextMatch(m)
	}
	return matches
}

func cutRunes(runes []rune, start, end int) string {
	if start < 0 {
		start = 0
	}
	if end > len(runes) {
		end = len(runes)
	}
	return string(runes[start:end])
}

// --- pre-tokenisation scanners -----------------------------------------------
//
// The five encodings tiktoken ships (four vocabularies: p50k_edit reuses
// p50k_base's) use three distinct patterns, and the regex engine walking them is
// the bulk of encode's cost: it converts the input to
// []rune, allocates a Match per piece, and backtracks a byte at a time. Each of
// the three is instead scanned directly below, over the original string, in one
// forward pass with no allocation per piece. An unrecognised pattern keeps the
// engine, so a custom NewCoreBPE still behaves exactly as it did.
//
// The scanners reproduce regexp2's semantics, not RE2's: leftmost match, the
// first alternative that can match wins, quantifiers are greedy and give ground
// only on failure. Every alternative of all three patterns consumes at least one
// rune and every rune starts some alternative, so the matches partition the
// input and a scanner never has to search forward for the next start.
//
// Character classes follow regexp2, which resolves \p{...} through Go's own
// unicode tables and \s through unicode.IsSpace, so unicode.IsLetter and friends
// are the exact predicates and not merely close ones. Likewise (?i:) folds with
// unicode.ToLower, whose only preimages for s, t, r, e, v, m, l and d are their
// ASCII uppercase twins, so no rune outside ASCII can start a contraction.

type splitKind uint8

const (
	splitRegexp2 splitKind = iota
	splitCl100k
	splitO200k
	splitP50k
)

// The three patterns, verbatim from encoding.go. o200k_base is assembled the
// same way its encoding is, so the two stay easy to compare by eye.
const (
	patCl100k = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

	patO200k = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?` +
		`|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?` +
		`|\p{N}{1,3}` +
		`| ?[^\s\p{L}\p{N}]+[\r\n/]*` +
		`|\s*[\r\n]+` +
		`|\s+(?!\S)` +
		`|\s+`

	patP50k = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
)

func splitKindFor(pattern string) splitKind {
	switch pattern {
	case patCl100k:
		return splitCl100k
	case patO200k:
		return splitO200k
	case patP50k:
		return splitP50k
	}
	return splitRegexp2
}

// scanCl100k returns the end of the piece cl100k_base's pattern matches at i.
func scanCl100k(s string, i int) int {
	r, size := decodeRuneAt(s, i)

	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if r == '\'' {
		if n := contractionLen(s, i+1, true); n > 0 {
			return i + 1 + n
		}
	}

	// [^\r\n\p{L}\p{N}]?\p{L}+
	if isLetterRune(r) {
		return scanLetters(s, i+size)
	}
	if isPrefixRune(r) && i+size < len(s) {
		if r2, n2 := decodeRuneAt(s, i+size); isLetterRune(r2) {
			return scanLetters(s, i+size+n2)
		}
	}

	// \p{N}{1,3}
	if isNumberRune(r) {
		return scanNumbersUpTo(s, i+size, 2)
	}

	// ` ?[^\s\p{L}\p{N}]+[\r\n]*`
	if end := scanPunct(s, i, r, false); end >= 0 {
		return end
	}

	// \s*[\r\n]+ | \s+(?!\S) | \s+
	return scanSpaceAlts(s, i, true)
}

// scanO200k returns the end of the piece o200k_base's pattern matches at i.
func scanO200k(s string, i int) int {
	r, size := decodeRuneAt(s, i)

	// [^\r\n\p{L}\p{N}]? in front of the two letter alternatives is greedy, so
	// the prefixed form of each is tried before the bare one.
	prefixed := isPrefixRune(r)

	// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:...)?
	if prefixed {
		if end := o200kLowerTail(s, i+size); end >= 0 {
			return end
		}
	}
	if end := o200kLowerTail(s, i); end >= 0 {
		return end
	}

	// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:...)?
	if prefixed {
		if end := o200kUpperHead(s, i+size); end >= 0 {
			return end
		}
	}
	if end := o200kUpperHead(s, i); end >= 0 {
		return end
	}

	// \p{N}{1,3}
	if isNumberRune(r) {
		return scanNumbersUpTo(s, i+size, 2)
	}

	// ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
	if end := scanPunct(s, i, r, true); end >= 0 {
		return end
	}

	// \s*[\r\n]+ | \s+(?!\S) | \s+
	return scanSpaceAlts(s, i, true)
}

// scanP50k returns the end of the piece the p50k/r50k pattern matches at i.
func scanP50k(s string, i int) int {
	r, size := decodeRuneAt(s, i)

	// 's|'t|'re|'ve|'m|'ll|'d. Case sensitive here, unlike the newer patterns.
	if r == '\'' {
		if n := contractionLen(s, i+1, false); n > 0 {
			return i + 1 + n
		}
	}

	// ` ?\p{L}+`, then ` ?\p{N}+`, then ` ?[^\s\p{L}\p{N}]+`. The optional lead is
	// a literal space, so only ' ' can carry a class across from the next rune.
	if isLetterRune(r) {
		return scanLetters(s, i+size)
	}
	if r == ' ' && i+1 < len(s) {
		r2, n2 := decodeRuneAt(s, i+1)
		switch {
		case isLetterRune(r2):
			return scanLetters(s, i+1+n2)
		case isNumberRune(r2):
			return scanNumbers(s, i+1+n2)
		case isPunctRune(r2):
			return scanPunctRun(s, i+1+n2)
		}
	}
	if isNumberRune(r) {
		return scanNumbers(s, i+size)
	}
	if isPunctRune(r) {
		return scanPunctRun(s, i+size)
	}

	// \s+(?!\S) | \s+. This pattern has no \s*[\r\n]+ alternative.
	return scanSpaceAlts(s, i, false)
}

// o200kLowerTail matches [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
// plus the optional contraction at p, or returns -1.
//
// The two classes overlap on Lm, Lo and M, so the greedy first loop can have to
// give ground: the engine settles on the last position at or before the end of
// the upper-ish run that the lower-ish class also accepts.
func o200kLowerTail(s string, p int) int {
	aEnd, lastLower := scanUpperish(s, p)
	q := -1
	if aEnd < len(s) {
		if r, _ := decodeRuneAt(s, aEnd); isLowerishRune(r) {
			q = aEnd
		}
	}
	if q < 0 {
		q = lastLower
	}
	if q < 0 {
		return -1
	}
	return optContraction(s, scanLowerish(s, q))
}

// o200kUpperHead matches [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
// plus the optional contraction at p, or returns -1. Nothing after the first
// loop can fail, so neither loop ever gives ground.
func o200kUpperHead(s string, p int) int {
	aEnd, _ := scanUpperish(s, p)
	if aEnd == p {
		return -1
	}
	return optContraction(s, scanLowerish(s, aEnd))
}

// scanPunct matches ` ?[^\s\p{L}\p{N}]+[\r\n]*` at i, or the o200k variant whose
// trailing class also holds '/'. It returns -1 when neither can start here: a
// leading ' ' is only kept if the rune after it is in the class, and ' ' is
// itself whitespace, so giving the space back never rescues the match.
func scanPunct(s string, i int, r rune, slash bool) int {
	p := i
	if r == ' ' {
		if i+1 >= len(s) {
			return -1
		}
		if r2, _ := decodeRuneAt(s, i+1); !isPunctRune(r2) {
			return -1
		}
		p = i + 1
	} else if !isPunctRune(r) {
		return -1
	}

	end := scanPunctRun(s, p)
	if slash {
		for end < len(s) && (s[end] == '\n' || s[end] == '\r' || s[end] == '/') {
			end++
		}
		return end
	}
	for end < len(s) && (s[end] == '\n' || s[end] == '\r') {
		end++
	}
	return end
}

// scanSpaceAlts resolves the whitespace tail every pattern ends with: first
// `\s*[\r\n]+` when the pattern has it, then `\s+(?!\S)`, then `\s+`.
//
// `\s*[\r\n]+` ends at the last \r or \n in the run, since the greedy \s* backs
// up to the last position where [\r\n]+ can still take a rune. `\s+(?!\S)` takes
// the whole run at end of input, and otherwise hands the final rune back so the
// lookahead sees whitespace, which needs a run of at least two.
func scanSpaceAlts(s string, i int, newlines bool) int {
	end, lastStart, lastNL := scanSpaceRun(s, i)
	if newlines && lastNL >= 0 {
		return lastNL + 1
	}
	if end == len(s) || lastStart <= i {
		return end
	}
	return lastStart
}

// optContraction matches (?i:'s|'t|'re|'ve|'m|'ll|'d)? at pos.
func optContraction(s string, pos int) int {
	if pos < len(s) && s[pos] == '\'' {
		if n := contractionLen(s, pos+1, true); n > 0 {
			return pos + 1 + n
		}
	}
	return pos
}

// contractionLen returns the byte length of the contraction body at j, or 0.
func contractionLen(s string, j int, fold bool) int {
	if j >= len(s) {
		return 0
	}
	c := s[j]
	if fold {
		c = lowerASCII(c)
	}
	switch c {
	case 's', 't', 'm', 'd':
		return 1
	case 'r', 'v':
		if j+1 < len(s) {
			e := s[j+1]
			if fold {
				e = lowerASCII(e)
			}
			if e == 'e' {
				return 2
			}
		}
	case 'l':
		if j+1 < len(s) {
			e := s[j+1]
			if fold {
				e = lowerASCII(e)
			}
			if e == 'l' {
				return 2
			}
		}
	}
	return 0
}

func lowerASCII(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + ('a' - 'A')
	}
	return c
}

// decodeRuneAt reads the rune at i. Callers hand it valid UTF-8, so a one-byte
// utf8.RuneError can only be a genuine U+FFFD.
func decodeRuneAt(s string, i int) (rune, int) {
	if c := s[i]; c < utf8.RuneSelf {
		return rune(c), 1
	}
	return utf8.DecodeRuneInString(s[i:])
}

// --- character classes -------------------------------------------------------

func isLetterRune(r rune) bool {
	if r < utf8.RuneSelf {
		return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
	}
	return unicode.IsLetter(r)
}

func isNumberRune(r rune) bool {
	if r < utf8.RuneSelf {
		return r >= '0' && r <= '9'
	}
	return unicode.IsNumber(r)
}

// isPrefixRune reports [^\r\n\p{L}\p{N}], the optional lead of the word
// alternatives. Marks are in it as well as in the letter classes, which is why
// the o200k scanner has to try both sides of the optional.
func isPrefixRune(r rune) bool {
	if r < utf8.RuneSelf {
		if r == '\r' || r == '\n' {
			return false
		}
		return !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'))
	}
	return !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

// isPunctRune reports [^\s\p{L}\p{N}].
func isPunctRune(r rune) bool {
	if r < utf8.RuneSelf {
		if r == ' ' || (r >= 0x09 && r <= 0x0d) {
			return false
		}
		return !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'))
	}
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

// isUpperishRune reports [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}].
func isUpperishRune(r rune) bool {
	if r < utf8.RuneSelf {
		return r >= 'A' && r <= 'Z'
	}
	return unicode.Is(unicode.Lu, r) || unicode.Is(unicode.Lt, r) ||
		unicode.Is(unicode.Lm, r) || unicode.Is(unicode.Lo, r) || unicode.Is(unicode.M, r)
}

// isLowerishRune reports [\p{Ll}\p{Lm}\p{Lo}\p{M}].
func isLowerishRune(r rune) bool {
	if r < utf8.RuneSelf {
		return r >= 'a' && r <= 'z'
	}
	return unicode.Is(unicode.Ll, r) || unicode.Is(unicode.Lm, r) ||
		unicode.Is(unicode.Lo, r) || unicode.Is(unicode.M, r)
}

// --- class runs --------------------------------------------------------------

func scanLetters(s string, i int) int {
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
				return i
			}
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(s[i:])
		if !unicode.IsLetter(r) {
			return i
		}
		i += n
	}
	return i
}

func scanNumbers(s string, i int) int {
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if c < '0' || c > '9' {
				return i
			}
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(s[i:])
		if !unicode.IsNumber(r) {
			return i
		}
		i += n
	}
	return i
}

// scanNumbersUpTo takes at most more further \p{N} runes, for \p{N}{1,3}.
func scanNumbersUpTo(s string, i, more int) int {
	for ; more > 0 && i < len(s); more-- {
		c := s[i]
		if c < utf8.RuneSelf {
			if c < '0' || c > '9' {
				return i
			}
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(s[i:])
		if !unicode.IsNumber(r) {
			return i
		}
		i += n
	}
	return i
}

func scanPunctRun(s string, i int) int {
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if c == ' ' || (c >= 0x09 && c <= 0x0d) ||
				(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
				return i
			}
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(s[i:])
		if !isPunctRune(r) {
			return i
		}
		i += n
	}
	return i
}

// scanUpperish walks [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]* from i and returns the end
// of the run together with the offset of its last rune that the lower-ish class
// also accepts, or -1. That offset is where the greedy loop lands when the
// following [\p{Ll}\p{Lm}\p{Lo}\p{M}]+ forces it to give ground. No ASCII rune is
// in both classes, so the fast path never has to track it.
func scanUpperish(s string, i int) (end, lastLower int) {
	lastLower = -1
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if c < 'A' || c > 'Z' {
				return i, lastLower
			}
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(s[i:])
		if !isUpperishRune(r) {
			return i, lastLower
		}
		if isLowerishRune(r) {
			lastLower = i
		}
		i += n
	}
	return i, lastLower
}

func scanLowerish(s string, i int) int {
	for i < len(s) {
		c := s[i]
		if c < utf8.RuneSelf {
			if c < 'a' || c > 'z' {
				return i
			}
			i++
			continue
		}
		r, n := utf8.DecodeRuneInString(s[i:])
		if !isLowerishRune(r) {
			return i
		}
		i += n
	}
	return i
}

// scanSpaceRun walks \s* from i. It returns the end of the run, the start of the
// run's final rune, and the offset of the last \r or \n in it (-1 if there is
// none): the three facts the whitespace alternatives need. An empty run leaves
// lastStart at i, which scanSpaceAlts reads as "no rune to give back".
func scanSpaceRun(s string, i int) (end, lastStart, lastNL int) {
	end, lastStart, lastNL = i, i, -1
	for end < len(s) {
		c := s[end]
		if c < utf8.RuneSelf {
			if c != ' ' && (c < 0x09 || c > 0x0d) {
				return end, lastStart, lastNL
			}
			if c == '\n' || c == '\r' {
				lastNL = end
			}
			lastStart = end
			end++
			continue
		}
		r, n := utf8.DecodeRuneInString(s[end:])
		if !unicode.IsSpace(r) {
			return end, lastStart, lastNL
		}
		lastStart = end
		end += n
	}
	return end, lastStart, lastNL
}
