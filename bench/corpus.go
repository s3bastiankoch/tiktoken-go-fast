package bench

import (
	_ "embed"
	"math/rand"
	"strings"
)

// Real text, vendored under testdata/. See testdata/README.md for provenance.
var (
	//go:embed testdata/udhr_zho.txt
	udhrZho string
	//go:embed testdata/udhr_jpn.txt
	udhrJpn string
	//go:embed testdata/udhr_kor.txt
	udhrKor string
	//go:embed testdata/hongloumeng.txt
	hongloumeng string
	//go:embed testdata/mobydick.txt
	mobyDick string
	//go:embed testdata/gosrc.txt
	goSrc string
)

// RealWorkloads are natural-language samples at their natural size. The
// synthetic cjk_* workloads sample ideographs uniformly, which lands on rare
// characters that miss the vocabulary; these are what the ratio looks like on
// text somebody actually wrote.
func RealWorkloads() []Workload {
	return []Workload{
		{"udhr_zho", udhrZho},
		{"udhr_jpn", udhrJpn},
		{"udhr_kor", udhrKor},
		{"hongloumeng_256k", hongloumeng},
		{"mobydick_256k", mobyDick},
		{"gosrc_256k", goSrc},
	}
}

// Workloads are the ones bench_test.go uses in the root module,
// regenerated here with the same seeds and the same fragments so that a number
// measured across libraries can be lined up against a number measured within
// one. The only change is that fill does not slice the result back to exactly n
// bytes: that cut can land inside a multi-byte rune, and two libraries are free
// to disagree about invalid UTF-8 without either being wrong.
type Workload struct {
	Name string
	Text string
}

func Workloads() []Workload {
	return []Workload{
		{"prose_4k", corpusText(rand.New(rand.NewSource(7919)), 4<<10)},
		{"prose_64k", corpusText(rand.New(rand.NewSource(7919)), 64<<10)},
		{"ascii_4k", asciiText(rand.New(rand.NewSource(11)), 4<<10)},
		{"cjk_4k", cjkText(rand.New(rand.NewSource(13)), 4<<10)},
		{"code_4k", codeText(rand.New(rand.NewSource(17)), 4<<10)},
		{"longwords_4k", longWordText(rand.New(rand.NewSource(19)), 4<<10)},
	}
}

func fill(n int, next func(*strings.Builder)) string {
	var b strings.Builder
	for b.Len() < n {
		next(&b)
	}
	return b.String()
}

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
	return fill(n, func(b *strings.Builder) {
		b.WriteString(fragments[rng.Intn(len(fragments))])
	})
}

func asciiText(rng *rand.Rand, n int) string {
	words := strings.Fields("the of and to in a is that for it as was with be by on not he i this " +
		"but had are they from or an she which you one we all were her would there their")
	return fill(n, func(b *strings.Builder) {
		b.WriteString(words[rng.Intn(len(words))])
		b.WriteByte(' ')
	})
}

func cjkText(rng *rand.Rand, n int) string {
	return fill(n, func(b *strings.Builder) {
		for i := 0; i < 8; i++ {
			b.WriteRune(rune(0x4E00 + rng.Intn(0x2000)))
		}
		b.WriteString("，")
	})
}

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

func longWordText(rng *rand.Rand, n int) string {
	const alphabet = "abcdefghijklmnopqrstuvwxyz"
	return fill(n, func(b *strings.Builder) {
		for i := 0; i < 24+rng.Intn(40); i++ {
			b.WriteByte(alphabet[rng.Intn(len(alphabet))])
		}
		b.WriteByte(' ')
	})
}

// LongWorkloads are the sizes where anything super-linear shows up: a per-call
// rune conversion, an allocation that grows with the input rather than with the
// piece, or GC pressure that only bites once the working set leaves cache. The
// two scripts that tied at 4 KiB are carried up as well, since a tie can move in
// either direction with size.
func LongWorkloads() []Workload {
	return []Workload{
		{"prose_256k", corpusText(rand.New(rand.NewSource(7919)), 256<<10)},
		{"prose_1m", corpusText(rand.New(rand.NewSource(7919)), 1<<20)},
		{"prose_4m", corpusText(rand.New(rand.NewSource(7919)), 4<<20)},
		{"cjk_1m", cjkText(rand.New(rand.NewSource(13)), 1<<20)},
		{"code_1m", codeText(rand.New(rand.NewSource(17)), 1<<20)},
	}
}
