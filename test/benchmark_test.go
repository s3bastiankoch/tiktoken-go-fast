package main

import (
	"os"
	"strings"
	"testing"

	"github.com/s3bastiankoch/tiktoken-go-fast"
)

// udhrPath is the full-language corpus written by get_udhr.py. Override it with
// UDHR_PATH; the benchmark skips rather than fails when the file is absent, so
// `go test ./...` does not depend on a file outside the repository.
func udhrPath() string {
	if p := os.Getenv("UDHR_PATH"); p != "" {
		return p
	}
	return "/tmp/udhr.txt"
}

// BenchmarkEncodingInFullLanguage measures EncodeOrdinary across every language
// in the UDHR corpus, one line per iteration.
//
//	python3 test/get_udhr.py
//	go test -benchmem -run '^$' -bench BenchmarkEncodingInFullLanguage ./test
func BenchmarkEncodingInFullLanguage(b *testing.B) {
	path := udhrPath()
	data, err := os.ReadFile(path)
	if err != nil {
		b.Skipf("corpus not available at %s (run test/get_udhr.py, or set UDHR_PATH): %v", path, err)
	}

	lines := strings.Split(string(data), "\n")
	if len(lines) == 0 {
		b.Skipf("corpus at %s is empty", path)
	}

	tkm, err := tiktoken.EncodingForModel("gpt-4o")
	if err != nil {
		b.Fatalf("EncodingForModel(gpt-4o): %v", err)
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		tkm.EncodeOrdinary(lines[n%len(lines)])
	}
}
