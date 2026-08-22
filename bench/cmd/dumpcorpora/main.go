// Command dumpcorpora writes every benchmark workload to a directory as a
// plain UTF-8 file, so an out-of-process arm -- a Rust binary, say -- is fed
// the exact bytes the in-process Go arms see rather than a reimplementation of
// the generators that happens to look similar.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	bench "github.com/s3bastiankoch/tiktoken-go-fast/bench"
)

func main() {
	out := flag.String("out", "", "directory to write corpora into")
	long := flag.Bool("long", true, "include the 256 KiB - 4 MiB workloads")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "usage: dumpcorpora -out DIR")
		os.Exit(2)
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		panic(err)
	}

	all := append(bench.Workloads(), bench.RealWorkloads()...)
	if *long {
		all = append(all, bench.LongWorkloads()...)
	}
	for _, w := range all {
		path := filepath.Join(*out, w.Name+".txt")
		if err := os.WriteFile(path, []byte(w.Text), 0o644); err != nil {
			panic(err)
		}
		fmt.Printf("%-20s %9d bytes\n", w.Name, len(w.Text))
	}
}
