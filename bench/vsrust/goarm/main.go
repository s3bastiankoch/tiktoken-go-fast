// Command goarm is the Go side of the cross-language benchmark: a long-lived
// round server that loads a vocabulary once and then encodes on demand.
//
// It exists because the two implementations cannot be linked into one binary,
// and two sequential sweeps on a loaded machine are not comparable -- bench/
// documents a 2x disagreement between sweeps of identical code. The driver
// keeps both arms resident and alternates single rounds between them, so drift
// lasting longer than a round cancels inside the paired ratio.
//
// Protocol, one command per line on stdin, one reply per line on stdout:
//
//	load <encoding>   -> ok
//	open <path>       -> ok <bytes>
//	check             -> ok <ntokens> <fnv1a64 of the ids>
//	run <iters>       -> ok <elapsed_ns> <sink>
//	quit
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	loader "github.com/pkoukk/tiktoken-go-loader"
	tiktoken "github.com/s3bastiankoch/tiktoken-go-fast"
)

// digest is FNV-1a over the little-endian token ids. The Rust arm computes the
// same thing over its own ids and the driver refuses to time a workload until
// the two agree, so a speed number can never come from an arm that tokenises
// differently.
func digest(ids []int) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for _, id := range ids {
		v := uint32(id)
		for shift := 0; shift < 32; shift += 8 {
			h ^= uint64(byte(v >> shift))
			h *= prime64
		}
	}
	return h
}

func main() {
	tiktoken.SetBpeLoader(loader.NewOfflineLoader())

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1<<16), 1<<20)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	var enc *tiktoken.Tiktoken
	var text string

	for in.Scan() {
		fields := strings.Fields(in.Text())
		if len(fields) == 0 {
			continue
		}
		// load, open and run each take one argument; check and quit take none.
		// Reply with an error rather than panicking, so a malformed line is a
		// visible protocol error in the driver instead of a dead arm and a
		// broken pipe.
		switch fields[0] {
		case "load", "open", "run":
			if len(fields) < 2 {
				fmt.Fprintf(out, "err %s needs an argument\n", fields[0])
				out.Flush()
				continue
			}
		}
		// check and run encode, so both need a vocabulary already loaded.
		if enc == nil && (fields[0] == "check" || fields[0] == "run") {
			fmt.Fprintf(out, "err %s before load\n", fields[0])
			out.Flush()
			continue
		}
		switch fields[0] {
		case "load":
			e, err := tiktoken.GetEncoding(fields[1])
			if err != nil {
				fmt.Fprintf(out, "err %v\n", err)
				break
			}
			enc = e
			fmt.Fprintln(out, "ok")
		case "open":
			b, err := os.ReadFile(fields[1])
			if err != nil {
				fmt.Fprintf(out, "err %v\n", err)
				break
			}
			text = string(b)
			fmt.Fprintf(out, "ok %d\n", len(b))
		case "check":
			ids := enc.EncodeOrdinary(text)
			fmt.Fprintf(out, "ok %d %016x\n", len(ids), digest(ids))
		case "run":
			n, err := strconv.Atoi(fields[1])
			if err != nil {
				fmt.Fprintf(out, "err %v\n", err)
				break
			}
			sink := 0
			start := time.Now()
			for i := 0; i < n; i++ {
				sink += len(enc.EncodeOrdinary(text))
			}
			elapsed := time.Since(start)
			fmt.Fprintf(out, "ok %d %d\n", elapsed.Nanoseconds(), sink)
		case "quit":
			out.Flush()
			return
		default:
			fmt.Fprintf(out, "err unknown command %q\n", fields[0])
		}
		out.Flush()
	}
}
