// Command driver measures this fork against openai/tiktoken's Rust core.
//
// The two implementations cannot be linked into one binary, so the comparison
// is made the same way bench/ makes its in-process ones: paired and
// interleaved. Both arms are long-lived child processes with their vocabulary
// already loaded; each round asks one arm for a burst, then the other, and the
// ratio is taken inside the round. Drift lasting longer than a round -- which
// is what a loaded laptop mostly produces -- cancels. The reported figure is
// the median of the per-round ratios with the interquartile range beside it.
//
// The order of the two arms alternates round by round, so neither arm
// systematically runs first into a cold cache or a just-freed core.
//
//	go run ./driver -corpora DIR -go-bin PATH -rust-bin PATH
//
// See README.md for the full build-and-run sequence.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type arm struct {
	name string
	cmd  *exec.Cmd
	in   io.WriteCloser
	out  *bufio.Reader
}

func start(name, bin string) (*arm, error) {
	cmd := exec.Command(bin)
	cmd.Stderr = os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &arm{name: name, cmd: cmd, in: stdin, out: bufio.NewReaderSize(stdout, 1<<16)}, nil
}

func (a *arm) ask(format string, args ...any) []string {
	if _, err := fmt.Fprintf(a.in, format+"\n", args...); err != nil {
		fatalf("%s: write: %v", a.name, err)
	}
	line, err := a.out.ReadString('\n')
	if err != nil {
		fatalf("%s: read: %v", a.name, err)
	}
	fields := strings.Fields(line)
	if len(fields) == 0 || fields[0] != "ok" {
		fatalf("%s: %s", a.name, strings.TrimSpace(line))
	}
	return fields[1:]
}

func (a *arm) quit() {
	fmt.Fprintln(a.in, "quit")
	a.in.Close()
	a.cmd.Wait()
}

// runBurst returns nanoseconds per encode over iters calls inside the arm.
func (a *arm) runBurst(iters int) float64 {
	reply := a.ask("run %d", iters)
	ns, err := strconv.ParseFloat(reply[0], 64)
	if err != nil {
		fatalf("%s: bad reply %q", a.name, reply[0])
	}
	return ns / float64(iters)
}

// calibrate picks an iteration count that makes one burst last about burstFor,
// so both arms are measured over a comparable slice of wall clock however far
// apart their per-call costs are.
func (a *arm) calibrate(burstFor time.Duration) int {
	iters := 1
	for {
		start := time.Now()
		a.ask("run %d", iters)
		if time.Since(start) >= burstFor || iters >= 1<<20 {
			return iters
		}
		iters *= 2
	}
}

func quantile(xs []float64, q float64) float64 {
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	return sorted[int(q*float64(len(sorted)-1))]
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func main() {
	corpora := flag.String("corpora", "", "directory of corpus .txt files (see cmd/dumpcorpora)")
	goBin := flag.String("go-bin", "", "path to the built goarm binary")
	rustBin := flag.String("rust-bin", "", "path to the built rustarm binary")
	encodings := flag.String("encodings", "cl100k_base,o200k_base", "comma-separated encodings")
	workloads := flag.String("workloads", "prose_4k,prose_64k,ascii_4k,cjk_4k,code_4k,longwords_4k,mobydick_256k,gosrc_256k,hongloumeng_256k,udhr_zho,udhr_jpn,udhr_kor", "comma-separated corpus names")
	rounds := flag.Int("rounds", 31, "interleaved rounds per cell")
	burst := flag.Duration("burst", 5*time.Millisecond, "target duration of one round's burst")
	flag.Parse()

	if *corpora == "" || *goBin == "" || *rustBin == "" {
		fatalf("usage: driver -corpora DIR -go-bin PATH -rust-bin PATH")
	}

	goArm, err := start("go", *goBin)
	if err != nil {
		fatalf("start go arm: %v", err)
	}
	defer goArm.quit()
	rustArm, err := start("rust", *rustBin)
	if err != nil {
		fatalf("start rust arm: %v", err)
	}
	defer rustArm.quit()

	// The ratio is rust/fork, so a figure above 1 means this fork is faster --
	// the same direction every other table in bench/ uses.
	fmt.Printf("| encoding | workload | KiB | rust ns/op | fork ns/op | vs rust | rust MB/s | fork MB/s |\n")
	fmt.Printf("|---|---|---|---|---|---|---|---|\n")

	for _, enc := range strings.Split(*encodings, ",") {
		goArm.ask("load %s", enc)
		rustArm.ask("load %s", enc)

		for _, name := range strings.Split(*workloads, ",") {
			path := filepath.Join(*corpora, name+".txt")
			if _, err := os.Stat(path); err != nil {
				fmt.Fprintf(os.Stderr, "skip %s: %v\n", name, err)
				continue
			}
			sizeReply := goArm.ask("open %s", path)
			rustArm.ask("open %s", path)
			nbytes, _ := strconv.Atoi(sizeReply[0])

			// Gate: identical token ids, or the timing means nothing.
			goCheck := goArm.ask("check")
			rustCheck := rustArm.ask("check")
			if goCheck[0] != rustCheck[0] || goCheck[1] != rustCheck[1] {
				fmt.Fprintf(os.Stderr, "MISMATCH %s/%s: fork %s tokens %s, rust %s tokens %s\n",
					enc, name, goCheck[0], goCheck[1], rustCheck[0], rustCheck[1])
				continue
			}

			goIters := goArm.calibrate(*burst)
			rustIters := rustArm.calibrate(*burst)

			ratios := make([]float64, 0, *rounds)
			goNs := make([]float64, 0, *rounds)
			rustNs := make([]float64, 0, *rounds)
			for r := 0; r < *rounds; r++ {
				var g, u float64
				if r%2 == 0 {
					g = goArm.runBurst(goIters)
					u = rustArm.runBurst(rustIters)
				} else {
					u = rustArm.runBurst(rustIters)
					g = goArm.runBurst(goIters)
				}
				goNs = append(goNs, g)
				rustNs = append(rustNs, u)
				ratios = append(ratios, u/g)
			}

			medGo, medRust := quantile(goNs, 0.5), quantile(rustNs, 0.5)
			med := quantile(ratios, 0.5)
			lo, hi := quantile(ratios, 0.25), quantile(ratios, 0.75)
			mb := float64(nbytes) / 1e6
			fmt.Printf("| %s | %s | %d | %.0f | %.0f | **%.2fx** [%.2f-%.2f] | %.0f | %.0f |\n",
				enc, name, nbytes/1024, medRust, medGo, med, lo, hi,
				mb/(medRust/1e9), mb/(medGo/1e9))
		}
	}
}
