package bench

import (
	"fmt"
	"sort"
	"testing"
	"time"
)

// go test -bench runs arms one after another, so on a loaded machine the two
// libraries are not measured under the same conditions: repeated sweeps here
// disagreed by 2x on individual cells while allocation counts stayed identical,
// which is drift, not a difference in the code.
//
// TestPairedRatio interleaves instead. Each round measures every arm once, back
// to back, and the ratio is taken within the round; drift that lasts longer than
// a round cancels. The reported figure is the median of the per-round ratios,
// with the interquartile range as the honest error bar.
//
//	go test -run TestPairedRatio -v .
const (
	pairRounds  = 31
	pairBurstNs = 5 * time.Millisecond
)

// burst runs fn repeatedly for at least pairBurstNs and returns ns per call.
func burst(fn func(string), s string, iters int) float64 {
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn(s)
	}
	return float64(time.Since(start).Nanoseconds()) / float64(iters)
}

// calibrate picks an iteration count that makes one burst last ~pairBurstNs, so
// every arm is measured over a comparable slice of wall clock.
func calibrate(fn func(string), s string) int {
	iters := 1
	for {
		start := time.Now()
		for i := 0; i < iters; i++ {
			fn(s)
		}
		if d := time.Since(start); d >= pairBurstNs || iters >= 1<<20 {
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

func TestPairedRatio(t *testing.T) {
	pairedTable(t, Workloads(), pairRounds)
}

// TestPairedRatioLong repeats the comparison at 256 KiB to 4 MiB. Fewer rounds,
// because one round already costs seconds at that size; the per-round ratio is
// also steadier there, since a single burst spans enough work to average over
// whatever else the machine is doing.
func TestPairedRatioLong(t *testing.T) {
	if testing.Short() {
		t.Skip("long workloads")
	}
	pairedTable(t, LongWorkloads(), 11)
}

// TestPairedRatioReal runs the comparison on real text rather than on generated
// text, which is the only way the CJK result means anything about production
// traffic.
func TestPairedRatioReal(t *testing.T) {
	pairedTable(t, RealWorkloads(), pairRounds)
}

// TestPairedRatioAllEncodings covers p50k_base and r50k_base against upstream.
// No current model uses them, so they are not in the routine tables, but the
// per-encoding claim in the top-level README covers all four.
func TestPairedRatioAllEncodings(t *testing.T) {
	pairedTableWith(t, gateEncodings(), []Workload{Workloads()[0], RealWorkloads()[4]}, pairRounds)
}

func pairedTable(t *testing.T, workloads []Workload, rounds int) {
	pairedTableWith(t, encodings(), workloads, rounds)
}

func pairedTableWith(t *testing.T, encs []namedPair, workloads []Workload, rounds int) {
	fmt.Printf("%-12s %-17s %12s %12s %12s %9s %9s %9s\n",
		"encoding", "workload", "fast ns", "upstream ns", "tokEnc ns", "vs upstr", "vs Enc", "vs Cnt")
	for _, e := range encs {
		for _, w := range workloads {
			arms := []func(string){
				func(s string) { sinkInts = e.fast.EncodeOrdinary(s) },
				func(s string) { sinkUints, sinkPieces, _ = e.tok.Encode(s) },
				func(s string) { sinkCount, _ = e.tok.Count(s) },
				func(s string) { sinkInts = e.up.EncodeOrdinary(s) },
			}
			iters := make([]int, len(arms))
			for i, fn := range arms {
				iters[i] = calibrate(fn, w.Text)
			}
			ns := make([][]float64, len(arms))
			ratioEnc := make([]float64, 0, rounds)
			ratioCnt := make([]float64, 0, rounds)
			ratioUp := make([]float64, 0, rounds)
			for r := 0; r < rounds; r++ {
				var round [4]float64
				for i, fn := range arms {
					round[i] = burst(fn, w.Text, iters[i])
					ns[i] = append(ns[i], round[i])
				}
				ratioEnc = append(ratioEnc, round[1]/round[0])
				ratioCnt = append(ratioCnt, round[2]/round[0])
				ratioUp = append(ratioUp, round[3]/round[0])
			}
			fmt.Printf("%-12s %-17s %12.0f %12.0f %12.0f %8.2fx %8.2fx %8.2fx  [upstr %.2f-%.2f]\n",
				e.name, w.Name,
				quantile(ns[0], 0.5), quantile(ns[3], 0.5), quantile(ns[1], 0.5),
				quantile(ratioUp, 0.5), quantile(ratioEnc, 0.5), quantile(ratioCnt, 0.5),
				quantile(ratioUp, 0.25), quantile(ratioUp, 0.75))
		}
	}
}
