package tiktoken

import (
	"testing"

	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

// The model tables are checked against upstream openai/tiktoken's
// tiktoken/model.py (main, fetched 2026-08-22). Prefix matching is what keeps
// new releases working without a library update, so the cases below are mostly
// names that do not appear in either table verbatim.
func TestEncodingNameForModel(t *testing.T) {
	cases := []struct {
		model string
		want  string
	}{
		// gpt-5 family: the prefix has no trailing dash, so point releases and
		// size variants all resolve without a table entry of their own.
		{"gpt-5", MODEL_O200K_BASE},
		{"gpt-5.5", MODEL_O200K_BASE},
		{"gpt-5.1-2025-11-13", MODEL_O200K_BASE},
		{"gpt-5-mini", MODEL_O200K_BASE},
		{"gpt-5-nano-2025-08-07", MODEL_O200K_BASE},
		// reasoning
		{"o1", MODEL_O200K_BASE},
		{"o1-preview", MODEL_O200K_BASE},
		{"o3-mini", MODEL_O200K_BASE},
		{"o4-mini", MODEL_O200K_BASE},
		{"o4-mini-2025-04-16", MODEL_O200K_BASE},
		// chat
		{"chatgpt-4o-latest", MODEL_O200K_BASE},
		{"gpt-4.1-2025-04-14", MODEL_O200K_BASE},
		{"gpt-4.5-preview", MODEL_O200K_BASE},
		{"gpt-4o-2024-05-13", MODEL_O200K_BASE},
		{"gpt-4-32k", MODEL_CL100K_BASE},
		{"gpt-3.5-turbo-0301", MODEL_CL100K_BASE},
		// Azure deployment names
		{"gpt-35-turbo", MODEL_CL100K_BASE},
		{"gpt-35-turbo-16k", MODEL_CL100K_BASE},
		// fine-tuned: "ft:gpt-4o" and "ft:gpt-4" both match the first name, and
		// only longest-prefix-first gives it o200k_base.
		{"ft:gpt-4o-2024-08-06:acme::A1b2C3d4", MODEL_O200K_BASE},
		{"ft:gpt-4-0613:acme::A1b2C3d4", MODEL_CL100K_BASE},
		{"ft:gpt-3.5-turbo:acme::A1b2C3d4", MODEL_CL100K_BASE},
		{"ft:davinci-002:acme::A1b2C3d4", MODEL_CL100K_BASE},
		// base and legacy
		{"davinci-002", MODEL_CL100K_BASE},
		{"babbage-002", MODEL_CL100K_BASE},
		{"text-embedding-3-large", MODEL_CL100K_BASE},
		{"text-davinci-003", MODEL_P50K_BASE},
		{"davinci", MODEL_R50K_BASE},
		{"gpt2", "gpt2"},
		{"gpt-2", "gpt2"},
	}
	for _, c := range cases {
		got, ok := encodingNameForModel(c.model)
		if !ok {
			t.Errorf("encodingNameForModel(%q): not found, want %s", c.model, c.want)
			continue
		}
		if got != c.want {
			t.Errorf("encodingNameForModel(%q) = %s, want %s", c.model, got, c.want)
		}
	}
}

// gpt-oss uses o200k_harmony, which this library does not ship. Reporting "no
// encoding" is the only honest answer; mapping it to o200k_base would return
// plausible token counts that are wrong.
func TestEncodingNameForModelUnknown(t *testing.T) {
	for _, model := range []string{"gpt-oss-20b", "gpt-oss-120b", "llama-3", ""} {
		if got, ok := encodingNameForModel(model); ok {
			t.Errorf("encodingNameForModel(%q) = %s, want not found", model, got)
		}
	}
}

// Go randomises map iteration, so a prefix table walked directly would resolve
// overlapping entries differently from run to run. The sorted slice is what
// makes this deterministic; 1000 lookups would catch a regression to map order.
func TestEncodingNameForModelDeterministic(t *testing.T) {
	const model = "ft:gpt-4o-2024-08-06:acme::A1b2C3d4"
	for i := 0; i < 1000; i++ {
		if got, _ := encodingNameForModel(model); got != MODEL_O200K_BASE {
			t.Fatalf("iteration %d: encodingNameForModel(%q) = %s, want %s", i, model, got, MODEL_O200K_BASE)
		}
	}
}

// End-to-end: a name no table lists verbatim reaches a loaded encoder and
// tokenises identically to the encoding it maps onto.
func TestEncodingForModelGPT5(t *testing.T) {
	// GetEncoding caches into the package-global encodingMap, and
	// TestGetEncoding_ErrorResponseNotCached asserts that o200k_base is *not*
	// cached. Leave the globals as they were found.
	prevLoader := bpeLoader
	t.Cleanup(func() {
		SetBpeLoader(prevLoader)
		l.Lock()
		delete(encodingMap, MODEL_O200K_BASE)
		l.Unlock()
	})
	SetBpeLoader(tiktoken_loader.NewOfflineLoader())

	enc, err := EncodingForModel("gpt-5.5")
	if err != nil {
		t.Fatalf("EncodingForModel(gpt-5.5): %v", err)
	}
	o200k, err := GetEncoding(MODEL_O200K_BASE)
	if err != nil {
		t.Fatalf("GetEncoding(o200k_base): %v", err)
	}
	const text = "Tokenisation is the first step of every language model pipeline. 你好世界"
	got := enc.EncodeOrdinary(text)
	want := o200k.EncodeOrdinary(text)
	if len(got) != len(want) {
		t.Fatalf("token count %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d = %d, want %d", i, got[i], want[i])
		}
	}
}
