package tiktoken

import (
	"errors"
	"sort"
	"strings"
	"sync"
)

const (
	ENDOFTEXT         string = "<|endoftext|>"
	FIM_PREFIX        string = "<|fim_prefix|>"
	FIM_MIDDLE        string = "<|fim_middle|>"
	FIM_SUFFIX        string = "<|fim_suffix|>"
	ENDOFPROMPT       string = "<|endofprompt|>"
	MODEL_O200K_BASE  string = "o200k_base"
	MODEL_CL100K_BASE string = "cl100k_base"
	MODEL_P50K_BASE   string = "p50k_base"
	MODEL_P50K_EDIT   string = "p50k_edit"
	MODEL_R50K_BASE   string = "r50k_base"

	BpeBaseURL = "https://openaipublic.blob.core.windows.net"

	O200kBaseURL  = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
	Cl100kBaseURL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
	P50kBaseURL   = "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
	R50kBaseURL   = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
)

var MODEL_TO_ENCODING = map[string]string{
	// reasoning
	"o1":      MODEL_O200K_BASE,
	"o3":      MODEL_O200K_BASE,
	"o4-mini": MODEL_O200K_BASE,
	// chat
	"gpt-5":         MODEL_O200K_BASE,
	"gpt-4.5":       MODEL_O200K_BASE,
	"gpt-4.1":       MODEL_O200K_BASE,
	"gpt-4o":        MODEL_O200K_BASE,
	"gpt-4":         MODEL_CL100K_BASE,
	"gpt-3.5-turbo": MODEL_CL100K_BASE,
	"gpt-3.5":       MODEL_CL100K_BASE, // common shorthand
	"gpt-35-turbo":  MODEL_CL100K_BASE, // Azure deployment name
	// base
	"davinci-002": MODEL_CL100K_BASE,
	"babbage-002": MODEL_CL100K_BASE,
	// text
	"text-davinci-003": MODEL_P50K_BASE,
	"text-davinci-002": MODEL_P50K_BASE,
	"text-davinci-001": MODEL_R50K_BASE,
	"text-curie-001":   MODEL_R50K_BASE,
	"text-babbage-001": MODEL_R50K_BASE,
	"text-ada-001":     MODEL_R50K_BASE,
	"davinci":          MODEL_R50K_BASE,
	"curie":            MODEL_R50K_BASE,
	"babbage":          MODEL_R50K_BASE,
	"ada":              MODEL_R50K_BASE,
	// code
	"code-davinci-002": MODEL_P50K_BASE,
	"code-davinci-001": MODEL_P50K_BASE,
	"code-cushman-002": MODEL_P50K_BASE,
	"code-cushman-001": MODEL_P50K_BASE,
	"davinci-codex":    MODEL_P50K_BASE,
	"cushman-codex":    MODEL_P50K_BASE,
	// edit
	"text-davinci-edit-001": MODEL_P50K_EDIT,
	"code-davinci-edit-001": MODEL_P50K_EDIT,
	// embeddings
	"text-embedding-ada-002": MODEL_CL100K_BASE,
	"text-embedding-3-large": MODEL_CL100K_BASE,
	"text-embedding-3-small": MODEL_CL100K_BASE,
	// old embeddings
	"text-similarity-davinci-001":  MODEL_R50K_BASE,
	"text-similarity-curie-001":    MODEL_R50K_BASE,
	"text-similarity-babbage-001":  MODEL_R50K_BASE,
	"text-similarity-ada-001":      MODEL_R50K_BASE,
	"text-search-davinci-doc-001":  MODEL_R50K_BASE,
	"text-search-curie-doc-001":    MODEL_R50K_BASE,
	"text-search-babbage-doc-001":  MODEL_R50K_BASE,
	"text-search-ada-doc-001":      MODEL_R50K_BASE,
	"code-search-babbage-code-001": MODEL_R50K_BASE,
	"code-search-ada-code-001":     MODEL_R50K_BASE,
	// open source
	"gpt2":  "gpt2",
	"gpt-2": "gpt2", // consistent with gpt-4 naming
}

// MODEL_PREFIX_TO_ENCODING is consulted when a model name is not an exact key of
// MODEL_TO_ENCODING. Matching is longest-prefix-first (see modelPrefixes), which
// upstream tiktoken gets for free from dict insertion order and Go does not: map
// iteration is randomised, so "ft:gpt-4o-2024-08-06" would otherwise resolve to
// cl100k_base or o200k_base depending on the run.
//
// Not listed: "gpt-oss-", which upstream maps to o200k_harmony. That encoding is
// not shipped here, so gpt-oss models fall through to a "no encoding for model"
// error rather than to a silently wrong vocabulary.
var MODEL_PREFIX_TO_ENCODING = map[string]string{
	// reasoning
	"o1-":      MODEL_O200K_BASE, // e.g., o1-preview, o1-mini
	"o3-":      MODEL_O200K_BASE, // e.g., o3-mini
	"o4-mini-": MODEL_O200K_BASE,
	// chat
	"gpt-5":          MODEL_O200K_BASE,  // no trailing dash: gpt-5, gpt-5.1, gpt-5.5, gpt-5-mini, ...
	"gpt-4.5-":       MODEL_O200K_BASE,  // e.g., gpt-4.5-preview, etc.
	"gpt-4.1-":       MODEL_O200K_BASE,  // e.g., gpt-4.1-2025-04-14, etc.
	"chatgpt-4o-":    MODEL_O200K_BASE,  // e.g., chatgpt-4o-latest
	"gpt-4o-":        MODEL_O200K_BASE,  // e.g., gpt-4o-2024-05-13, etc.
	"gpt-4-":         MODEL_CL100K_BASE, // e.g., gpt-4-0314, etc., plus gpt-4-32k
	"gpt-3.5-turbo-": MODEL_CL100K_BASE, // e.g, gpt-3.5-turbo-0301, -0401, etc.
	"gpt-35-turbo-":  MODEL_CL100K_BASE, // Azure deployment name
	// fine-tuned
	"ft:gpt-4o":        MODEL_O200K_BASE,
	"ft:gpt-4":         MODEL_CL100K_BASE,
	"ft:gpt-3.5-turbo": MODEL_CL100K_BASE,
	"ft:davinci-002":   MODEL_CL100K_BASE,
	"ft:babbage-002":   MODEL_CL100K_BASE,
}

// modelPrefixes holds the keys of MODEL_PREFIX_TO_ENCODING sorted longest first,
// so that a name matched by two prefixes takes the more specific one.
var modelPrefixes []string

func init() {
	modelPrefixes = make([]string, 0, len(MODEL_PREFIX_TO_ENCODING))
	for prefix := range MODEL_PREFIX_TO_ENCODING {
		modelPrefixes = append(modelPrefixes, prefix)
	}
	sort.Slice(modelPrefixes, func(i, j int) bool {
		if len(modelPrefixes[i]) != len(modelPrefixes[j]) {
			return len(modelPrefixes[i]) > len(modelPrefixes[j])
		}
		return modelPrefixes[i] < modelPrefixes[j]
	})
}

var encodingMap map[string]*Encoding
var l *sync.RWMutex

func init() {
	encodingMap = make(map[string]*Encoding)
	l = &sync.RWMutex{}
}

type Encoding struct {
	Name           string
	PatStr         string
	MergeableRanks map[string]int
	SpecialTokens  map[string]int
	ExplicitNVocab int
}

func getEncoding(encodingName string) (*Encoding, error) {
	l.RLock()
	encoding, ok := encodingMap[encodingName]
	l.RUnlock()
	if ok {
		return encoding, nil
	}

	l.Lock()
	defer l.Unlock()
	initEncoding, err := initEncoding(encodingName)
	if err != nil {
		return nil, err
	}
	encodingMap[encodingName] = initEncoding
	return encodingMap[encodingName], nil
}

func initEncoding(encodingName string) (*Encoding, error) {
	switch encodingName {
	case MODEL_O200K_BASE:
		return o200k_base()
	case MODEL_CL100K_BASE:
		return cl100k_base()
	case MODEL_P50K_BASE:
		return p50k_base()
	case MODEL_R50K_BASE:
		return r50k_base()
	case MODEL_P50K_EDIT:
		return p50k_edit()
	default:
		return nil, errors.New("unknown encoding: " + encodingName)
	}
}

func o200k_base() (*Encoding, error) {
	ranks, err := bpeLoader.LoadTiktokenBpe(O200kBaseURL)
	if err != nil {
		return nil, err
	}
	special_tokens := map[string]int{
		ENDOFTEXT:   199999,
		ENDOFPROMPT: 200018,
	}
	pats := []string{
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`\p{N}{1,3}`,
		` ?[^\s\p{L}\p{N}]+[\r\n/]*`,
		`\s*[\r\n]+`,
		`\s+(?!\S)`,
		`\s+`,
	}
	return &Encoding{
		Name:           MODEL_O200K_BASE,
		PatStr:         strings.Join(pats, "|"),
		MergeableRanks: ranks,
		SpecialTokens:  special_tokens,
	}, nil
}

func cl100k_base() (*Encoding, error) {
	ranks, err := bpeLoader.LoadTiktokenBpe(Cl100kBaseURL)
	if err != nil {
		return nil, err
	}
	special_tokens := map[string]int{
		ENDOFTEXT:   100257,
		FIM_PREFIX:  100258,
		FIM_MIDDLE:  100259,
		FIM_SUFFIX:  100260,
		ENDOFPROMPT: 100276,
	}
	return &Encoding{
		Name:           MODEL_CL100K_BASE,
		PatStr:         `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		MergeableRanks: ranks,
		SpecialTokens:  special_tokens,
	}, nil
}

func p50k_edit() (*Encoding, error) {
	ranks, err := bpeLoader.LoadTiktokenBpe(P50kBaseURL)
	if err != nil {
		return nil, err
	}
	special_tokens := map[string]int{ENDOFTEXT: 50256, FIM_PREFIX: 50281, FIM_MIDDLE: 50282, FIM_SUFFIX: 50283}
	return &Encoding{
		Name:           MODEL_P50K_EDIT,
		PatStr:         `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
		MergeableRanks: ranks,
		SpecialTokens:  special_tokens,
	}, nil
}

func p50k_base() (*Encoding, error) {
	ranks, err := bpeLoader.LoadTiktokenBpe(P50kBaseURL)
	if err != nil {
		return nil, err
	}
	special_tokens := map[string]int{ENDOFTEXT: 50256}
	return &Encoding{
		Name:           MODEL_P50K_BASE,
		PatStr:         `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
		MergeableRanks: ranks,
		SpecialTokens:  special_tokens,
		ExplicitNVocab: 50281,
	}, nil
}

func r50k_base() (*Encoding, error) {
	ranks, err := bpeLoader.LoadTiktokenBpe(R50kBaseURL)
	if err != nil {
		return nil, err
	}
	special_tokens := map[string]int{ENDOFTEXT: 50256}
	return &Encoding{
		Name:           MODEL_R50K_BASE,
		MergeableRanks: ranks,
		PatStr:         `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
		SpecialTokens:  special_tokens,
		ExplicitNVocab: 50257,
	}, nil
}
