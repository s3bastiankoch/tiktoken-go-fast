package tiktoken

import (
	"os"
	"testing"

	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

// TestMain installs the offline loader for the whole package.
//
// Without it the suite is not hermetic: TestGetEncoding_ErrorResponseNotCached
// swaps the loader for one pointed at a local httptest server, and whatever it
// restores afterwards is what TestEncodingForModel_Names and _Prefixes run
// under. Restoring the default loader there sent those two tests to
// openaipublic.blob.core.windows.net for four vocabularies, so `go test ./...`
// silently depended on the network, and on that blob store staying up.
//
// The embedded vocabularies in tiktoken-go-loader are the same bytes, so this
// costs nothing but the dependency the test binary already has.
func TestMain(m *testing.M) {
	SetBpeLoader(tiktoken_loader.NewOfflineLoader())
	os.Exit(m.Run())
}
