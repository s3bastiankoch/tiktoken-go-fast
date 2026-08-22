module github.com/s3bastiankoch/tiktoken-go-fast/bench

go 1.26

require (
	github.com/dlclark/regexp2 v1.10.0
	github.com/google/uuid v1.3.0
	github.com/pkoukk/tiktoken-go-loader v0.0.2
	github.com/s3bastiankoch/tiktoken-go-fast v0.0.0
	github.com/tiktoken-go/tokenizer v0.8.1
)

require github.com/dlclark/regexp2/v2 v2.5.1 // indirect

replace github.com/s3bastiankoch/tiktoken-go-fast => ../
