module github.com/s3bastiankoch/tiktoken-go-fast/bench/vsrust

go 1.21

require (
	github.com/pkoukk/tiktoken-go-loader v0.0.2
	github.com/s3bastiankoch/tiktoken-go-fast v0.0.0
)

require (
	github.com/dlclark/regexp2 v1.10.0 // indirect
	github.com/google/uuid v1.3.0 // indirect
)

replace github.com/s3bastiankoch/tiktoken-go-fast => ../../
