# upstream

`pkoukk/tiktoken-go` at
[`200cf60`](https://github.com/pkoukk/tiktoken-go/commit/200cf60), the commit
this repository forked from, extracted verbatim. To regenerate from a checkout
of that commit:

```sh
for f in bpe.go core_bpe.go encoding.go load.go tiktoken.go; do
  sed 's/^package tiktoken$/package upstream/' <path-to-200cf60>/$f > $f
done
```

The only edit is the package clause, so that both implementations can be linked
into one binary and measured in the same process, in interleaved rounds, against
the same vocabulary. Without that they cannot share a module path, and a
cross-process comparison would reintroduce exactly the drift the paired
methodology exists to remove.

MIT, as upstream; copyright remains with the original authors. Regenerate with
the command above if the fork point ever moves.
