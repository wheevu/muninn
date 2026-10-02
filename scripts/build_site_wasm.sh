#!/usr/bin/env bash
# Builds the playground's interpreter and drops it next to the page.
#
# The commit is compiled in, so the page can say which Muninn it is running
# rather than only that it is running one.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit="$(git -C "$repo_root" rev-parse --short HEAD)"

# A tree with uncommitted changes is not that commit. The module is built from
# what is on disk, so without this the page would name a commit it does not
# match, and a reader comparing the page against the repository would find
# code the page is not running.
if [ -n "$(git -C "$repo_root" status --porcelain)" ]; then
  commit="${commit}-dirty"
fi

MUNINN_COMMIT="$commit" cargo build \
  --manifest-path "$repo_root/wasm/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --release

cp "$repo_root/wasm/target/wasm32-unknown-unknown/release/muninn_wasm.wasm" \
  "$repo_root/site/muninn.wasm"

printf 'site/muninn.wasm  %s bytes  (muninn %s)\n' \
  "$(wc -c < "$repo_root/site/muninn.wasm" | tr -d ' ')" "$commit"
